import os, sys, json, time, statistics, random, subprocess, gc, hashlib, itertools
from pathlib import Path
import numpy as np
import argparse
parser=argparse.ArgumentParser(description="Compare Qwen3 Puppygrad CUDA with native tinygrad.llm math.")
parser.add_argument('--model-dir',type=Path,default=Path('models/qwen3-0.6b'))
parser.add_argument('--source',type=Path,default=Path('examples/qwen3.pup'))
parser.add_argument('--tinygrad-root',type=Path)
parser.add_argument('--output-dir',type=Path,default=Path('.cache/qwen3-compare'))
parser.add_argument('--warmups',type=int,default=3)
parser.add_argument('--runs',type=int,default=11)
args=parser.parse_args()
if args.warmups<3 or args.runs<1:parser.error('at least 3 warmups and 1 run required')
retained='state(' in args.source.read_text()
pg_name='puppygrad_cached' if retained else 'puppygrad_full_prefix'
ROOT=args.output_dir.resolve()
ROOT.mkdir(parents=True,exist_ok=True)
PIN='1a58c3ae9d5ff5605d81085cf1a364c113e95cf6'
TINY=args.tinygrad_root or Path('.cache/compare-tinygrad')/('tinygrad-'+PIN)
sys.path.insert(0,str(TINY.resolve()))
os.environ.update(DEV='CUDA', BEAM='0', DEBUG='0', XDG_CACHE_HOME=str(ROOT/'xdg-cache'))
os.environ.setdefault('NVRTC_PATH',str(Path('.cache/cuda-toolchain/nvidia/cuda_nvrtc/lib/libnvrtc.so.12').resolve()))
os.environ.setdefault('PUPPYGRAD_NVRTC',os.environ['NVRTC_PATH'])
from tinygrad import Tensor, dtypes, TinyJit, Device, GlobalCounters
from tinygrad.llm.model import Transformer, TransformerConfig, precompute_freqs_cis
from tinygrad.nn.state import load_state_dict
CFG=json.loads((args.model_dir/'config.json').read_text())
config=TransformerConfig(num_blocks=CFG['num_hidden_layers'], dim=CFG['hidden_size'], hidden_dim=CFG['intermediate_size'], n_heads=CFG['num_attention_heads'], n_kv_heads=CFG['num_key_value_heads'], norm_eps=CFG['rms_norm_eps'], vocab_size=CFG['vocab_size'], head_dim=CFG['head_dim'], rope_theta=CFG['rope_theta'], rope_dim=CFG['head_dim'], v_head_dim=CFG['head_dim'], max_context=512, qk_norm=CFG['head_dim'])
model=Transformer(config)
path=(args.model_dir/'model.safetensors')
with path.open('rb') as f:
    hsize=int.from_bytes(f.read(8),'little');header=json.loads(f.read(hsize))
raw=np.memmap(path,mode='r',dtype=np.uint8)
map_tail={'input_layernorm':'attn_norm','post_attention_layernorm':'ffn_norm','self_attn.q_proj':'attn_q','self_attn.k_proj':'attn_k','self_attn.v_proj':'attn_v','self_attn.o_proj':'attn_output','self_attn.q_norm':'attn_q_norm','self_attn.k_norm':'attn_k_norm','mlp.gate_proj':'ffn_gate','mlp.up_proj':'ffn_up','mlp.down_proj':'ffn_down'}
state={};weight_bytes=0
started=time.perf_counter()
for name,meta in sorted(header.items()):
    if name in ('__metadata__','lm_head.weight'):continue
    assert meta['dtype']=='BF16'
    lo,hi=meta['data_offsets'];array=(raw[8+hsize+lo:8+hsize+hi].view(np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(meta['shape'])
    if name=='model.embed_tokens.weight':key='token_embd.weight'
    elif name=='model.norm.weight':key='output_norm.weight'
    else:
        pieces=name.split('.');key=f'blk.{pieces[2]}.'+map_tail['.'.join(pieces[3:-1])]+'.weight'
    state[key]=Tensor(array,device='CUDA',dtype=dtypes.float32).realize()
    weight_bytes+=array.nbytes
state['output.weight']=state['token_embd.weight']
load_state_dict(model,state,verbose=False,consume=True)
# Native tinygrad defaults to half KV even with float weights. Select float KV
# explicitly for parity with Puppygrad; keep all actual model math unchanged.
for block in model.blk:
    block.cache_kv=Tensor.zeros(2,1,config.n_kv_heads,config.max_context,config.head_dim,dtype=dtypes.float32,device='CUDA').realize()
    block.freqs_cis=precompute_freqs_cis(config.rope_dim,config.max_context,config.rope_theta,device='CUDA').realize()
Device['CUDA'].synchronize()
print('native tinygrad Qwen3 weights loaded',weight_bytes,'bytes in',time.perf_counter()-started,'s',flush=True)
deps=Path('target/release/deps')
compile_args=['rustc','--edition=2021','-O',str(Path(__file__).with_name('runner.rs')),'-L','dependency='+str(deps),'-o',str(ROOT/'runner')]
for name in ['puppygrad','serde_json','tokenizers']:
    library=max(deps.glob(f'lib{name}-*.rlib'),key=lambda p:p.stat().st_mtime)
    compile_args+=['--extern',f'{name}={library}']
subprocess.run(compile_args,check=True)
runner=subprocess.Popen([str(ROOT/'runner'),str(args.model_dir),str(args.source),'.cache/pup/cuda'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
def command(**kwargs):
    runner.stdin.write(json.dumps(kwargs)+'\n');runner.stdin.flush();line=runner.stdout.readline()
    if not line:raise RuntimeError('Puppygrad exited: '+str(runner.poll()))
    return json.loads(line)
def pup(tokens,**kwargs):return command(tokens=list(map(int,tokens)),**kwargs)
def forward(tokens):
    x=model.token_embd(tokens).float()
    for block in model.blk:x=block(x,0)
    return model.output(model.output_norm(x[:,-1:]))[:,-1,:].realize()
def tiny(fn,tokens):return fn(Tensor(np.array(tokens,dtype=np.int32).reshape(1,-1),device='CUDA')).numpy().reshape(-1)
def stats(values):return dict(median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),samples_ms=values)
WARMUPS=args.warmups;REPEATS=args.runs
base=command(prompt='Explain why the sky is blue in one short sentence.')['tokens']
print('chat prompt tokens',len(base),base,flush=True)
with path.open('rb') as checkpoint_file:
    checkpoint_sha=hashlib.file_digest(checkpoint_file,'sha256').hexdigest()
result={'checkpoint_sha256':checkpoint_sha,'model':'Qwen/Qwen3-0.6B','model_revision':'c1899de289a04d12100db370d81485cdf75e47ca','tinygrad_revision':subprocess.run(['git','-C',str(TINY),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip() if (TINY/'.git').exists() else (PIN if TINY.name.endswith(PIN) else 'unknown'),'tinygrad_implementation':'tinygrad.llm.model.Transformer','device':'NVIDIA GeForce RTX 2070','dtype':'F32','kv_dtype':'F32','tinygrad_max_context':config.max_context,'tinygrad_chunk_size':32,'tinygrad_beam':0,'weights_bytes':weight_bytes,'warmups':WARMUPS,'repeats':REPEATS,'prompt_tokens':base,'rows':[], 'puppygrad_source':str(args.source),'puppygrad_retained':retained,'notes':['Both implementations use identical BF16 checkpoint values expanded to F32, with tied embedding/output storage.','Full-prefix forwards start at position zero; no previous-token cache reuse.','Full-prefix Puppygrad times Executable.run including token upload and host logits; source parsing/compilation excluded.','Tinygrad times actual Transformer blocks in TinyJit including token upload and host logits.','Tinygrad native KV storage overridden to F32 for equal arithmetic; its default is F16.','Compilation and GPU warmup excluded from measured samples.','Generation compares actual Puppygrad Model.infer FFI, including CPU sampling and callbacks, with native tinygrad Transformer.generate including GPU sampling and token fetch.']}
def save(): (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
rng=random.Random(42)
try:
    for n in [len(base),64,128]:
        tokens=(base*((n+len(base)-1)//len(base)))[:n]
        fn=TinyJit(forward)
        start=time.perf_counter();expected=tiny(fn,tokens);first=(time.perf_counter()-start)*1000
        print('tiny first full-prefix',n,first,'ms',flush=True)
        initial=pup(tokens,save=str(ROOT/f'pup-logits-{n}.f32'))
        actual=np.fromfile(ROOT/f'pup-logits-{n}.f32',dtype=np.float32)
        error=float(np.abs(expected-actual).max());rms=float(np.sqrt(np.mean((expected-actual)**2)))
        print('parity',n,'max',error,'rms',rms,'argmax',int(actual.argmax()),int(expected.argmax()),flush=True)
        assert error<.003 and int(actual.argmax())==int(expected.argmax()),(n,error)
        for _ in range(WARMUPS):tiny(fn,tokens);pup(tokens)
        samples={'puppygrad':[],'tinygrad':[]}
        for rep in range(REPEATS):
            order=list(samples);rng.shuffle(order)
            for name in order:
                if name=='puppygrad':elapsed=pup(tokens)['elapsed_ms']
                else:
                    start=time.perf_counter_ns();tiny(fn,tokens);elapsed=(time.perf_counter_ns()-start)/1e6
                samples[name].append(elapsed)
        GlobalCounters.reset();tiny(fn,tokens)
        tinyprofile={'kernel_count':GlobalCounters.kernel_count,'estimated_flops':GlobalCounters.global_ops,'estimated_memory_bytes':GlobalCounters.global_mem,'allocated_bytes_cuda':GlobalCounters.mem_used_per_device['CUDA']}
        pp=pup(tokens)['profile'];assert pp['run_allocations']==0 and pp['run_input_uploaded_bytes']==n*4,pp
        row={'sequence_length':n,'logits_max_abs_error':error,'logits_rms_error':rms,'argmax':int(actual.argmax()),'puppygrad_compile_ms':initial['compile_ms'],'tinygrad_first_forward_ms':first,'timings':{k:stats(v) for k,v in samples.items()},'puppygrad_profile':pp,'tinygrad_profile':tinyprofile}
        result['rows'].append(row);save();print('row',n,{k:round(v['median_ms'],3) for k,v in row['timings'].items()},flush=True)
        fn.reset();del fn;gc.collect()
    # Validate real greedy text, comparing full-prefix Puppygrad against native
    # tinygrad generate (retained KV). Warm both paths before decode timings.
    print('preparing native cached generation',flush=True)
    model._cached_tokens=[]
    generator=model.generate(base.copy(),chunk_size=32,temperature=0.)
    tiny_ids=list(itertools.islice(generator,8));generator.close()
    pup_ids=[]
    print('native greedy IDs',tiny_ids,flush=True)
    if not retained:
        for _ in range(8):
            call=pup(base+pup_ids)
            pup_ids.append(call['argmax'])
            print('Puppygrad prefix prepared',len(base)+len(pup_ids)-1,'compile_ms',call['compile_ms'],flush=True)
    else:
        call=pup(base)
        prefill_profile=call['profile']
        pup_ids.append(call['argmax'])
        for _ in range(7):
            call=pup([pup_ids[-1]],retain=True)
            pup_ids.append(call['argmax'])
        result['puppygrad_generation_profile']={'first_token':prefill_profile,'decode_token':call['profile']}
    print('greedy parity',pup_ids,tiny_ids,flush=True)
    assert pup_ids==tiny_ids,(pup_ids,tiny_ids)
    result['generation']={'token_ids':pup_ids,'text':command(decode=pup_ids)['text']}
    def tg_generation():
        model._cached_tokens=[]
        gen=model.generate(base.copy(),chunk_size=32,temperature=0.)
        times=[];ids=[]
        for _ in range(8):
            start=time.perf_counter_ns();ids.append(next(gen));times.append((time.perf_counter_ns()-start)/1e6)
        gen.close();assert ids==pup_ids;return times
    # Use the real model FFI for generation, including host sampling, token
    # binding, callbacks and provider overhead. Close the forward worker first
    # so the two Puppygrad instances never duplicate resident weights on the GPU.
    runner.stdin.close();runner.wait(timeout=30)
    runner=subprocess.Popen([str(ROOT/'runner'),str(args.model_dir),str(args.source),'.cache/pup/cuda','ffi'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    def pg_generation():
        call=command(tokens=base)
        assert call['tokens']==pup_ids,call
        return call['token_ms']
    for _ in range(WARMUPS):tg_generation();pg_generation()
    gs={pg_name:[],'tinygrad_cached':[]}
    for rep in range(REPEATS):
        order=list(gs);rng.shuffle(order)
        for name in order:gs[name].append(tg_generation() if name=='tinygrad_cached' else pg_generation())
    result['generation']['timings']={name:{'first_token':stats([x[0] for x in seq]),'decode_token':stats([statistics.mean(x[1:]) for x in seq]),'eight_tokens':stats([sum(x) for x in seq])} for name,seq in gs.items()}
    model._cached_tokens=[]
    gen=model.generate(base.copy(),chunk_size=32,temperature=0.)
    counts={}
    for name in ['first_token','decode_token']:
        GlobalCounters.reset();next(gen)
        counts[name]={'kernel_count':GlobalCounters.kernel_count,'estimated_flops':GlobalCounters.global_ops,'estimated_memory_bytes':GlobalCounters.global_mem}
    gen.close()
    result['generation']['tinygrad_profile']=counts
    save();print('generation',result['generation'],flush=True)
finally:
    runner.stdin.close();runner.wait(timeout=30)
