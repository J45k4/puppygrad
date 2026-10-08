"""Measure production HIP Qwen generation across context lengths, through the maximum."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import selectors
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('models/qwen3-0.6b'))
    parser.add_argument('--source', type=Path, default=Path('examples/qwen3_cached.pup'))
    parser.add_argument('--output-dir', type=Path, default=Path('.cache/hip-context'))
    parser.add_argument('--prompt-lengths', type=int, nargs='+')
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.tokens < 2 or args.warmups < 1 or args.repeats < 1:
        parser.error('require at least two output tokens, one warmup and one measured request')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = args.model_dir.resolve()
    with (model / 'model.safetensors').open('rb') as checkpoint:
        header_size = int.from_bytes(checkpoint.read(8), 'little')
        header = json.loads(checkpoint.read(header_size))
    weight_dtypes = sorted({'BF16' if tensor['dtype'] == 'BF16' else 'F32'
                           for name, tensor in header.items() if name != '__metadata__'})
    fixture = out / 'fixed-length-checkpoint'
    fixture.mkdir(exist_ok=True)
    config = json.loads((model / 'config.json').read_text())
    # Only the provider stop policy changes. All checkpoint tensors are unchanged.
    config.pop('eos_token_id', None)
    (fixture / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    for name in ['model.safetensors', 'tokenizer.json']:
        link = fixture / name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(model / name)
    plan = json.loads(subprocess.check_output([
        'target/release/puppygrad', 'llm', 'capacity', str(args.source),
        '--model-dir', str(model), '--device', 'hip:0', '--json'], text=True))
    maximum = plan['max_context_tokens']
    lengths = args.prompt_lengths or sorted(set(
        [24] + [n-args.tokens for n in [512, 1024, 2048, 4096, 8192, 16384, 32768] if n <= maximum]
        + [maximum-args.tokens]))
    if any(n < 1 or n+args.tokens > maximum for n in lengths):
        parser.error('prompt lengths plus output tokens must fit the planned context')
    deps = Path('target/release/deps')
    runner = out / 'runner'
    build = ['rustc', '--edition=2021', '-O', str(Path(__file__).with_suffix('.rs')),
             '-L', f'dependency={deps}', '-o', str(runner)]
    for name in ['puppygrad', 'serde_json', 'tokenizers']:
        lib = max(deps.glob(f'lib{name}-*.rlib'), key=lambda p: p.stat().st_mtime)
        build += ['--extern', f'{name}={lib}']
    subprocess.run(build, check=True)
    vram_devices = [p for p in Path('/sys/class/drm').glob('card*/device')
                    if (p/'mem_info_vram_total').exists()]
    vram_device = vram_devices[0] if len(vram_devices) == 1 else None
    def vram_used():
        return int((vram_device/'mem_info_vram_used').read_text()) if vram_device else None
    result = {'date':time.strftime('%Y-%m-%d'), 'backend':'HIP', 'source':str(args.source),
              'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
              'model_dir':str(model), 'dtype':'+'.join(weight_dtypes), 'kv_dtype':'F32', 'build':'release',
              'capacity_estimate':plan, 'generated_tokens':args.tokens, 'warmups':args.warmups,
              'repeats':args.repeats, 'prompt_lengths':lengths,
              'method':'Fresh production LLM FFI worker per row; synthetic repeated chat token IDs; greedy sampling; EOS stop disabled in a separate config; original tensors unchanged. Warm timers include reset, chunked prefill, host sampling, uploads, host logits and streaming callbacks. Model loading and compilation excluded. Decode throughput excludes first token.',
              'vram_total_bytes':int((vram_device/'mem_info_vram_total').read_text()) if vram_device else None,
              'vram_usage_scope':'Total device use, including desktop and other processes; not just this model.',
              'rows':[]}
    version = Path('/opt/rocm/.info/version')
    if version.exists():
        result['rocm_version'] = version.read_text().strip()
    def save():
        (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        with (out/'results.csv').open('w') as csv:
            csv.write('prompt_tokens,total_tokens,kv_capacity,first_token_ms,decode_tokens_per_second,request_ms,request_tokens_per_second,device_vram_gib,status\n')
            for r in result['rows']:
                if 'error' in r:
                    csv.write(f"{r['prompt_tokens']},{r['total_tokens']},,,,,,,failed\n")
                else:
                    csv.write(','.join(str(r[k]) for k in ['prompt_tokens','total_tokens','kv_capacity','first_token_ms','decode_tokens_per_second','request_ms','request_tokens_per_second','device_vram_gib'])+',passed\n')
    save()
    print(json.dumps({'maximum':maximum,'prompt_lengths':lengths,'plan':plan}), flush=True)
    for length in lengths:
        row = {'prompt_tokens':length,'total_tokens':length+args.tokens,
               'kv_capacity':min(1 << (max(512,length+args.tokens-1)-1).bit_length(), maximum)}
        result['active_row'] = row
        log = (out/f'context-{length}.stderr').open('w')
        worker = subprocess.Popen([str(runner), str(fixture), str(args.source)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        selector = selectors.DefaultSelector()
        selector.register(worker.stdout, selectors.EVENT_READ)
        peak_vram = vram_used() or 0
        def call(command):
            nonlocal peak_vram
            worker.stdin.write(json.dumps(command)+'\n'); worker.stdin.flush()
            waiting = time.perf_counter()
            next_notice = waiting+30
            while True:
                peak_vram = max(peak_vram, vram_used() or 0)
                if selector.select(timeout=1):
                    line = worker.stdout.readline()
                    if not line:
                        raise RuntimeError(f'worker exited with {worker.poll()}; see context-{length}.stderr')
                    response = json.loads(line)
                    if 'error' in response:
                        raise RuntimeError(response['error'])
                    return response
                now = time.perf_counter()
                if now >= next_notice:
                    print(json.dumps({'prompt_tokens':length,'phase':row.get('phase','setup'),'waiting_s':round(now-waiting)}), flush=True)
                    next_notice = now+30
        try:
            setup_started = time.perf_counter()
            row['model_info'] = call({'info':True})
            row['setup_ms'] = (time.perf_counter()-setup_started)*1000
            if length+args.tokens > row['model_info']['context_length']:
                raise RuntimeError('live provider context dropped below requested size')
            base = call({'prompt':'Explain why the sky is blue in one short sentence.'})['tokens']
            tokens = (base*((length+len(base)-1)//len(base)))[:length]
            command = {'tokens':tokens,'count':args.tokens}
            expected = None
            row['warmup_samples'] = []
            for i in range(args.warmups):
                row['phase'] = f'warmup {i+1}/{args.warmups}'
                sample = call(command)
                assert len(sample['tokens']) == len(sample['token_ms']) == args.tokens
                expected = sample['tokens'] if expected is None else expected
                assert sample['tokens'] == expected, 'warmup greedy tokens changed'
                row['warmup_samples'].append(sample)
                print(json.dumps({'prompt_tokens':length,'phase':row['phase'],'request_ms':sample['elapsed_ms']}), flush=True)
            row['samples'] = []
            for i in range(args.repeats):
                row['phase'] = f'measured {i+1}/{args.repeats}'
                sample = call(command)
                assert sample['tokens'] == expected, 'repeated greedy tokens changed'
                assert len(sample['tokens']) == len(sample['token_ms']) == args.tokens
                row['samples'].append(sample)
                save()
                print(json.dumps({'prompt_tokens':length,'phase':row['phase'],'request_ms':sample['elapsed_ms']}), flush=True)
            samples = row['samples']
            decode = [latency for s in samples for latency in s['token_ms'][1:]]
            row.update({'first_token_ms':statistics.median(s['token_ms'][0] for s in samples),
                        'decode_token_median_ms':statistics.median(decode),
                        'decode_tokens_per_second':len(decode)*1000/sum(decode),
                        'request_ms':statistics.median(s['elapsed_ms'] for s in samples),
                        'request_tokens_per_second':statistics.median(args.tokens*1000/s['elapsed_ms'] for s in samples),
                        'device_vram_gib':(vram_used() or 0)/2**30,
                        'peak_device_vram_gib':peak_vram/2**30,
                        'status':'passed'})
            print(json.dumps({k:v for k,v in row.items() if k not in ['samples','warmup_samples']}), flush=True)
        except (Exception, KeyboardInterrupt) as error:
            row.update({'error':str(error) or type(error).__name__,'status':'failed'})
            print(json.dumps({'prompt_tokens':length,'error':str(error)}), flush=True)
            if isinstance(error, KeyboardInterrupt):
                worker.terminate()
                raise
        finally:
            selector.close()
            worker.stdin.close()
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.kill(); worker.wait()
            log.close()
            device = re.search(r'compiled HIP: .* contractions, (.*?); source:',
                               (out/f'context-{length}.stderr').read_text())
            if device:
                result['gpu_name'] = device.group(1)
            row['worker_exit_code'] = worker.returncode
            row.pop('phase', None)
            result['rows'].append(row)
            result.pop('active_row', None)
            save()
    with (model/'model.safetensors').open('rb') as checkpoint:
        result['checkpoint_sha256'] = hashlib.file_digest(checkpoint,'sha256').hexdigest()
    save()
    if any(r['status'] != 'passed' for r in result['rows']):
        raise RuntimeError('one or more contexts failed; see results.json')


if __name__ == '__main__':
    main()
