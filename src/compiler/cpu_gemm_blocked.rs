//! Private cache panels let workers finish a macro tile without K-loop barriers.
use super::Kernel;

const BLOCK_K: usize = 512;
const BLOCK_M: usize = 128;
const BLOCK_N: usize = 16;
const MAX_WORKERS: usize = 16;

pub(super) fn generate(
    id: usize,
    m: usize,
    n: usize,
    k: usize,
    batches: usize,
    ar: &str,
    br: &str,
    destination: &str,
    epilogue: &str,
    fields: &str,
    locals: &str,
    vars: &str,
    profile: bool,
) -> Kernel {
    let row_tiles = m.div_ceil(BLOCK_M);
    let columns = n.div_ceil(BLOCK_N);
    let tasks = row_tiles * columns * batches;
    // Never reserve more storage than the old full-K panel for small K.
    let slots = MAX_WORKERS.min(tasks).min(k * 4 / BLOCK_K);
    let packed_bytes = slots * BLOCK_K * BLOCK_N * 4;
    let timers = if profile {
        format!("uint64_t packing[{slots}],compute[{slots}];")
    } else {
        String::new()
    };
    let mut worker = format!("enum {{ K_BLOCK{id}={BLOCK_K},M_BLOCK{id}={BLOCK_M},N_BLOCK{id}={BLOCK_N} }};\ntypedef struct {{ {fields} float *packed; size_t workers; {timers} }} context{id};\nstatic void tile{id}(const pup_pool *pool,size_t id) {{\ncontext{id} *ctx=(context{id}*)pool->context;\nif(id>=ctx->workers) return;\n{locals}\nfloat *packed=ctx->packed+id*K_BLOCK{id}*N_BLOCK{id};\n");
    if profile {
        worker += "uint64_t packing_ns=0,compute_ns=0;\n";
    }
    worker += &format!("for(size_t task=id;task<{tasks}ULL;task+=ctx->workers) {{\nconst size_t batch=task/({row_tiles}*{columns}),local=task%({row_tiles}*{columns});\nconst size_t rowbase=(local/{columns})*M_BLOCK{id},panelcol=(local%{columns})*N_BLOCK{id};\nconst size_t height={m}-rowbase<M_BLOCK{id}?{m}-rowbase:M_BLOCK{id},width={n}-panelcol<N_BLOCK{id}?{n}-panelcol:N_BLOCK{id};\nfor(size_t begin=0;begin<{k};begin+=K_BLOCK{id}) {{\nconst size_t end=begin+K_BLOCK{id}<{k}?begin+K_BLOCK{id}:{k};\n{{ const size_t col=panelcol;\n");
    if profile {
        worker += "uint64_t started=pup_clock_ns();\n";
    }
    worker += &format!("for(size_t base=begin;base<end;base+=32) {{\nconst size_t stop=base+32<end?base+32:end;\nfor(size_t jj=0;jj<width;jj++) for(size_t r=base;r<stop;r++) packed[(r-begin)*N_BLOCK{id}+jj]={br};\nif(width<N_BLOCK{id}) for(size_t r=base;r<stop;r++) for(size_t jj=width;jj<N_BLOCK{id};jj++) packed[(r-begin)*N_BLOCK{id}+jj]=0;\n}}\n");
    if profile {
        worker += "packing_ns+=pup_clock_ns()-started;\n";
    }
    worker += "}\n";
    if profile {
        worker += "uint64_t started=pup_clock_ns();\n";
    }
    worker += &format!("for(size_t localrow=0;localrow<height;localrow+=4) for(size_t localcol=0;localcol<width;localcol+=PUP_TILE_N) {{\nconst size_t row=rowbase+localrow,col=panelcol+localcol,rows=height-localrow<4?height-localrow:4,cols=width-localcol<PUP_TILE_N?width-localcol:PUP_TILE_N;\nfloat acc[4][PUP_TILE_N];\n");
    for group in 0..2 {
        for row in 0..4 {
            worker += &format!("pup_vec c{row}_{group}=(pup_vec){{0}};\n");
        }
    }
    worker += &format!("if(begin) {{\nfor(size_t ii=0;ii<4;ii++) for(size_t jj=0;jj<PUP_TILE_N;jj++) acc[ii][jj]=ii<rows&&jj<cols?{destination}[batch*{m}*{n}+(row+ii)*{n}+col+jj]:0;\n");
    for group in 0..2 {
        for row in 0..4 {
            worker += &format!("memcpy(&c{row}_{group},acc[{row}]+{group}*PUP_VEC_LANES,sizeof(c{row}_{group}));\n");
        }
    }
    worker += "}\nfor(size_t r=begin;r<end;r++) {\n";
    for group in 0..2 {
        worker += &format!("pup_vec b{group};memcpy(&b{group},packed+(r-begin)*N_BLOCK{id}+localcol+{group}*PUP_VEC_LANES,sizeof(b{group}));\n");
    }
    for row in 0..4 {
        worker += &format!(
            "const float a{row}=rows>{row}?{}:0;\n",
            ar.replace("ii", &row.to_string())
        );
    }
    for group in 0..2 {
        for row in 0..4 {
            worker += &format!("c{row}_{group}=c{row}_{group}+a{row}*b{group};\n");
        }
    }
    worker += "}\n";
    for group in 0..2 {
        for row in 0..4 {
            worker += &format!("memcpy(acc[{row}]+{group}*PUP_VEC_LANES,&c{row}_{group},sizeof(c{row}_{group}));\n");
        }
    }
    worker += &format!("for(size_t ii=0;ii<rows;ii++) for(size_t jj=0;jj<cols;jj++) {{\nconst size_t out_index=batch*{m}*{n}+(row+ii)*{n}+col+jj;\nif(end=={k}) {{ {epilogue}; }} else {destination}[out_index]=acc[ii][jj];\n}}\n}}\n");
    if profile {
        worker += "compute_ns+=pup_clock_ns()-started;\n";
    }
    worker += "}\n}\n";
    if profile {
        worker += "ctx->packing[id]=packing_ns;ctx->compute[id]=compute_ns;\n";
    }
    worker += "}\n";
    let init = if vars.is_empty() {
        String::new()
    } else {
        format!("{vars},")
    };
    let counters = if profile { ",{0},{0}" } else { "" };
    let mut dispatch = format!("float *packed=malloc({slots}ULL*K_BLOCK{id}*N_BLOCK{id}*4);if(!packed) return 1;\ncontext{id} ctx={{{init}packed,pool->threads<{slots}?pool->threads:{slots}{counters}}};\npup_dispatch(pool,tile{id},&ctx);\n");
    if profile {
        // Work overlaps across threads: report the average worker phase time,
        // while the enclosing kernel counter still measures total wall time.
        dispatch += &format!("uint64_t packing=0,compute=0;\nfor(size_t i=0;i<ctx.workers;i++) {{packing+=ctx.packing[i];compute+=ctx.compute[i];}}\nstats[{}]+=packing/ctx.workers;stats[{}]+=compute/ctx.workers;\n",4+id*4+2,4+id*4+3);
    }
    dispatch += "free(packed);\n";
    Kernel {
        worker,
        dispatch,
        packed_bytes,
        phase_timing_kind: "worker_average",
    }
}
