//! Portable C vector microkernels with bounded, shared weight panels.
//! Packing is invocation-local: mutable parameters never use stale cached data.
#[path = "cpu_gemm_blocked.rs"]
mod blocked;
const PANEL_N: usize = 64;
const MAX_PACKED_BYTES: usize = 8 * 1024 * 1024;
// Small panels stay serial: a second pool barrier can cost more than the copy.
const PARALLEL_PACK_BYTES: usize = 256 * 1024;

pub(super) struct Kernel {
    pub worker: String,
    pub dispatch: String,
    pub packed_bytes: usize,
    pub phase_timing_kind: &'static str,
}

pub(super) fn generate(
    id: usize,
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
    ar: &str,
    br: &str,
    contiguous_columns: bool,
    destination: &str,
    epilogue: &str,
    fields: &str,
    locals: &str,
    vars: &str,
    profile: bool,
) -> Kernel {
    if m >= 16 && n >= 16 && k >= 1024 && !contiguous_columns {
        return blocked::generate(
            id,
            m,
            n,
            k,
            batch_count,
            ar,
            br,
            destination,
            epilogue,
            fields,
            locals,
            vars,
            profile,
        );
    }
    let packed_bytes = k
        .checked_mul(PANEL_N * 4)
        .filter(|&bytes| {
            m >= 16 && n >= 16 && k >= 128 && !contiguous_columns && bytes <= MAX_PACKED_BYTES
        })
        .unwrap_or(0);
    let packed = packed_bytes > 0;
    let mut worker = format!("typedef struct {{ {fields} float *packed; size_t batch,col,width; }} context{id};\nstatic void tile{id}(const pup_pool *pool,size_t id) {{ const context{id} *ctx=pool->context;\n{locals}\n");
    if packed {
        worker += &format!("const size_t columns=(ctx->width+PUP_TILE_N-1)/PUP_TILE_N;\nfor(size_t tile=id;tile<{}ULL*columns;tile+=pool->threads) {{\nconst size_t batch=ctx->batch,row=(tile/columns)*4,localcol=(tile%columns)*PUP_TILE_N,col=ctx->col+localcol;\n",m.div_ceil(4));
    } else {
        worker += &format!("const size_t columns=({n}+PUP_TILE_N-1)/PUP_TILE_N,tiles={}ULL*columns;\nfor(size_t tile=id;tile<tiles*{batch_count};tile+=pool->threads) {{\nconst size_t batch=tile/tiles,local=tile%tiles,row=(local/columns)*4,col=(local%columns)*PUP_TILE_N;\n",m.div_ceil(4));
    }
    worker += &format!(
        "const size_t rows={m}-row<4?{m}-row:4,cols={n}-col<PUP_TILE_N?{n}-col:PUP_TILE_N;\n"
    );
    for group in 0..2 {
        for row in 0..4 {
            worker += &format!("pup_vec c{row}_{group}=(pup_vec){{0}};\n");
        }
    }
    worker += &format!("for(size_t r=0;r<{k};r++) {{\n");
    if !packed {
        worker += &format!(
            "float bv[PUP_TILE_N]; for(size_t jj=0;jj<PUP_TILE_N;jj++) bv[jj]=jj<cols?{br}:0;\n"
        );
    }
    for group in 0..2 {
        let pointer = if packed {
            format!("ctx->packed+r*{PANEL_N}+localcol+{group}*PUP_VEC_LANES")
        } else {
            format!("bv+{group}*PUP_VEC_LANES")
        };
        worker += &format!("pup_vec b{group}; memcpy(&b{group},{pointer},sizeof(b{group}));\n");
    }
    for row in 0..4 {
        let a = ar.replace("ii", &row.to_string());
        worker += &format!("const float a{row}=rows>{row}?{a}:0;\n");
    }
    for group in 0..2 {
        for row in 0..4 {
            worker += &format!("c{row}_{group}=c{row}_{group}+a{row}*b{group};\n");
        }
    }
    worker += "}\nfloat acc[4][PUP_TILE_N];\n";
    for group in 0..2 {
        for row in 0..4 {
            worker += &format!("memcpy(acc[{row}]+{group}*PUP_VEC_LANES,&c{row}_{group},sizeof(c{row}_{group}));\n");
        }
    }
    worker += &format!("for(size_t ii=0;ii<rows;ii++) for(size_t jj=0;jj<cols;jj++) {{\nconst size_t out_index=batch*{m}*{n}+(row+ii)*{n}+col+jj;\n{epilogue};\n}}\n}}\n}}\n");
    if packed {
        // Each worker owns whole reduction-row blocks, including tail zeroes.
        // The compute dispatch starts only after every packing worker finishes.
        worker += &format!("static void pack_rows{id}(const context{id} *ctx,size_t first,size_t last) {{\n{locals}\nconst size_t batch=ctx->batch,col=ctx->col,width=ctx->width;\nfloat *packed=ctx->packed;\nfor(size_t block=first;block<last;block++) {{\nconst size_t base=block*32,end=base+32<{k}?base+32:{k};\nfor(size_t jj=0;jj<width;jj++) for(size_t r=base;r<end;r++) packed[r*{PANEL_N}+jj]={br};\nif(width<{PANEL_N}) for(size_t r=base;r<end;r++) for(size_t jj=width;jj<{PANEL_N};jj++) packed[r*{PANEL_N}+jj]=0;\n}}\n}}\n");
        if packed_bytes >= PARALLEL_PACK_BYTES {
            let blocks = k.div_ceil(32);
            worker += &format!("static void pack_tile{id}(const pup_pool *pool,size_t id) {{\nconst size_t base={blocks}/pool->threads,extra={blocks}%pool->threads;\nconst size_t begin=id*base+(id<extra?id:extra),end=begin+base+(id<extra);\npack_rows{id}(pool->context,begin,end);\n}}\n");
        }
    }
    let init = if vars.is_empty() {
        String::new()
    } else {
        format!("{vars},")
    };
    let mut dispatch = format!("context{id} ctx={{{init}NULL,0,0,{n}}};\n");
    if packed {
        dispatch += &format!("float *packed=malloc({packed_bytes}ULL); if(!packed) return 1;\nctx.packed=packed;\nfor(size_t batch=0;batch<{batch_count};batch++) for(size_t col=0;col<{n};col+={PANEL_N}) {{\nconst size_t width={n}-col<{PANEL_N}?{n}-col:{PANEL_N};\nctx.batch=batch;ctx.col=col;ctx.width=width;\n");
        if profile {
            dispatch += "uint64_t packing_start=pup_clock_ns();\n";
        }
        if packed_bytes >= PARALLEL_PACK_BYTES {
            dispatch += &format!("pup_dispatch(pool,pack_tile{id},&ctx);\n");
        } else {
            dispatch += &format!("pack_rows{id}(&ctx,0,{});\n", k.div_ceil(32));
        }
        if profile {
            dispatch += &format!("stats[{}]+=pup_clock_ns()-packing_start;\n", 4 + id * 4 + 2);
        }
    }
    if profile {
        dispatch += "uint64_t compute_start=pup_clock_ns();\n";
    }
    dispatch += &format!("pup_dispatch(pool,tile{id},&ctx);\n");
    if profile {
        dispatch += &format!("stats[{}]+=pup_clock_ns()-compute_start;\n", 4 + id * 4 + 3);
    }
    if packed {
        dispatch += "}\nfree(packed);\n";
    }
    Kernel {
        worker,
        dispatch,
        packed_bytes,
        phase_timing_kind: "wall",
    }
}
