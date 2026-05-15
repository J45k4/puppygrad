# AMD ACO Machine Instruction Map

This table is based on Mesa ACO's opcode database in `/Users/puppy/work/others/mesa/src/amd/compiler/aco_opcodes.py`.

AMD instruction encoding is grouped by **format** more than by one global opcode number. The same mnemonic can have different opcode values across `gfx6`, `gfx7`, `gfx8`, `gfx9`, `gfx10`, `gfx11`, `gfx11_7`, and `gfx12`. ACO stores those per-generation opcode values in the `Opcode(gfx6, ..., gfx12)` tuple.

ACO currently defines **1631 instruction entries** in that file, including pseudo ops. The table below is the useful compilerkit view: what each machine instruction family is for, how many ACO op entries use it, and examples.

## Encoding Formats

| Format | Count | What it is for | Example ops |
| --- | ---: | --- | --- |
| `PSEUDO` | 43 | Compiler-only ops lowered before final assembly. | `p_parallelcopy`, `p_startpgm`, `p_return`, `p_phi` |
| `PSEUDO_BRANCH` | 4 | Compiler-only branch forms before final control-flow lowering. | `p_branch`, `p_cbranch`, `p_cbranch_z`, `p_cbranch_nz` |
| `PSEUDO_BARRIER` | 1 | Compiler-only barrier form with memory/scope metadata. | `p_barrier` |
| `PSEUDO_REDUCTION` | 3 | Compiler-only subgroup reduction/scan forms. | `p_reduce`, `p_inclusive_scan`, `p_exclusive_scan` |
| `PSEUDO_CALL` | 1 | Compiler-only call abstraction. | `p_call` |
| `SOP1` | 96 | Scalar ALU/control instruction with one scalar input. Uses SGPR/SCC-style scalar state. | `s_mov_b64`, `s_brev_b64`, `s_ceil_f16`, `s_setpc_b64` |
| `SOP2` | 78 | Scalar ALU/control instruction with two scalar inputs. | `s_add_i32`, `s_addc_u32`, `s_lshl_b64`, `s_mul_hi_u32` |
| `SOPK` | 28 | Scalar instruction with inline immediate field. | `s_movk_i32`, `s_cmpk_eq_u32`, `s_call_b64` |
| `SOPP` | 53 | Scalar program-flow/wait/barrier/sendmsg/trap instructions. | `s_branch`, `s_cbranch_scc0`, `s_waitcnt`, `s_barrier`, `s_endpgm` |
| `SOPC` | 48 | Scalar compare instruction, usually writing `SCC`. | `s_cmp_lt_i32`, `s_cmp_eq_u64`, `s_cmp_gt_f32` |
| `SMEM` | 101 | Scalar memory load/store/atomic/scratch/buffer operations. | `s_load_dword`, `s_store_dwordx2`, `s_buffer_atomic_or` |
| `DS` | 168 | LDS/GDS data-share memory operations and atomics. | `ds_read_b32`, `ds_write_b32`, `ds_add_u32`, `ds_cmpst_rtn_f64` |
| `LDSDIR` | 2 | Direct LDS/parameter load forms. | `lds_param_load`, `lds_direct_load` |
| `MTBUF` | 16 | Typed buffer memory load/store. | `tbuffer_load_format_x`, `tbuffer_store_format_xyzw` |
| `MUBUF` | 83 | Untyped buffer memory load/store/atomic/cache operations. | `buffer_load_dword`, `buffer_store_dwordx4`, `buffer_atomic_add` |
| `MIMG` | 109 | Image/sample/texture/bvh operations. | `image_sample`, `image_load`, `image_atomic_umax`, `image_bvh8_intersect_ray` |
| `EXP` | 1 | Export data from shader stage. | `exp` |
| `FLAT` | 59 | Flat-address memory operations. | `flat_load_dword`, `flat_store_dword`, `flat_atomic_add` |
| `GLOBAL` | 67 | Global-address memory operations. | `global_load_dword`, `global_store_dword`, `global_atomic_add` |
| `SCRATCH` | 22 | Scratch/private memory operations. | `scratch_load_dword`, `scratch_store_dwordx4` |
| `VINTRP` | 3 | Older vector parameter interpolation. | `v_interp_p1_f32`, `v_interp_p2_f32`, `v_interp_mov_f32` |
| `VINTERP_INREG` | 6 | Newer in-register interpolation forms. | `v_interp_p10_f32_inreg`, `v_interp_p2_f16_f32_inreg` |
| `VOPD` | 17 | RDNA dual-issue packed pair of vector ops. | `v_dual_add_f32`, `v_dual_fmac_f32`, `v_dual_mov_b32` |
| `VOP1` | 104 | Vector ALU instruction with one input. | `v_mov_b32`, `v_floor_f32`, `v_cvt_i32_f64`, `v_exp_f32` |
| `VOP2` | 81 | Vector ALU instruction with two inputs. | `v_add_f32`, `v_mul_f32`, `v_add_co_u32`, `v_max_i32` |
| `VOPC` | 198 | Vector compare instruction, typically producing `VCC`/exec-related predicate state. | `v_cmp_eq_f32`, `v_cmp_lt_i32`, `v_cmpx_class_f64` |
| `VOP3` | 177 | Extended vector ALU encoding, often three inputs or extra modifiers. | `v_fma_f32`, `v_fma_f64`, `v_max3_f32`, `v_div_scale_f64` |
| `VOP3P` | 62 | Packed vector ALU / dot / matrix forms. | `v_pk_fma_f16`, `v_dot4_i32_i8`, `v_wmma_f32_16x16x16_f16` |

## Functional Classes

| Class | Count | What it means | Example ops |
| --- | ---: | --- | --- |
| `Salu` | 189 | Scalar ALU integer/bit/control work on SGPRs. | `s_add_i32`, `s_bfe_u64`, `s_nand_b32` |
| `SFPU` | 63 | Scalar floating-point operations, mostly newer architectures. | `s_add_f32`, `s_mul_f32`, `s_cvt_f32_i32` |
| `Valu32` | 480 | Main 32-bit vector ALU work on VGPRs. | `v_add_f32`, `v_mul_f32`, `v_min_f32`, `v_add_co_u32` |
| `Valu64` | 41 | 64-bit vector integer/compare/shift style work. | `v_lshlrev_b64`, `v_cmp_lt_i64` |
| `ValuFma` | 3 | Vector fused multiply-add family. | `v_fma_f32`, `v_fma_legacy_f16`, `v_fma_legacy_f32` |
| `ValuDouble` | 49 | f64 vector math. | `v_floor_f64`, `v_fract_f64`, `v_cmp_class_f64` |
| `ValuDoubleAdd` | 5 | f64 add/mul/min/max subset. | `v_add_f64`, `v_mul_f64`, `v_min_f64`, `v_max_f64` |
| `ValuDoubleConvert` | 6 | f64 conversion ops. | `v_cvt_f32_f64`, `v_cvt_f64_i32` |
| `ValuDoubleTranscendental` | 5 | f64 reciprocal/sqrt style ops. | `v_rcp_f64`, `v_rsq_f64`, `v_sqrt_f64` |
| `ValuTranscendental32` | 22 | f32/f16 reciprocal, sqrt, sin/cos, exp/log style ops. | `v_rcp_f32`, `v_sqrt_f16`, `v_exp_f32` |
| `ValuConvert32` | included in format counts | 32-bit conversion class used by ACO metadata. | conversion-style `v_cvt_*` ops |
| `ValuQuarterRate32` | 5 | Slower 32-bit integer multiply/SAD style ops. | `v_mul_lo_i32`, `v_mul_hi_u32`, `v_mqsad_u32_u8` |
| `ValuPseudoScalarTrans` | 10 | Vector encodings used for scalarized transcendental lowering. | `v_s_rcp_f32`, `v_s_sqrt_f16` |
| `WMMA` | 22 | Wave matrix multiply-accumulate operations. | `v_wmma_f32_16x16x16_f16`, `v_swmmac_f32_16x16x32_bf16` |
| `VMem` | 356 | Vector memory operations: buffer, image, flat, global, scratch. | `buffer_load_dword`, `global_store_dword`, `image_sample`, `flat_atomic_add` |
| `SMem` | 101 | Scalar memory operations. | `s_load_dword`, `s_buffer_load_dword`, `s_atomic_add` |
| `DS` | 170 | LDS/GDS data-share operations. | `ds_read_b32`, `ds_write_b32`, `ds_add_rtn_u32` |
| `Branch` | 21 | Control-flow branch/call/return forms. | `s_branch`, `s_cbranch_scc1`, `s_swappc_b64` |
| `Barrier` | 9 | Workgroup/barrier synchronization. | `s_barrier`, `s_barrier_wait`, `s_barrier_signal` |
| `Waitcnt` | 18 | Wait/dependency counter instructions. | `s_waitcnt`, `s_wait_idle`, `s_wait_loadcnt` |
| `Export` | 1 | Shader export. | `exp` |
| `Sendmsg` | 2 | Message send/halt to fixed hardware paths. | `s_sendmsg`, `s_sendmsghalt` |
| `Other` | 53 | Pseudo/miscellaneous compiler instructions. | `p_phi`, `p_parallelcopy`, `s_trap` |

## Compilerkit Takeaway

For an AMD backend, the low-level IR should not try to use one flat opcode namespace only. AMD has a strong split between:

- **SGPR/SALU** scalar instructions for uniform work and control.
- **VGPR/VALU** vector instructions for per-lane work.
- **VMEM/SMEM/DS** memory families.
- **EXEC/VCC/SCC** predicate and execution-mask state.
- **waitcnt/barrier** instructions for memory dependency correctness.

A minimal AMD compute subset for puppygrad-style kernels would start with:

- `SOPP`: `s_endpgm`, `s_waitcnt`
- `SOP1/SOP2/SOPC`: scalar moves, compares, address/uniform math
- `VOP1/VOP2/VOP3/VOPC`: vector moves, f32 add/mul/fma, compares
- `GLOBAL` or `FLAT`: global loads/stores
- `DS`: later, LDS/shared-memory loads/stores and barriers
- `VOP3P`/`WMMA`: much later, matrix/tensor-style operations
