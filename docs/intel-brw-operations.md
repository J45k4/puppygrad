# Intel Mesa Compiler Operations

This table is based on Mesa's Intel compiler in `/Users/puppy/work/others/mesa/src/intel/compiler`.

There are two useful layers:

- **BRW EU hardware opcodes** in `brw/brw_eu.c` and `brw/brw_eu_defines.h`. These are closest to Intel GPU machine instructions.
- **BRW shader logical opcodes** in `brw/brw_eu_defines.h`. These are compiler IR/logical ops that lower to one or more hardware instructions or send messages.
- **Jay opcodes** in `jay/jay_opcodes.py`, a newer Intel compiler IR layer in this checkout.

## BRW EU Hardware Opcodes

The hardware opcode table lives in `/Users/puppy/work/others/mesa/src/intel/compiler/brw/brw_eu.c` as `opcode_descs[]`. Hardware opcode numbers vary by generation; the table below uses the names Mesa exposes.

| Group | Operations |
| --- | --- |
| Invalid / no-op / sync | `illegal`, `nop`, `sync`, `wait` |
| Moves / select | `mov`, `movi`, `smov`, `sel`, `csel` |
| Boolean / bitwise | `not`, `and`, `or`, `xor`, `bfn` |
| Shifts / rotates | `shr`, `shl`, `asr`, `ror`, `rol` |
| Compare | `cmp`, `cmpn` |
| Bit-field / bit count | `bfrev`, `bfe`, `bfi1`, `bfi2`, `lzd`, `fbh`, `fbl`, `cbit` |
| Structured control flow | `if`, `else`, `endif`, `while`, `break`, `cont`, `halt`, `goto`, `join` |
| Branch / call / return | `jmpi`, `brd`, `brc`, `calla`, `call`, `ret` |
| Messages / memory / sampler gateway | `send`, `sendc`, `sends`, `sendsc` |
| Math gateway | `math` |
| Arithmetic / rounding | `add`, `mul`, `avg`, `frc`, `rndu`, `rndd`, `rnde`, `rndz`, `mac`, `mach`, `macl`, `addc`, `subb`, `add3`, `mad`, `madm`, `lrp` |
| Dot / packed / matrix | `dp4`, `dph`, `dp3`, `dp2`, `dp4a`, `line`, `pln`, `dpas`, `srnd` |

Full hardware-op list:

```text
illegal sync mov sel movi not and or xor bfn shr shl smov asr ror rol
cmp cmpn csel bfrev bfe bfi1 bfi2 jmpi brd if brc else endif do while
break cont halt calla call ret goto join wait send sendc sends sendsc math
add mul avg frc rndu rndd rnde rndz mac mach lzd fbh fbl cbit addc subb
add3 macl dp4 srnd dph dp3 dp2 dp4a line dpas pln mad lrp madm nop
```

`do` is marked as a pseudo opcode in the BRW table.

## BRW Shader Logical Opcodes

These are compiler-level operations in `brw_eu_defines.h`. Some lower to hardware ALU, some to `send` messages, and some are pseudo/logical operations used before lowering.

| Group | Operations |
| --- | --- |
| Framebuffer | `FS_OPCODE_FB_WRITE_LOGICAL`, `FS_OPCODE_FB_READ_LOGICAL` |
| Transcendentals / math lowering | `SHADER_OPCODE_RCP`, `RSQ`, `SQRT`, `EXP2`, `LOG2`, `POW`, `SIN`, `COS`, `INT_QUOTIENT`, `INT_REMAINDER` |
| Sends / sampler / memory messages | `SHADER_OPCODE_SEND`, `SEND_GATHER`, `SAMPLER`, `MEMORY_FENCE`, `MEMORY_LOAD_LOGICAL`, `MEMORY_STORE_LOGICAL`, `MEMORY_ATOMIC_LOGICAL` |
| Payload / packing | `SHADER_OPCODE_LOAD_PAYLOAD`, `FS_OPCODE_PACK`, `SHADER_OPCODE_LOAD_ATTRIBUTE_PAYLOAD`, `SHADER_OPCODE_SCRATCH_HEADER` |
| Undefined / register utility | `SHADER_OPCODE_UNDEF`, `MOV_INDIRECT`, `MOV_RELOC_IMM`, `READ_ARCH_REG`, `LOAD_REG` |
| Mode control | `SHADER_OPCODE_RND_MODE`, `FLOAT_CONTROL_MODE` |
| URB / attributes | `SHADER_OPCODE_URB_READ_LOGICAL`, `URB_WRITE_LOGICAL` |
| Subgroup / lane operations | `FIND_LIVE_CHANNEL`, `FIND_LAST_LIVE_CHANNEL`, `LOAD_LIVE_CHANNELS`, `BROADCAST`, `SHUFFLE`, `REDUCE`, `INCLUSIVE_SCAN`, `EXCLUSIVE_SCAN`, `VOTE_ANY`, `VOTE_ALL`, `VOTE_EQUAL`, `BALLOT`, `SEL_EXEC`, `QUAD_SWAP`, `READ_FROM_LIVE_CHANNEL`, `READ_FROM_CHANNEL`, `QUAD_SWIZZLE`, `CLUSTER_BROADCAST`, `LOAD_SUBGROUP_INVOCATION` |
| Interlock / barrier / flow | `SHADER_OPCODE_INTERLOCK`, `HALT_TARGET`, `BARRIER`, `FLOW` |
| Fragment derivatives / coordinates | `FS_OPCODE_DDX_COARSE`, `DDX_FINE`, `DDY_COARSE`, `DDY_FINE`, `PIXEL_X`, `PIXEL_Y` |
| Constants / interpolation | `FS_OPCODE_UNIFORM_PULL_CONSTANT_LOAD`, `VARYING_PULL_CONSTANT_LOAD_LOGICAL`, `INTERPOLATE_AT_SAMPLE`, `INTERPOLATE_AT_SHARED_OFFSET`, `INTERPOLATE_AT_PER_SLOT_OFFSET` |
| Small helpers | `FS_OPCODE_PACK_HALF_2x16_SPLIT`, `SHADER_OPCODE_MULH`, `ISUB_SAT`, `USUB_SAT` |
| Ray tracing / BTD | `SHADER_OPCODE_BTD_SPAWN_LOGICAL`, `BTD_RETIRE_LOGICAL`, `RT_OPCODE_TRACE_RAY_LOGICAL` |
| Spill/fill | `SHADER_OPCODE_LSC_FILL`, `LSC_SPILL` |

## Jay IR Opcodes

The newer `jay` compiler layer defines a smaller typed opcode set in `/Users/puppy/work/others/mesa/src/intel/compiler/jay/jay_opcodes.py`.

| Group | Operations |
| --- | --- |
| Bitwise | `and`, `or`, `xor`, `not`, `bfn` |
| Integer / float arithmetic | `add`, `add3`, `avg`, `mad`, `mac`, `max`, `min`, `mul`, `mul_high`, `mul_32x16`, `mul_32`, `dp4a_uu`, `dp4a_ss`, `dp4a_su` |
| Shifts / rotates / bit ops | `asr`, `bfe`, `bfi1`, `bfi2`, `bfrev`, `cbit`, `fbh`, `fbl`, `lzd`, `rol`, `ror`, `shl`, `shr` |
| Compare / select / convert / round | `cmp`, `cvt`, `sel`, `csel`, `rndd`, `rndz`, `rnde`, `frc`, `modifier`, `mov`, `mov_imm64` |
| Math | `math` with `inv`, `log`, `exp`, `sqrt`, `rsq`, `sin`, `cos` |
| Control flow | `brd`, `illegal`, `goto`, `join`, `if`, `else`, `endif`, `while`, `break`, `cont`, `call`, `calla`, `jmpi`, `ret`, `loop_once` |
| Send / sync / scheduling | `send`, `sync`, `schedule_barrier` |
| Relocation / preload / deswizzle | `reloc`, `preload`, `deswizzle`, `deswizzle_odd`, `deswizzle_even` |
| Lane / pixel helpers | `lane_id_8`, `lane_id_expand`, `extract_byte_per_8lanes`, `shr_odd_subspans_by_4`, `and_u32_u16`, `expand_quad`, `offset_packed_pixel_coords`, `extract_layer`, `quad_swizzle`, `shuffle`, `broadcast_imm` |
| Phi / SSA utilities | `phi_src`, `phi_dst`, `unit_test`, `undef`, `cast_canonical_to_flag`, `zero_flag` |

## Compilerkit Takeaway

Intel's backend is different from NVIDIA and AMD in one important way: many memory, sampler, URB, ray tracing, and synchronization operations are expressed as **send messages**. A useful Intel machine-near IR should model:

- EU ALU ops such as `add`, `mul`, `mad`, `cmp`, `sel`, shifts, and bitfield ops.
- Structured control-flow opcodes such as `if`, `else`, `endif`, `while`, `break`, `cont`.
- `send`/`sendc` messages with SFID/message descriptors for memory, sampler, URB, LSC, and other fixed-function paths.
- Register regions and execution size, because Intel EU instructions operate over SIMD regions rather than only scalar operands.
- Scoreboard/scheduling sync on newer generations.
