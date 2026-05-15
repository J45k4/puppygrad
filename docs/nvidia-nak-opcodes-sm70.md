# Mesa NAK NVIDIA Ops, SM70+ Opcode Fields

This table is based on Mesa's Nouveau NAK IR in `/Users/puppy/work/others/mesa/src/nouveau/compiler/nak/ir.rs` and SM70+ encoder in `/Users/puppy/work/others/mesa/src/nouveau/compiler/nak/sm70_encode.rs`.

The hex column is not a complete instruction binary. On Volta+ NVIDIA instructions are 128 bits. The final four `u32` words also include operands, predicates, modifiers, labels, reuse flags, barriers, dependency/yield bits, and SM-specific fields. Values below are Mesa's visible base opcode or opcode-form constants for SM70+ when directly encoded by `sm70_encode.rs`.

`n/a` means the op is present in NAK IR but has no direct SM70 encoder implementation in Mesa's `sm70_encode.rs`; it may be for older architectures, lowered away, or not emitted on SM70+.

| NAK op | What it does | SM70+ base opcode field(s) |
| --- | --- | --- |
| `FAdd` | f32 add. | `0x021` |
| `FFma` | f32 fused multiply-add. | `0x023` |
| `FMnMx` | f32 min/max selected by predicate. | `0x009` |
| `FMul` | f32 multiply. | `0x020` |
| `Rro` | range-reduction helper for transcendental math. | n/a |
| `MuFu` | multi-function unit operation such as reciprocal, reciprocal sqrt, sin/cos/ex2/lg2 forms. | see `MuFuOp` encoding; form-dependent |
| `FSet` | f32 compare producing a scalar value. | `0x00a` |
| `FSetP` | f32 compare producing predicate result(s). | `0x00b` |
| `FSwzAdd` | f32 swizzled add. | `0x822` |
| `FSwz` | f32 swizzle operation. | n/a |
| `DAdd` | f64 add. | `0x029` |
| `DFma` | f64 fused multiply-add. | `0x02b` |
| `DMnMx` | f64 min/max. | n/a |
| `DMul` | f64 multiply. | `0x028` |
| `DSetP` | f64 compare producing predicate result(s). | `0x02a` |
| `HAdd2` | packed half2 add. | `0x030` |
| `HFma2` | packed half2 fused multiply-add. | `0x031` |
| `HMul2` | packed half2 multiply. | `0x032` |
| `HSet2` | packed half2 compare producing scalar value. | `0x033` |
| `HSetP2` | packed half2 compare producing predicate result(s). | `0x034` |
| `Imma` | integer matrix multiply-accumulate/tensor-core op. | `0x237` |
| `Hmma` | half/mixed precision matrix multiply-accumulate/tensor-core op. | `0x23c` |
| `Ldsm` | load matrix data from shared memory for tensor-core instructions. | `0x83b` |
| `HMnMx2` | packed half2 min/max. | `0x040` |
| `BMsk` | build bit mask. | `0x01b`, `0x09b` |
| `BRev` | bit reverse. | `0x0be`, `0x101` |
| `Bfe` | bit-field extract. | n/a |
| `Flo` | find leading one / bit scan style operation. | `0x0bd`, `0x100` |
| `IAbs` | integer absolute value. | `0x013` |
| `IAdd2` | older two-source integer add. | n/a |
| `IAdd2X` | older integer add with carry/extended predicate behavior. | n/a |
| `IAdd3` | three-source integer add. | `0x010`, `0x090` |
| `IAdd3X` | three-source integer add with carry/extended predicate behavior. | `0x010`, `0x090` |
| `IDp4` | 4-lane integer dot product. | `0x026` |
| `IMad` | integer multiply-add. | `0x024`, `0x0a4` |
| `IMad64` | wide integer multiply-add helper for 64-bit products. | `0x025`, `0x0a5` |
| `IMul` | integer multiply. | n/a |
| `IMnMx` | integer min/max. | `0x017` |
| `ISetP` | integer compare producing predicate result(s). | `0x00c`, `0x08c` |
| `Lea` | integer address calculation. | `0x011`, `0x091` |
| `LeaX` | extended address calculation with predicate/carry behavior. | `0x011`, `0x091` |
| `Lop2` | two-input logical operation. | n/a |
| `Lop3` | three-input logical operation with truth-table immediate. | `0x012`, `0x092` |
| `PopC` | population count. | `0x0bf`, `0x109` |
| `Shf` | funnel shift. | `0x019`, `0x099` |
| `Shl` | shift left. | n/a |
| `Shr` | shift right. | n/a |
| `F2F` | floating-point conversion. | form-dependent in encoder |
| `F2FP` | floating-point pair/pack conversion. | `0x03e` |
| `F2I` | float-to-integer conversion. | form-dependent in encoder |
| `I2F` | integer-to-float conversion. | `0x106`, `0x112` |
| `I2I` | integer-to-integer conversion. | n/a |
| `FRnd` | floating-point rounding operation. | form-dependent in encoder |
| `Mov` | move/copy value. | `0x002`, `0xc82` |
| `Movm` | matrix/tensor-core register move. | `0x23a` |
| `Prmt` | byte/word permute. | `0x016`, `0x096` |
| `Sel` | select between values based on predicate. | `0x007`, `0x087` |
| `Sgxt` | sign extend. | `0x01a`, `0x09a` |
| `Shfl` | warp shuffle. | `0x389`, `0x589`, `0x989`, `0xf89` |
| `PLop3` | predicate logical op with three inputs/truth table. | `0x81c`, `0x89c` |
| `PSetP` | predicate set/compare operation. | n/a |
| `R2UR` | move regular register value to uniform register. | `0x2ca`, `0x3c2` |
| `Redux` | warp/subgroup reduction. | `0x3c4` |
| `Tex` | texture sample. | `0x361`, `0xb60`, `0xd61` |
| `Tld` | texture load. | `0x367`, `0xb66`, `0xd67` |
| `Tld4` | texture gather/load four components. | `0x364`, `0xb63`, `0xd64` |
| `Tmml` | texture mipmap/lod helper op. | `0x36a`, `0xb69` |
| `Txd` | texture sample/load with derivatives. | `0x36d`, `0xb6c`, `0xd6d` |
| `Txq` | texture query. | `0x370`, `0xb6f` |
| `SuLd` | surface/image load. | `0x998`, `0x99a` |
| `SuSt` | surface/image store. | `0x99c`, `0x99e` |
| `SuAtom` | surface/image atomic. | `0x394`, `0x396`, `0x3a0` |
| `SuClamp` | surface/image coordinate clamp helper. | n/a |
| `SuBfm` | surface/image bit-field mask helper. | n/a |
| `SuEau` | surface/image effective-address/update helper. | n/a |
| `IMadSp` | specialized integer multiply-add form used by surface/image addressing. | n/a |
| `SuLdGa` | surface global-address load helper. | n/a |
| `SuStGa` | surface global-address store helper. | n/a |
| `Ld` | generic/global/local/shared memory load, depending on fields. | `0x381`, `0x983`, `0x984` |
| `Ldc` | constant-buffer load. | `0x582`, `0x7ac`, `0xab9`, `0xb82`, `0xbac` |
| `LdSharedLock` | shared-memory load with lock semantics. | n/a |
| `St` | generic/global/local/shared memory store, depending on fields. | `0x386`, `0x387`, `0x388` |
| `StSCheckUnlock` | shared-memory store/check/unlock helper. | n/a |
| `Atom` | memory atomic operation. | `0x38c`, `0x38d`, `0x3a3`, `0x3a8`, `0x3a9`, `0x98e`, `0x9a6` |
| `AL2P` | attribute load to predicate. | `0x920` |
| `ALd` | attribute load. | `0x321` |
| `ASt` | attribute store. | `0x322` |
| `Ipa` | interpolate attribute. | `0x326` |
| `LdTram` | load from tram/attribute-related storage. | `0x3ad` |
| `CCtl` | cache control operation. | `0x98f` |
| `MemBar` | memory barrier. | `0x992` |
| `BClear` | barrier/control-flow mask clear. | `0x355` |
| `BMov` | move barrier/control-flow mask value. | `0x355`, `0x356` |
| `Break` | break from structured control flow. | `0x942` |
| `BSSy` | set synchronization barrier for structured control flow. | `0x945` |
| `BSync` | synchronize to control-flow barrier. | `0x941` |
| `Bra` | branch. | `0x547`, `0x947` |
| `SSy` | older set-synchronization instruction. | n/a |
| `Sync` | older synchronization/reconvergence instruction. | n/a |
| `Brk` | older break instruction. | n/a |
| `PBk` | older pre-break instruction. | n/a |
| `Cont` | older continue instruction. | n/a |
| `PCnt` | older pre-continue instruction. | n/a |
| `Exit` | terminate current thread/program. | `0x94d` |
| `WarpSync` | synchronize lanes in a warp. | `0x148` |
| `Bar` | CTA/shared-memory barrier operation. | `0x31d`, `0xb1d` |
| `TexDepBar` | texture dependency barrier. | n/a |
| `CS2R` | copy/control-special register read. | `0x805` |
| `Isberd` | ISBE read for vertex/tess/geometry IO. | `0x923` |
| `Isbewr` | ISBE write for vertex/tess/geometry IO. | `0x927` |
| `ViLd` | vertex input load. | n/a |
| `Kill` | fragment kill/discard. | `0x95b` |
| `Nop` | no operation. | `0x918` |
| `PixLd` | pixel/sample related load. | `0x925` |
| `S2R` | read special register, such as thread/block IDs. | `0x919`, `0x9c3` |
| `Vote` | warp vote operation. | `0x806`, `0x886` |
| `Match` | warp match operation. | `0x3a1` |
| `Out` | geometry/tessellation output operation. | `0x124` |
| `OutFinal` | finalize output emission. | `0x124` |

Virtual NAK IR ops are intentionally omitted from this table because they should be removed before final hardware encoding: `Undef`, `SrcBar`, `PhiSrcs`, `PhiDsts`, `Copy`, `Pin`, `Unpin`, `Swap`, `ParCopy`, `RegOut`, and `Annotate`.
