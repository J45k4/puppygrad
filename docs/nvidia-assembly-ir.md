# NVIDIA Assembly IR Design

Puppygrad should not lower tensors directly to final NVIDIA bytes. Mesa's Nouveau compiler shows the shape we want:

- `src/nouveau/compiler/nak/ir.rs` defines a typed instruction IR with SSA values, physical registers, predicates, memory operands, virtual copy/phi operations, instruction dependencies, and shader metadata.
- `src/nouveau/compiler/nak/api.rs` runs explicit passes: lower from a higher IR, optimize, legalize, schedule before register allocation, assign registers, lower parallel copies, schedule again, calculate instruction dependencies, gather metadata, then encode.
- `src/nouveau/compiler/nak/sm70_encode.rs` keeps SM-specific bit encoding out of the generic IR.
- `src/nouveau/compiler/nak/qmd.rs` and `src/nouveau/vulkan/nvk_cmd_dispatch.c` show that runnable compute work needs code plus launch metadata/QMD state, not only SASS instruction words.

## Pipeline

The first NVIDIA target should be SM70+ only.

```text
puppygrad graph
  -> tensor kernel IR
  -> nvidia asm SSA IR
  -> legal nvidia asm IR
  -> scheduled SSA IR
  -> physical-register asm IR
  -> scheduled machine IR with dependency bits
  -> SASS words
  -> runnable package: Nouveau QMD or CUDA cubin
```

## IR Layers

Tensor kernel IR is device-independent and should know about elementwise maps, reductions, matmul tiles, launch geometry, buffer arguments, and shapes.

NVIDIA assembly IR is device-specific and should know about:

- SM version and feature gates.
- compute local size and shared-memory footprint.
- kernel parameters with offsets, sizes, and alignment.
- SSA values before register allocation.
- GPR, uniform GPR, predicate, uniform predicate, and barrier register files.
- predicated instructions.
- global/shared/local/constant/parameter memory references.
- virtual ops such as copy and parallel copy before final lowering.
- scheduling fields: delay, yield hint, read/write barriers, wait barrier mask, reuse mask.

Machine binary packaging is a separate layer. Raw SASS words are useful for testing and disassembly, but a runnable kernel also needs launch metadata. Mesa/NVK uses uploaded shader code plus a QMD. CUDA-style execution needs a cubin/fatbin-compatible container.

## Initial Opcode Surface

Start with the minimum needed for elementwise kernels:

- `S2R` for thread/block special registers.
- integer address math: `IADD3`, `IMAD`, `SHL`.
- f32 math: `FADD`, `FMUL`, `FFMA`.
- predicates: `SETP`, predicated `BRA`, predicated `EXIT`.
- memory: global `LD`/`ST`, later shared `LD`/`ST`.
- synchronization: `BAR.SYNC`.
- virtual: `COPY`, later `PHI_SRC`, `PHI_DST`, `PAR_COPY`.

This is enough for vector add, unary ops, scalar broadcast, and simple tiled kernels once shared memory is introduced.

## Runnable Binary Target

There are two realistic packaging targets:

- Nouveau/NVK-style: SASS words plus shader metadata, shader upload layout, QMD, and push-buffer dispatch. This follows Mesa most directly.
- CUDA-style: SASS words packaged into an ELF cubin/fatbin accepted by NVIDIA's driver APIs. This is the path for proprietary-driver execution, but the cubin container details are separate from the instruction IR.

The IR should therefore produce a `MachineCode` object first and only then package it as `NouveauQmd` or `CudaCubin`.

## Current Scaffold

The Rust scaffold lives in `src/gpu/nvidia`:

- `asm.rs` defines SM70+ assembly IR types and a readable dump format.
- `binary.rs` defines machine-code and launch-package metadata.

The scaffold intentionally does not encode real SASS yet. The next step is a tiny SM80 encoder for one known kernel, then validation against `nvdisasm` or Mesa NAK test output.
