"""Regenerate the static UOp conformance corpus using the pinned tinygrad checkout.

No tensor realization, GPU initialization, or model assets are needed.
Usage: python python/tinygrad_spec_fixtures.py --tinygrad /path/to/tinygrad
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys

REVISION = "1a58c3ae9d5ff5605d81085cf1a364c113e95cf6"
ROOT = Path(__file__).resolve().parents[1]


def record(op, src=(), arg="None"):
    return dict(op=op, src=list(src), arg=arg)


def const(n):
    value = {"Bool": n} if isinstance(n, bool) else {"Int": n} if isinstance(n, int) else {"Float": struct.unpack("Q", struct.pack("d", n))[0]}
    return record("CONST", arg={"Scalar": value})


def param(slot, size=None, dtype="f32"):
    return record("PARAM", arg={"Param": dict(slot=slot, dtype=dtype, size=size)})


def cases():
    basic = [param(0, 6), const(2), const(3), record("STACK", [1, 2]), record("RESHAPE", [0, 3])]
    yield "reshape", basic
    yield "leading_expand", basic + [const(4), record("EXPAND", [4, 5])]
    for op in ("ADD", "MUL", "MAX"):
        yield "leading_reduce_" + op.lower(), basic + [record("REDUCE", [4], {"Reduce": dict(op=op, num_axes=1)})]
    yield "permute", basic + [record("PERMUTE", [4], {"Axes": [1, 0]})]
    yield "flip", basic + [record("FLIP", [4], {"Flip": [True, False]})]
    yield "bad_permute", basic + [record("PERMUTE", [4], {"Axes": [1, 1]})]
    yield "bad_flip_rank", basic + [record("FLIP", [4], {"Flip": [True]})]
    yield "bad_reduce_rank", basic + [record("REDUCE", [4], {"Reduce": dict(op="ADD", num_axes=3)})]
    yield "bad_reshape", [param(0, 6), const(5), record("RESHAPE", [0, 1])]
    yield "negative_shape", [param(0, 6), const(-6), record("RESHAPE", [0, 1])]
    yield "weak_int", [const(2)]
    yield "weak_float", [const(2.5)]
    yield "bool", [const(True)]
    yield "empty_stack", [record("STACK")]
    yield "scalar_reshape", [param(0, 1), record("STACK"), record("RESHAPE", [0, 1])]
    yield "typed_constant", [const(2.5), record("CAST", [0], {"DType": "f32"})]
    yield "weak_add", [const(2), const(3), record("ADD", [0, 1])]
    yield "program_add", [param(0), const(2.5), record("CAST", [1], {"DType": "f32"}), record("ADD", [0, 2]), record("SINK", [3])]
    yield "kernel_parameters", [param(0, 6), param(1, 12), record("SINK", [0, 1])]
    yield "bad_dtype", [param(0), param(1, dtype="i32"), record("ADD", [0, 1])]
    yield "where", [param(0), param(1), record("CMPLT", [0, 1]), record("WHERE", [2, 0, 1])]
    yield "bad_where", [param(0), record("WHERE", [0, 0, 0])]
    yield "scalar_broadcast", basic + [const(2.0), record("MUL", [4, 5])]
    yield "incompatible_broadcast", [param(0, 2), param(1, 3), record("MUL", [0, 1])]
    for op in ("SUB", "FDIV"):
        yield op.lower(), [param(0), param(1), record(op, [0, 1])]
    yield "index_rows", basic + [param(1, 2, "i32"), record("INDEX", [4, 5])]
    yield "load_rows", basic + [param(1, 2, "i32"), record("INDEX", [4, 5]), record("LOAD", [6])]
    yield "index_scalar", [param(0, 6), param(1, dtype="i32"), record("INDEX", [0, 1]), record("LOAD", [2])]
    yield "bad_index_dtype", [param(0, 6), param(1, dtype="f32"), record("INDEX", [0, 1])]
    movement = basic + [const(1), record("STACK", [5, 1]), const(4), const(6), record("STACK", [7, 8]), record("PAD", [4, 6, 9])]
    yield "pad", movement
    yield "shrink", movement + [record("SHRINK", [10, 6, 3])]
    yield "bad_pad", basic + [record("PAD", [4, 3, 3])]
    yield "bad_shrink", basic + [record("SHRINK", [4, 3, 3])]
    for op in ("ADD", "MUL", "MAX"):
        yield "empty_reduce_" + op.lower(), [param(0, 0), record("REDUCE", [0], {"Reduce": dict(op=op, num_axes=1)})]
    for op in ("NEG", "EXP2", "LOG2", "SQRT"):
        yield op.lower(), [param(0), record(op, [0])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tinygrad", required=True, type=Path)
    args = parser.parse_args()
    rev = subprocess.check_output(["git", "-C", str(args.tinygrad), "rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION:
        raise SystemExit(f"expected tinygrad {REVISION}, got {rev}")
    subprocess.run(["git", "-C", str(args.tinygrad), "diff", "--exit-code", "HEAD", "--", "tinygrad"], check=True, stdout=subprocess.DEVNULL)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.tinygrad.resolve()))
    from tinygrad.uop.ops import UOp, Ops, ParamArg
    from tinygrad.uop.spec import type_verify, spec_tensor, spec_program, spec_kernel_graph
    from tinygrad.dtype import dtypes, ConstFloat
    dtype_map = dict(f32=dtypes.float32, i32=dtypes.int32, bool=dtypes.bool, void=dtypes.void, weakint=dtypes.weakint, weakfloat=dtypes.weakfloat)
    dtype_names = {v: k for k, v in dtype_map.items()}

    def argument(arg):
        if arg == "None": return None
        kind, value = next(iter(arg.items()))
        if kind == "Param": return ParamArg(value["slot"], dtype_map[value["dtype"]], size=value["size"])
        if kind == "DType": return dtype_map[value]
        if kind in ("Axes", "Flip"): return tuple(value)
        if kind == "Reduce": return (getattr(Ops, value["op"]), value["num_axes"])
        if kind == "Scalar":
            scalar, bits = next(iter(value.items()))
            return ConstFloat(struct.unpack("d", struct.pack("Q", bits))[0]) if scalar == "Float" else bits
        raise ValueError(kind)

    corpus = []
    for name, instructions in cases():
        nodes, expected = [], []
        failed_at = None
        for i, ins in enumerate(instructions):
            try:
                node = UOp(getattr(Ops, ins["op"]), src=tuple(nodes[j] for j in ins["src"]), arg=argument(ins["arg"]))
                type_verify(node, spec_tensor)
                metadata = dict(dtype=dtype_names[node.dtype], shape=list(node._shape) if node._shape is not None else None)
            except (RuntimeError, ValueError, AssertionError, TypeError, IndexError):
                failed_at = i
                break
            nodes.append(node)
            expected.append(metadata)
        stages = {}
        if failed_at is None:
            for stage, spec in (("program", spec_program), ("kernel_graph", spec_kernel_graph)):
                try:
                    type_verify(nodes[-1], spec)
                    stages[stage] = True
                except (RuntimeError, ValueError, AssertionError, TypeError, IndexError):
                    stages[stage] = False
        corpus.append(dict(name=name, nodes=instructions, expected=expected, failed_at=failed_at, **stages))
    output = ROOT / "tests/data/compiler/tinygrad-spec.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(revision=REVISION, cases=corpus), indent=2) + "\n")
    print(f"wrote {len(corpus)} cases to {output}")


if __name__ == "__main__":
    main()
