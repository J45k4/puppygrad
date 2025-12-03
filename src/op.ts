// puppygrad lazy op IR (version 0)
//
// Goal: keep the frontend `Tensor` ergonomic while making the internal engine:
// - backend-agnostic (CPU/WebGPU/CUDA/ROCm)
// - optimizable (fusion, CSE, scheduling)
// - shape/dtype-aware
//
// Think of this file as the "vocabulary" of computation. You can always build
// higher-level conveniences (softmax, layernorm, conv, attention) as composites
// and later fuse them.

import type { DType } from "./tensor"

export type Shape = readonly number[]

// Small helper types. Keep IR serializable (plain objects).
export type Axis = number | readonly number[]
export type Permutation = readonly number[]

// ---- Op kind taxonomy ------------------------------------------------------
//
// Keep this list intentionally minimal in v0.
// Prefer:
//  - a small number of orthogonal primitives
//  - everything else as a composite/fused op later

export type OpKind =
    // Creation / constants
    | "Const" // scalar const (number in JS) or small literal
    | "Full" // tensor filled with a scalar
    | "Uniform" // random uniform (optionally low/high)

    // Unary elementwise (shape-preserving)
    | "Neg"
    | "Abs"
    | "Exp"
    | "Log"
    | "Sqrt"
    | "Rsqrt"
    | "Tanh"
    | "Sigmoid"
    | "Relu"
    | "Gelu" // can be composite initially
    | "Cast" // dtype conversion

    // Binary elementwise (broadcasting)
    | "Add"
    | "Sub"
    | "Mul"
    | "Div"
    | "Pow"
    | "Maximum"
    | "Minimum"

    // Comparisons / logical (broadcasting)
    | "Eq"
    | "Ne"
    | "Lt"
    | "Le"
    | "Gt"
    | "Ge"
    | "Where" // ternary select: cond ? a : b

    // Reductions
    | "Sum"
    | "MaxReduce"
    | "MinReduce"

    // Linear algebra
    | "MatMul" // (batched) matmul; start with 2D then extend

    // Movement / view ops (no compute when possible)
    | "Reshape"
    | "Transpose" // general permute
    | "Squeeze"
    | "Unsqueeze"
    | "Slice" // basic slicing in each dim
    | "Concat" // concat along axis
    | "BroadcastTo" // explicit broadcast/expand
    | "Contiguous" // materialize into compact contiguous layout

// ---- Attribute payloads ----------------------------------------------------

type AttrfulKind =
    | "Const"
    | "Full"
    | "Uniform"
    | "Cast"
    | "Sum"
    | "MaxReduce"
    | "MinReduce"
    | "MatMul"
    | "Reshape"
    | "Transpose"
    | "Squeeze"
    | "Unsqueeze"
    | "Slice"
    | "Concat"
    | "BroadcastTo"
    | "Contiguous"

export type OpAttrs =
    | { kind: "Const"; value: number; dtype?: DType }
    | { kind: "Full"; shape: Shape; value: number; dtype?: DType }
    | { kind: "Uniform"; shape: Shape; low?: number; high?: number; dtype?: DType }

    | { kind: "Cast"; to: DType }

    | { kind: "Sum"; axis?: Axis; keepDims?: boolean }
    | { kind: "MaxReduce"; axis?: Axis; keepDims?: boolean }
    | { kind: "MinReduce"; axis?: Axis; keepDims?: boolean }

    | { kind: "MatMul"; transA?: boolean; transB?: boolean }

    | { kind: "Reshape"; shape: Shape }
    | { kind: "Transpose"; perm?: Permutation } // default: reverse for 2D convenience
    | { kind: "Squeeze"; axis?: Axis }
    | { kind: "Unsqueeze"; axis: Axis }
    | {
          kind: "Slice"
          // per-dimension [start, end, step]; undefined -> full range
          slices: ReadonlyArray<readonly [number | undefined, number | undefined, number | undefined]>
      }
    | { kind: "Concat"; axis: number }
    | { kind: "BroadcastTo"; shape: Shape }
    | { kind: "Contiguous" }

    // The majority of ops have no attrs.
    | { kind: Exclude<OpKind, AttrfulKind> }

// ---- Graph node ------------------------------------------------------------

export type NodeId = string

export interface LazyNode {
    id: NodeId
    op: OpKind
    inputs: readonly LazyNode[]

    // These are the key invariants. They should be
    // - inferable from inputs + attrs
    // - cached for fast checks
    shape: Shape
    dtype: DType

    // Operation-specific attributes.
    // (Keep as plain object for easy hashing/serialization.)
    attrs?: Omit<OpAttrs, "kind">
}

// ---- Roadmap notes ---------------------------------------------------------
//
// 1) Fusion-friendly v0 primitives:
//    - elementwise unary/binary
//    - reductions
//    - matmul
//    - movement/view
//
// 2) Composites (build from primitives first, then add fused kernels):
//    - softmax = exp(x - max(x)) / sum(exp(...))
//    - layernorm/rmsnorm
//    - attention blocks
//
// 3) Future ops (don’t add until you need them):
//    - conv2d, pool, gather/scatter, pad, dropout
//    - advanced indexing
//    - quantize/dequantize


export class Op {
    constructor(public kind: OpKind = "Const", public inputs: readonly Op[] = [], public attrs?: Omit<OpAttrs, "kind">) {}

    public static add(left: Op, right: Op): Op {
        return new Op("Add", [left, right])
    }

    public static sub(left: Op, right: Op): Op {
        return new Op("Sub", [left, right])
    }

    public static mul(left: Op, right: Op): Op {
        return new Op("Mul", [left, right])
    }

    public static sum(input: Op, axis?: number): Op {
        const attrs: Omit<OpAttrs, "kind"> = axis === undefined ? {} : { axis }
        return new Op("Sum", [input], attrs as any)
    }
}
