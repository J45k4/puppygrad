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
    | { kind: "Const"; value: number; dtype?: DType; data?: readonly number[] }
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

export type NodeId = number

export class Op {
    private static counter = 0

    public readonly id: NodeId
    public readonly shape: Shape
    public readonly dtype: DType
    private readonly _attrs?: Omit<OpAttrs, "kind">

    constructor(
        public kind: OpKind = "Const",
        public inputs: readonly Op[] = [],
        public attrs?: Omit<OpAttrs, "kind">,
        opts?: { id?: NodeId; shape?: Shape; dtype?: DType }
    ) {
        this.id = opts?.id ?? Op.counter++
        this.shape = opts?.shape ?? []
        this.dtype = opts?.dtype ?? "float32"
        this._attrs = attrs
    }

    public get op(): OpKind {
        return this.kind
    }

    public get attrsView(): Omit<OpAttrs, "kind"> | undefined {
        return this._attrs
    }

    private static makeOpts(left: Op, right?: Op, opts?: { shape?: Shape; dtype?: DType; id?: NodeId }) {
        return {
            id: opts?.id,
            shape: opts?.shape ?? left.shape,
            dtype: opts?.dtype ?? (right ? right.dtype : left.dtype),
        }
    }

    public static add(left: Op, right: Op, opts?: { shape?: Shape; dtype?: DType; id?: NodeId }): Op {
        return new Op("Add", [left, right], undefined, this.makeOpts(left, right, opts))
    }

    public static sub(left: Op, right: Op, opts?: { shape?: Shape; dtype?: DType; id?: NodeId }): Op {
        return new Op("Sub", [left, right], undefined, this.makeOpts(left, right, opts))
    }

    public static mul(left: Op, right: Op, opts?: { shape?: Shape; dtype?: DType; id?: NodeId }): Op {
        return new Op("Mul", [left, right], undefined, this.makeOpts(left, right, opts))
    }

    public static sum(input: Op, axis?: number, opts?: { shape?: Shape; dtype?: DType; id?: NodeId }): Op {
        const attrs: Omit<OpAttrs, "kind"> = axis === undefined ? {} : { axis }
        return new Op("Sum", [input], attrs as any, {
            id: opts?.id,
            shape: opts?.shape ?? input.shape,
            dtype: opts?.dtype ?? input.dtype,
        })
    }

    public topo(): Op[] {
        return topo([this])
    }
}

/**
 * Collect a DAG of ops starting from `roots` and return them in
 * topologically-sorted order (inputs before consumers).
 */
export const topo = (roots: readonly Op[]): Op[] => {
    const visited = new Set<NodeId>()
    const ordered: Op[] = []

    function dfs(node: Op) {
        if (visited.has(node.id)) return
        visited.add(node.id)
        for (const inp of node.inputs) dfs(inp)
        ordered.push(node)
    }

    for (const root of roots) dfs(root)
    return ordered
}

// LazyNode is the runtime Op (graph node)
export type LazyNode = Op

const dtypeToCType = (dtype?: DType): string => {
    switch (dtype) {
        case "bool":
            return "bool"
        case "int32":
            return "int"
        case "float16":
            return "uint16_t" // placeholder for half type
        case "float64":
            return "double"
        case "float32":
        default:
            return "float"
    }
}

const varName = (op: Op) => `v${op.id}`

const emitOpToC = (op: Op): string[] => {
    const inputs = op.inputs.map(varName)
    const attrs = op.attrs as any
    const lines: string[] = []

    switch (op.kind) {
        case "Const": {
            const ctype = dtypeToCType(attrs?.dtype)
            lines.push(`${ctype} ${varName(op)} = ${attrs?.value ?? 0};`)
            break
        }
        case "Full": {
            const ctype = dtypeToCType(attrs?.dtype)
            lines.push(
                `// Full(${attrs?.value ?? 0}) shape=${JSON.stringify(attrs?.shape ?? [])}`,
                `${ctype} ${varName(op)} = ${attrs?.value ?? 0}; // scalar placeholder`
            )
            break
        }
        case "Uniform": {
            const ctype = dtypeToCType(attrs?.dtype)
            lines.push(
                `// Uniform${attrs?.low !== undefined ? ` low=${attrs.low}` : ""}${attrs?.high !== undefined ? ` high=${attrs.high}` : ""} shape=${JSON.stringify(attrs?.shape ?? [])}`,
                `${ctype} ${varName(op)} = 0; // TODO: random generation`
            )
            break
        }
        case "Add":
            lines.push(`auto ${varName(op)} = ${inputs[0]} + ${inputs[1]};`)
            break
        case "Sub":
            lines.push(`auto ${varName(op)} = ${inputs[0]} - ${inputs[1]};`)
            break
        case "Mul":
            lines.push(`auto ${varName(op)} = ${inputs[0]} * ${inputs[1]};`)
            break
        case "Sum":
            lines.push(`// Sum axis=${attrs?.axis ?? "all"}`)
            lines.push(`auto ${varName(op)} = sum(${inputs[0]}); // TODO: expand over shape`)
            break
        default:
            lines.push(`// TODO: codegen for ${op.kind}`)
            lines.push(`auto ${varName(op)} = ${inputs[0] ?? "0"};`)
    }

    return lines
}

/**
 * Very small C-ish code generator: takes a topo-sorted op list and emits a string.
 * This is intentionally minimal and leaves shape/loop details as TODOs.
 */
export const generateC = (ops: readonly Op[]): string => {
    const lines: string[] = [
        "// Generated C (skeleton)",
        "#include <math.h>",
        "#include <stdbool.h>",
        "#include <stdint.h>",
        "",
        "int main() {",
    ]

    for (const op of ops) {
        for (const ln of emitOpToC(op)) {
            lines.push(`  ${ln}`)
        }
    }

    lines.push("  return 0;")
    lines.push("}")

    return lines.join("\n")
}
