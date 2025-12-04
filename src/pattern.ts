import type { DType } from "./tensor"
import type { LazyNode, OpKind } from "./op"

// --- Pattern IR --------------------------------------------------------------

export type Pat = PatVar | PatNode

export interface PatVar {
    kind: "Var"
    name: string
}

export interface PatNode {
    kind: "Node"
    op: OpKind
    inputs: readonly Pat[]
    attrs?: Record<string, unknown>
}

export type Bindings = Record<string, LazyNode>

// --- Pattern constructors (native TS) ---------------------------------------

export const Var = <const N extends string>(name: N): PatVar => ({ kind: "Var", name })

export const Const = (value: number, opts?: { dtype?: DType }): PatNode => ({
    kind: "Node",
    op: "Const",
    inputs: [],
    attrs: { value, ...(opts ?? {}) },
})

export const Op = (op: OpKind, ...inputs: Pat[]): PatNode => ({ kind: "Node", op, inputs })

// Common helpers (add as needed)
export const Add = (a: Pat, b: Pat): PatNode => ({ kind: "Node", op: "Add", inputs: [a, b] })
export const Sub = (a: Pat, b: Pat): PatNode => ({ kind: "Node", op: "Sub", inputs: [a, b] })
export const Mul = (a: Pat, b: Pat): PatNode => ({ kind: "Node", op: "Mul", inputs: [a, b] })
export const Div = (a: Pat, b: Pat): PatNode => ({ kind: "Node", op: "Div", inputs: [a, b] })
export const Neg = (a: Pat): PatNode => ({ kind: "Node", op: "Neg", inputs: [a] })

export const Cast = (a: Pat, to: DType): PatNode => ({
    kind: "Node",
    op: "Cast",
    inputs: [a],
    attrs: { to },
})

export const Sum = (a: Pat, opts?: { axis?: number | readonly number[]; keepDims?: boolean }): PatNode => ({
    kind: "Node",
    op: "Sum",
    inputs: [a],
    attrs: { ...(opts ?? {}) },
})

export const MatMul = (a: Pat, b: Pat, opts?: { transA?: boolean; transB?: boolean }): PatNode => ({
    kind: "Node",
    op: "MatMul",
    inputs: [a, b],
    attrs: { ...(opts ?? {}) },
})

export const Reshape = (a: Pat, shape: readonly number[]): PatNode => ({
    kind: "Node",
    op: "Reshape",
    inputs: [a],
    attrs: { shape },
})

export const Transpose = (a: Pat, perm?: readonly number[]): PatNode => ({
    kind: "Node",
    op: "Transpose",
    inputs: [a],
    attrs: perm ? { perm } : undefined,
})

// --- Matching ----------------------------------------------------------------

function sameNode(a: LazyNode, b: LazyNode): boolean {
    // Prefer id when available, otherwise reference equality
    return (a as any).id !== undefined && (b as any).id !== undefined ? (a as any).id === (b as any).id : a === b
}

function attrsMatch(node: LazyNode, pat: PatNode): boolean {
    if (!pat.attrs) return true
    const nAttrs = (node as any).attrs ?? {}
    for (const [k, v] of Object.entries(pat.attrs)) {
        if (!(k in nAttrs)) return false
        const nv = nAttrs[k]
        // v0 compare: supports primitives + arrays
        if (Array.isArray(v) || Array.isArray(nv)) {
            if (JSON.stringify(v) !== JSON.stringify(nv)) return false
        } else if (nv !== v) {
            return false
        }
    }
    return true
}

function matchPat(node: LazyNode, pat: Pat, bindings: Bindings): boolean {
    if (pat.kind === "Var") {
        const prev = bindings[pat.name]
        if (!prev) {
            bindings[pat.name] = node
            return true
        }
        return sameNode(prev, node)
    }

    if (node.op !== pat.op) return false
    if (!attrsMatch(node, pat)) return false
    if (node.inputs.length !== pat.inputs.length) return false

    for (let i = 0; i < pat.inputs.length; i++) {
        const subPat = pat.inputs[i]
        if (!subPat) return false
        const subNode = node.inputs[i] as any
        if (!matchPat(subNode, subPat, bindings)) return false
    }
    return true
}

export function tryMatch(root: LazyNode, lhs: Pat): Bindings | null {
    const b: Bindings = {}
    return matchPat(root, lhs, b) ? b : null
}

// --- Generic “cases” runner --------------------------------------------------
//
// This is the generic thing you asked for: it’s NOT tied to gradients.
// It just: match → call handler → return result.

export type CaseLHS = Pat | readonly Pat[]

// Tuple form: [lhs, fn, optionalDescription]
export type Case<Ctx, R> = readonly [
    lhs: CaseLHS,
    fn: (ctx: Ctx, ret: LazyNode, b: Bindings) => R,
    desc?: string,
]

export interface PatternRunnerOptions<Ctx, R> {
    // If provided and no pattern matches, this is called.
    // If omitted, runner returns null when no match.
    onNoMatch?: (ctx: Ctx, ret: LazyNode) => R
}

export function pattern<Ctx, R>(cases: readonly Case<Ctx, R>[], opts?: PatternRunnerOptions<Ctx, R>) {
    return (ctx: Ctx, ret: LazyNode): R | null => {
        for (const [lhs, fn] of cases) {
            const alts = Array.isArray(lhs) ? lhs : [lhs]
            for (const p of alts) {
                const b = tryMatch(ret, p)
                if (b) return fn(ctx, ret, b)
            }
        }
        return opts?.onNoMatch ? opts.onNoMatch(ctx, ret) : null
    }
}
