import type { LazyNode, Op } from "./op"
import { Add, Var, pattern } from "./pattern"
import type { DType } from "./tensor"

type VarName = string

export interface PrimLowering {
    kind: "Prim"
    node: LazyNode
}

export interface BufferLowering {
    kind: "Buffer"
    name: string
    data: readonly number[]
    shape: readonly number[]
    dtype: DType
    node: LazyNode
}

export interface RangeLowering {
    kind: "Range"
    index: VarName
    extent: number | null
    inputs: ReadonlyArray<{ name: VarName; node: LazyNode }>
    body: { kind: "Add"; lhs: VarName; rhs: VarName }
    output: LazyNode
}

export type Lowering = PrimLowering | RangeLowering | BufferLowering

const sizeOfShape = (shape?: readonly number[]): number | null => {
    if (!shape) return null
    if (shape.length === 0) return 1
    return shape.reduce((a, b) => a * b, 1)
}

// Lower high-level ops into a simple range form.
// Currently: Add(x, y) => Range over output extent that performs elementwise add.
export const lower = pattern<null, Lowering>(
    [
        [
            Add(Var("a"), Var("b")),
            (_ctx, ret, { a, b }) => ({
                kind: "Range" as const,
                index: "i",
                extent: sizeOfShape((ret as any).shape),
                inputs: [
                    { name: "a", node: a! },
                    { name: "b", node: b! },
                ],
                body: { kind: "Add" as const, lhs: "a", rhs: "b" },
                output: ret,
            }),
            "add->range",
        ],
    ] as const,
    {
        onNoMatch: (_ctx, ret) => ({ kind: "Prim", node: ret }),
    }
)

export const lowerGraph = (ops: readonly Op[]): Lowering[] => {
    const res: Lowering[] = []

    for (const o of ops) {
        if (o.kind === "Const") {
            const data = (o.attrsView as any)?.data
            if (Array.isArray(data)) {
                res.push({
                    kind: "Buffer",
                    name: `buf${o.id}`,
                    data,
                    shape: o.shape,
                    dtype: o.dtype,
                    node: o,
                })
                continue
            }
        }
        res.push(lower(null, o)!)
    }

    return res
}

const shapeSize = (shape: readonly number[]) => (shape.length === 0 ? 1 : shape.reduce((a, b) => a * b, 1))

export const generateCLowered = (lowerings: readonly Lowering[], outputs: readonly LazyNode[]): string => {
    const varNames = new Map<number, string>()
    for (const l of lowerings) {
        if (l.kind === "Buffer") varNames.set(l.node.id, l.name)
        else if (l.kind === "Range") varNames.set(l.output.id, `out${l.output.id}`)
    }

    const nameFor = (node: LazyNode) => varNames.get(node.id) ?? `tmp${node.id}`

    const lines: string[] = [
        "// Generated C from lowered IR (skeleton)",
        "#include <math.h>",
        "#include <stdbool.h>",
        "#include <stdint.h>",
        "",
        "",
    ]

    for (const l of lowerings) {
        if (l.kind === "Buffer") {
            const ctype = l.dtype === "float64" ? "double" : "float"
            const flat = l.data.join(", ")
            const size = l.data.length
            lines.push(
                `// Buffer for op ${l.node.kind} (id=${l.node.id}) shape=${JSON.stringify(l.shape)}`,
                `static ${ctype} ${l.name}[${size}] = { ${flat} };`,
                ""
            )
            continue
        }

        if (l.kind === "Prim") {
            lines.push(`// Prim op ${l.node.kind} (id=${l.node.id}) not lowered; placeholder`, "")
            continue
        }

        const extent = l.extent ?? 0
        const outName = nameFor(l.output)
        lines.push(
            `// Range lower of ${l.output.kind} (id=${l.output.id})`,
            `static float ${outName}[${extent}];`,
            `static inline void kernel_${l.output.id}() {`,
            `  for (int ${l.index} = 0; ${l.index} < ${extent}; ++${l.index}) {`
        )

        switch (l.body.kind) {
            case "Add":
                {
                    const lhsNode = l.inputs.find((i) => i.name === l.body.lhs)?.node
                    const rhsNode = l.inputs.find((i) => i.name === l.body.rhs)?.node
                    const lhsName = lhsNode ? nameFor(lhsNode) : l.body.lhs
                    const rhsName = rhsNode ? nameFor(rhsNode) : l.body.rhs
                    lines.push(`    ${outName}[${l.index}] = ${lhsName}[${l.index}] + ${rhsName}[${l.index}];`)
                }
                break
            default:
                lines.push(`    // TODO: lower body ${l.body.kind}`)
                break
        }

        lines.push("  }", "}", "")
    }

    const primaryOutput = outputs[outputs.length - 1]
    const outName = primaryOutput ? nameFor(primaryOutput) : "out0"
    const outSize = primaryOutput ? shapeSize(primaryOutput.shape) : 0

    // Emit entry points
    for (const l of lowerings) {
        if (l.kind === "Range") {
            lines.push(`// Execute kernel for op id=${l.output.id}`, `static inline void run_${l.output.id}() { kernel_${l.output.id}(); }`, "")
        }
    }

    lines.push(`void entry_fill(float* out) {`)
    for (const l of lowerings) {
        if (l.kind === "Range") {
            lines.push(`  run_${l.output.id}();`)
        }
    }
    if (outSize > 0) {
        lines.push(`  for (int i = 0; i < ${outSize}; ++i) out[i] = ${outName}[i];`)
    }
    lines.push(`}`, "", `int entry_size() { return ${outSize}; }`)

    lines.push("", "int main() { return 0; }")

    return lines.join("\n")
}
