
import { topo } from "./op"
import { generateCLowered, lowerGraph } from "./lower"
import type { Tensor } from "./tensor"
import { ClangProgram } from "./clang"

/**
    High level "realize" entry point:
    1) Collect the op DAG from the tensor(s)
    2) Topologically sort
    3) (Optionally) lower/pattern-match — skipped for now
    4) Generate C code and log it
*/
export const realize = (tensors: readonly Tensor<any, any>[]) => {
    const roots = tensors.map((t) => t.op)
    const ordered = topo(roots)
    const lowered = lowerGraph(ordered)
    const c = generateCLowered(lowered, roots)
    if (process.env.DEBUG === "4") {
        console.log("=== Generated C ===")
        console.log(c)
        console.log("===================")
    }
    const outShape = roots[roots.length - 1]?.shape ?? []
    const outSize = outShape.length === 0 ? 1 : outShape.reduce((a, b) => a * b, 1)
    const outDType = roots[roots.length - 1]?.dtype ?? "float32"
    const program = new ClangProgram(c, { outputLength: outSize, outputDtype: outDType })
    return program.run()
}
