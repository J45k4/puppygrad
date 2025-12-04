import { expect, test } from "bun:test"
import { Op } from "./op"
import { lower } from "./lower"

test("lowers Add to a range form", () => {
    const a = new Op("Const", [], undefined, { shape: [3], dtype: "float32" })
    const b = new Op("Const", [], undefined, { shape: [3], dtype: "float32" })
    const add = new Op("Add", [a, b], undefined, { shape: [3], dtype: "float32" })

    const res = lower(null, add)
    console.log(res)
    expect(res).not.toBeNull()
    expect(res!.kind).toBe("Range")
    if (res && res.kind === "Range") {
        expect(res.extent).toBe(3)
        expect(res.inputs.map((i) => i.name)).toEqual(["a", "b"])
        expect(res.inputs[0]?.node).toBe(a)
        expect(res.inputs[1]?.node).toBe(b)
        expect(res.body).toEqual({ kind: "Add", lhs: "a", rhs: "b" })
        expect(res.output).toBe(add)
    }
})

test("non-matching ops fall back to Prim", () => {
    const neg = new Op("Neg")
    const res = lower(null, neg)
    expect(res).not.toBeNull()
    expect(res!.kind).toBe("Prim")
    if (res && res.kind === "Prim") {
        expect(res.node).toBe(neg)
    }
})
