import { expect, test } from "bun:test"
import { ones, tensor, uniform, uniformRange, zeros, zeroes } from "./tensor"

function flattenDeep(x: any): number[] {
    if (!Array.isArray(x)) return [x]
    const out: number[] = []
    for (const v of x) out.push(...flattenDeep(v))
    return out
}

test("add tensors", async () => {
    const a = tensor([1, 2, 3])
    const b = tensor([1, 2, 3])
    const c = await a.add(b).data
    expect(c).toEqual([2, 4, 6])
})

test("sub tensors", async () => {
    const a = tensor([5, 7, 9])
    const b = tensor([1, 2, 3])
    const c = await a.sub(b).data
    expect(c).toEqual([4, 5, 6])
})

test("sum tensor", async () => {
    const a = tensor([1, 2, 3, 4])
    const c = await a.sum().data
    expect(c).toEqual([10])
})

test("sum tensor along axis", async () => {
    const a = tensor([[1, 2], [3, 4]])
    const c = await a.sum(0).data
    expect(c).toEqual([4, 6])
})

test("transpose tensor", async () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]])
    const c = await a.transpose().data
    expect(c).toEqual([[1, 4], [2, 5], [3, 6]])
})

test("test broadcasting add", async () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]])
    const b = tensor([10, 20, 30])
    const c = await a.add(b).data
    expect(c).toEqual([[11, 22, 33], [14, 25, 36]])
})

test("test broadcasting sub", async () => {
    const a = tensor([[10, 20, 30], [40, 50, 60]])
    const b = tensor([1, 2, 3])
    const c = await a.sub(b).data
    expect(c).toEqual([[9, 18, 27], [39, 48, 57]])
})

test("matmul two tensors", async () => {
    const a = tensor([[2, 2], [2, 2]])
    const b = tensor([[2, 2], [2, 2]])
    const c = await a.matmul(b).data
    expect(c).toEqual([[8, 8], [8, 8]])
})

test("zeros initializer", async () => {
    const z = await zeros(2, 3).data
    expect(z).toEqual([[0, 0, 0], [0, 0, 0]])
})

test("zeroes alias matches zeros", async () => {
    expect(await zeroes(2, 2).data).toEqual(await zeros(2, 2).data)
})

test("ones initializer", async () => {
    const o = await ones(2, 2).data
    expect(o).toEqual([[1, 1], [1, 1]])
})

test("uniform initializer values are in [0, 1)", async () => {
    const u = await uniform(2, 3).data
    const flat = flattenDeep(u)
    expect(flat.length).toBe(6)
    for (const v of flat) {
        expect(v).toBeGreaterThanOrEqual(0)
        expect(v).toBeLessThan(1)
    }
})

test("uniformRange initializer values are in [low, high)", async () => {
    const low = -2
    const high = 5
    const u = await uniformRange(low, high, 4).data
    const flat = flattenDeep(u)
    expect(flat.length).toBe(4)
    for (const v of flat) {
        expect(v).toBeGreaterThanOrEqual(low)
        expect(v).toBeLessThan(high)
    }
})

test("3D broadcasting add: (1,2,3) + (3)", async () => {
    const a = tensor([[[1, 2, 3], [4, 5, 6]]])
    const b = tensor([10, 20, 30])
    const c = await a.add(b).data
    expect(c).toEqual([[[11, 22, 33], [14, 25, 36]]])
})

test("3D broadcasting add: (2,1,3) + (1,2,1) -> (2,2,3)", async () => {
    const a = tensor([
        [[1, 2, 3]],
        [[4, 5, 6]],
    ])
    const b = tensor([
        [[10], [20]],
    ])
    const c = await a.add(b).data
    expect(c).toEqual([
        [[11, 12, 13], [21, 22, 23]],
        [[14, 15, 16], [24, 25, 26]],
    ])
})

test("sum axis 2 on 3D tensor", async () => {
    const a = tensor([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]) // shape (2,2,3)
    const c = await a.sum(2).data
    expect(c).toEqual([
        [6, 15],
        [24, 33],
    ]) // shape (2,2)
})

test("sum negative axis (-1) equals sum last axis", async () => {
    const a = tensor([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ]) // shape (2,2,2)
    const c1 = await a.sum(-1).data
    const c2 = await a.sum(2).data
    expect(c1).toEqual(c2)
    expect(c1).toEqual([
        [3, 7],
        [11, 15],
    ])
})

test("broadcasting incompatible shapes throws", async () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]]) // (2,3)
    const b = tensor([1, 2]) // (2)
    expect(() => a.add(b)).toThrow()
})

test("default dtype is float32", () => {
    expect(tensor([1, 2, 3]).dtype).toBe("float32")
    expect(zeros(2, 2).dtype).toBe("float32")
    expect(ones(2, 2).dtype).toBe("float32")
})

test("tensor dtype option is stored", () => {
    expect(tensor([1, 2, 3], { dtype: "int32" }).dtype).toBe("int32")
    expect(tensor([[1, 2], [3, 4]], { dtype: "float16" }).dtype).toBe("float16")
    expect(tensor([0, 1, 0], { dtype: "bool" }).dtype).toBe("bool")
})

test("initializer dtype option is stored", () => {
    expect(zeros(2, 3, { dtype: "int32" }).dtype).toBe("int32")
    expect(zeroes(2, 3, { dtype: "float16" }).dtype).toBe("float16")
    expect(ones(2, 3, { dtype: "float64" }).dtype).toBe("float64")
})

test("dtype promotion in add/sub", async () => {
    const a = tensor([1, 2, 3], { dtype: "int32" })
    const b = tensor([1, 2, 3], { dtype: "float32" })
    expect((await a.add(b).realize()).dtype).toBe("float32")
    expect((await b.sub(a).realize()).dtype).toBe("float32")

    const c = tensor([1, 2, 3], { dtype: "float16" })
    expect((await c.add(b).realize()).dtype).toBe("float32")

    const d = tensor([1, 0, 1], { dtype: "bool" })
    expect((await d.add(a).realize()).dtype).toBe("int32")
})

test("dtype promotion in matmul", async () => {
    const a = tensor(
        [
            [1, 2],
            [3, 4],
        ],
        { dtype: "float16" }
    )
    const b = tensor(
        [
            [1, 0],
            [0, 1],
        ],
        { dtype: "float32" }
    )
    expect((await a.matmul(b).realize()).dtype).toBe("float32")
})

test("sum dtype rules", async () => {
    const a = tensor([1, 0, 1], { dtype: "bool" })
    expect((await a.sum().realize()).dtype).toBe("int32")

    const b = tensor([1, 2, 3], { dtype: "int32" })
    expect((await b.sum().realize()).dtype).toBe("int32")

    const c = tensor([1, 2, 3], { dtype: "float32" })
    expect((await c.sum().realize()).dtype).toBe("float32")
})

test("uniform rejects non-float dtype at runtime", async () => {
    expect(() => (uniform as any)(2, 2, { dtype: "int32" })).toThrow()
    expect(() => (uniformRange as any)(0, 1, 2, 2, { dtype: "bool" })).toThrow()
})
