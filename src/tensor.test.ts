import { expect, test } from "bun:test"
import { ones, tensor, uniform, uniformRange, zeros, zeroes } from "./tensor"

function flattenDeep(x: any): number[] {
    if (!Array.isArray(x)) return [x]
    const out: number[] = []
    for (const v of x) out.push(...flattenDeep(v))
    return out
}

test("add tensors", () => {
    const a = tensor([1, 2, 3])
    const b = tensor([1, 2, 3])
    const c = a.add(b).realize().list()
    expect(c).toEqual([2, 4, 6])
})

test("sub tensors", () => {
    const a = tensor([5, 7, 9])
    const b = tensor([1, 2, 3])
    const c = a.sub(b).realize().list()
    expect(c).toEqual([4, 5, 6])
})

test("sum tensor", () => {
    const a = tensor([1, 2, 3, 4])
    const c = a.sum().realize().list()
    expect(c).toEqual([10])
})

test("sum tensor along axis", () => {
    const a = tensor([[1, 2], [3, 4]])
    const c = a.sum(0).realize().list()
    expect(c).toEqual([4, 6])
})

test("transpose tensor", () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]])
    const c = a.transpose().realize().list()
    expect(c).toEqual([[1, 4], [2, 5], [3, 6]])
})

test("test broadcasting add", () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]])
    const b = tensor([10, 20, 30])
    const c = a.add(b).realize().list()
    expect(c).toEqual([[11, 22, 33], [14, 25, 36]])
})

test("test broadcasting sub", () => {
    const a = tensor([[10, 20, 30], [40, 50, 60]])
    const b = tensor([1, 2, 3])
    const c = a.sub(b).realize().list()
    expect(c).toEqual([[9, 18, 27], [39, 48, 57]])
})

test("matmul two tensors", () => {
    const a = tensor([[2, 2], [2, 2]])
    const b = tensor([[2, 2], [2, 2]])
    const c = a.matmul(b).realize().list()
    expect(c).toEqual([[8, 8], [8, 8]])
})

test("zeros initializer", () => {
    const z = zeros(2, 3).realize().list()
    expect(z).toEqual([[0, 0, 0], [0, 0, 0]])
})

test("zeroes alias matches zeros", () => {
    expect(zeroes(2, 2).list()).toEqual(zeros(2, 2).list())
})

test("ones initializer", () => {
    const o = ones(2, 2).realize().list()
    expect(o).toEqual([[1, 1], [1, 1]])
})

test("uniform initializer values are in [0, 1)", () => {
    const u = uniform(2, 3).realize().list()
    const flat = flattenDeep(u)
    expect(flat.length).toBe(6)
    for (const v of flat) {
        expect(v).toBeGreaterThanOrEqual(0)
        expect(v).toBeLessThan(1)
    }
})

test("uniformRange initializer values are in [low, high)", () => {
    const low = -2
    const high = 5
    const u = uniformRange(low, high, 4).realize().list()
    const flat = flattenDeep(u)
    expect(flat.length).toBe(4)
    for (const v of flat) {
        expect(v).toBeGreaterThanOrEqual(low)
        expect(v).toBeLessThan(high)
    }
})

test("3D broadcasting add: (1,2,3) + (3)", () => {
    const a = tensor([[[1, 2, 3], [4, 5, 6]]])
    const b = tensor([10, 20, 30])
    const c = a.add(b).realize().list()
    expect(c).toEqual([[[11, 22, 33], [14, 25, 36]]])
})

test("3D broadcasting add: (2,1,3) + (1,2,1) -> (2,2,3)", () => {
    const a = tensor([
        [[1, 2, 3]],
        [[4, 5, 6]],
    ])
    const b = tensor([
        [[10], [20]],
    ])
    const c = a.add(b).realize().list()
    expect(c).toEqual([
        [[11, 12, 13], [21, 22, 23]],
        [[14, 15, 16], [24, 25, 26]],
    ])
})

test("sum axis 2 on 3D tensor", () => {
    const a = tensor([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]) // shape (2,2,3)
    const c = a.sum(2).realize().list()
    expect(c).toEqual([
        [6, 15],
        [24, 33],
    ]) // shape (2,2)
})

test("sum negative axis (-1) equals sum last axis", () => {
    const a = tensor([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ]) // shape (2,2,2)
    const c1 = a.sum(-1).realize().list()
    const c2 = a.sum(2).realize().list()
    expect(c1).toEqual(c2)
    expect(c1).toEqual([
        [3, 7],
        [11, 15],
    ])
})

test("broadcasting incompatible shapes throws", () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]]) // (2,3)
    const b = tensor([1, 2]) // (2)
    expect(() => a.add(b).realize()).toThrow()
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

test("dtype promotion in add/sub", () => {
    const a = tensor([1, 2, 3], { dtype: "int32" })
    const b = tensor([1, 2, 3], { dtype: "float32" })
    expect(a.add(b).dtype).toBe("float32")
    expect(b.sub(a).dtype).toBe("float32")

    const c = tensor([1, 2, 3], { dtype: "float16" })
    expect(c.add(b).dtype).toBe("float32")

    const d = tensor([1, 0, 1], { dtype: "bool" })
    expect(d.add(a).dtype).toBe("int32")
})

test("dtype promotion in matmul", () => {
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
    expect(a.matmul(b).dtype).toBe("float32")
})

test("sum dtype rules", () => {
    const a = tensor([1, 0, 1], { dtype: "bool" })
    expect(a.sum().dtype).toBe("int32")

    const b = tensor([1, 2, 3], { dtype: "int32" })
    expect(b.sum().dtype).toBe("int32")

    const c = tensor([1, 2, 3], { dtype: "float32" })
    expect(c.sum().dtype).toBe("float32")
})

test("uniform rejects non-float dtype at runtime", () => {
    expect(() => (uniform as any)(2, 2, { dtype: "int32" })).toThrow()
    expect(() => (uniformRange as any)(0, 1, 2, 2, { dtype: "bool" })).toThrow()
})