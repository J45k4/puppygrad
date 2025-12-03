import { Op } from "./op"

type Shape = readonly number[]

export type DType = "bool" | "int32" | "float16" | "float32" | "float64"
export type FloatDType = "float16" | "float32" | "float64"

type NestedArray<T> = T | readonly NestedArray<T>[] | NestedArray<T>[]
type Decrement<D extends number> = D extends 5 ? 4 : D extends 4 ? 3 : D extends 3 ? 2 : D extends 2 ? 1 : D extends 1 ? 0 : 0
type ShapeOf<T, Depth extends number = 5> =
    Depth extends 0 ? number[] :
    T extends number ? [] :
    // If `T` is a tuple/readonly tuple, `T["length"]` is a numeric literal.
    // If it is a plain array, `T["length"]` is just `number`.
    T extends readonly (infer U)[] ? [T["length"], ...ShapeOf<U, Decrement<Depth>>] :
    T extends (infer U)[] ? [T["length"], ...ShapeOf<U, Decrement<Depth>>] :
    []
type DimsArgs<D extends DType> = Array<number | { dtype?: D }>

// Type-level dtype promotion (simple lattice; must mirror runtime)
export type Promote<A extends DType, B extends DType> =
    A extends B ? A :
    A extends "float64" ? "float64" :
    B extends "float64" ? "float64" :
    A extends "float32" ? "float32" :
    B extends "float32" ? "float32" :
    A extends "float16" ? "float16" :
    B extends "float16" ? "float16" :
    A extends "int32" ? "int32" :
    B extends "int32" ? "int32" :
    "bool"

export type SumDType<D extends DType> = D extends "bool" ? "int32" : D extends "int32" ? "int32" : D

function isFloatDType(dt: DType): dt is FloatDType {
    return dt === "float16" || dt === "float32" || dt === "float64"
}

function promoteDType(a: DType, b: DType): DType {
    if (a === b) return a
    if (a === "float64" || b === "float64") return "float64"
    if (a === "float32" || b === "float32") return "float32"
    if (a === "float16" || b === "float16") return "float16"
    if (a === "int32" || b === "int32") return "int32"
    return "bool"
}

function sumDType(a: DType): DType {
    return a === "bool" ? "int32" : a
}

function parseDimsAndOpts<D extends DType>(args: DimsArgs<D>): { dims: number[]; dtype: D } {
    const work = [...args]
    const last = work[work.length - 1]
    const hasOpts = typeof last === "object" && last !== null && !Array.isArray(last)
    const opts = (hasOpts ? work.pop() : undefined) as { dtype?: D } | undefined
    const dims = work as number[]
    const dtype = (opts?.dtype ?? "float32") as D
    return { dims, dtype }
}

// Helper: compute shape of a nested array at runtime
function computeShape(data: NestedArray<number>): number[] {
    const shape: number[] = []
    let current: any = data
    while (Array.isArray(current)) {
        shape.push(current.length)
        current = current[0]
    }
    return shape
}

// Helper: flatten nested arrays into a 1D array
function flatten(data: NestedArray<number>): number[] {
    const out: number[] = []

    function rec(d: NestedArray<number>): void {
        if (Array.isArray(d)) {
            for (const v of d) rec(v as any)
        } else {
            out.push(d as number)
        }
    }

    rec(data)
    return out
}

// Helper: unflatten a 1D array into a nested structure based on shape
function unflatten(data: number[], shape: Shape): any {
    if (shape.length === 0) {
        return data[0]
    }
    if (shape.length === 1) {
        return data.slice()
    }

    const [dim, ...rest] = shape as [number, ...number[]]
    const step = rest.reduce((a, b) => a * b, 1)
    const out: any[] = []
    for (let i = 0; i < dim; i++) {
        const start = i * step
        const slice = data.slice(start, start + step)
        out.push(unflatten(slice, rest))
    }
    return out
}

function sizeOfShape(shape: readonly number[]): number {
    if (shape.length === 0) return 1
    return shape.reduce((a, b) => a * b, 1)
}

function assertValidShapeDims(shape: readonly number[]): void {
    for (const d of shape) {
        if (!Number.isFinite(d) || !Number.isInteger(d)) {
            throw new Error(`Shape dims must be finite integers, got ${d}`)
        }
        if (d < 1) {
            throw new Error(`Shape dims must be >= 1, got ${d}`)
        }
    }
}

// Helpers for broadcasting and indexing
function computeStrides(shape: number[]): number[] {
    const strides = new Array(shape.length)
    let stride = 1
    for (let i = shape.length - 1; i >= 0; i--) {
        strides[i] = stride
        const dim = shape[i]!
        stride *= dim
    }
    return strides
}

function indexToCoords(index: number, shape: number[]): number[] {
    const coords = new Array(shape.length)
    for (let i = shape.length - 1; i >= 0; i--) {
        const dim = shape[i]!
        coords[i] = index % dim
        index = Math.floor(index / dim)
    }
    return coords
}

function coordsToIndex(coords: number[], strides: number[]): number {
    let idx = 0
    for (let i = 0; i < coords.length; i++) {
        const c = coords[i] ?? 0
        const s = strides[i] ?? 0
        idx += c * s
    }
    return idx
}

function broadcastShapes(a: number[], b: number[]): number[] {
    const len = Math.max(a.length, b.length)
    const out = new Array(len)
    for (let i = 0; i < len; i++) {
        const ad = a[a.length - 1 - i] ?? 1
        const bd = b[b.length - 1 - i] ?? 1
        if (ad !== bd && ad !== 1 && bd !== 1) {
            throw new Error(`Cannot broadcast shapes [${a}] and [${b}]`)
        }
        out[len - 1 - i] = Math.max(ad, bd)
    }
    return out
}

function alignCoords(coords: number[], outShape: number[], inShape: number[]): number[] {
    const offset = outShape.length - inShape.length
    const res = new Array(inShape.length)
    for (let i = 0; i < inShape.length; i++) {
        const dim = inShape[i]!
        const c = coords[i + offset] ?? 0
        res[i] = dim === 1 ? 0 : c
    }
    return res
}

export class Tensor<S extends Shape = Shape, D extends DType = "float32"> {
    public op: Op

    // Flat data + shape
    constructor(private _data: number[], public shape: S, public dtype: D, op?: Op) {
        this.op = op ?? new Op()
    }

    // --- Core elementwise ops with broadcasting ---

    public add<S2 extends Shape, D2 extends DType>(other: Tensor<S2, D2>): Tensor<Shape, Promote<D, D2>> {
        const outShape = broadcastShapes([...this.shape], [...other.shape])
        const outDType = promoteDType(this.dtype, other.dtype) as Promote<D, D2>
        const out = new Tensor(new Array(outShape.reduce((a, b) => a * b, 1)).fill(0), outShape, outDType)
        out.op = Op.add(this.op, other.op)
        return out
    }

    public sub<S2 extends Shape, D2 extends DType>(other: Tensor<S2, D2>): Tensor<Shape, Promote<D, D2>> {
        const outShape = broadcastShapes([...this.shape], [...other.shape])
        const outDType = promoteDType(this.dtype, other.dtype) as Promote<D, D2>
        const out = new Tensor(new Array(outShape.reduce((a, b) => a * b, 1)).fill(0), outShape, outDType)
        out.op = Op.sub(this.op, other.op)
        return out
    }

    public mul<S2 extends Shape, D2 extends DType>(other: Tensor<S2, D2>): Tensor<Shape, Promote<D, D2>> {
        const outShape = broadcastShapes([...this.shape], [...other.shape])
        const outDType = promoteDType(this.dtype, other.dtype) as Promote<D, D2>
        const out = new Tensor(new Array(outShape.reduce((a, b) => a * b, 1)).fill(0), outShape, outDType)
        out.op = Op.mul(this.op, other.op)
        return out
    }

    // --- Reductions ---

    public sum(axis?: number): Tensor<Shape, SumDType<D>> {
        const inShape = [...this.shape] as number[]

        // Sum all elements → keep as [sum] to match tests
        const rank = inShape.length
        const ax = axis === undefined ? undefined : axis < 0 ? rank + axis : axis
        if (ax !== undefined && (ax < 0 || ax >= rank)) {
            throw new Error(`Axis ${axis} out of range for shape [${inShape}]`)
        }

        const outShapeRaw =
            ax === undefined ? [] : inShape.slice(0, ax).concat(inShape.slice(ax + 1))
        const outShape = outShapeRaw.length === 0 ? [1] : outShapeRaw
        const outSize = outShape.reduce((a, b) => a * b, 1)
        const out = new Tensor(new Array(outSize).fill(0), outShape, sumDType(this.dtype) as SumDType<D>)
        out.op = Op.sum(this.op, ax)
        return out
    }

    // --- Transpose (2D only for now) ---

    public transpose(): Tensor<Shape, D> {
        const shape = [...this.shape] as number[]
        if (shape.length <= 1) {
            return new Tensor(this._data.slice(), shape, this.dtype)
        }
        if (shape.length !== 2) {
            throw new Error(`transpose currently only supports 2D tensors, got rank ${shape.length}`)
        }

        const [rows, cols] = shape as [number, number]
        const outShape = [cols, rows]
        const outData = new Array(this._data.length)

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const inIdx = i * cols + j
                const outIdx = j * rows + i
                outData[outIdx] = this._data[inIdx]
            }
        }

        return new Tensor(outData, outShape, this.dtype)
    }

    // --- Matmul (2D only for now, no batching) ---

    public matmul<S2 extends Shape, D2 extends DType>(other: Tensor<S2, D2>): Tensor<Shape, Promote<D, D2>> {
        const aShape = [...this.shape] as number[]
        const bShape = [...other.shape] as number[]

        if (aShape.length !== 2 || bShape.length !== 2) {
            throw new Error("matmul currently only supports 2D tensors")
        }

        const [m, k] = aShape as [number, number]
        const [k2, n] = bShape as [number, number]
        if (k !== k2) {
            throw new Error(`Cannot matmul shapes [${aShape}] and [${bShape}]`)
        }

        const outShape = [m, n]
        const outData = new Array(m * n).fill(0)

        // Correct matmul computation
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                let sum = 0
                for (let t = 0; t < k; t++) {
                    const aIdx = i * k + t
                    const bIdx = t * n + j
                    sum += (this._data[aIdx] ?? 0) * (other._data[bIdx] ?? 0)
                }
                outData[i * n + j] = sum
            }
        }

        const outDType = promoteDType(this.dtype, other.dtype) as Promote<D, D2>
        return new Tensor(outData, outShape, outDType)
    }

    // Placeholder for future lazy execution / backends
    public realize(): Tensor<S, D> {
        return this
    }

    // Convert back to nested JS arrays
    public list(): any {
        return unflatten(this._data, this.shape)
    }
}

// Factory function: infer shape from nested array at compile time (depth-limited for type checker)
export function tensor<const T extends NestedArray<number>, D extends DType = "float32">(
    data: T,
    opts?: { dtype?: D }
): Tensor<ShapeOf<T>, D> {
    const shape = computeShape(data) as ShapeOf<T>
    const flat = flatten(data)
    const dtype = (opts?.dtype ?? "float32") as D
    return new Tensor(flat, shape, dtype)
}

export function zeros<const Dims extends number[], D extends DType = "float32">(...dims: Dims): Tensor<Dims, D>
export function zeros<const Dims extends number[], D extends DType = "float32">(
    ...args: [...dims: Dims, opts: { dtype?: D }]
): Tensor<Dims, D>
export function zeros<const Dims extends number[], D extends DType = "float32">(
    ...args: [...dims: Dims, opts?: { dtype?: D }]
): Tensor<Dims, D> {
    const { dims, dtype } = parseDimsAndOpts<D>(args as unknown as DimsArgs<D>)
    const shape = [...dims] as number[]
    assertValidShapeDims(shape)
    const size = sizeOfShape(shape)
    const data = new Array(size).fill(0)
    return new Tensor(data, shape as unknown as Dims, dtype)
}

// British spelling alias
export function zeroes<const Dims extends number[], D extends DType = "float32">(...dims: Dims): Tensor<Dims, D>
export function zeroes<const Dims extends number[], D extends DType = "float32">(
    ...args: [...dims: Dims, opts: { dtype?: D }]
): Tensor<Dims, D>
export function zeroes<const Dims extends number[], D extends DType = "float32">(
    ...args: [...dims: Dims, opts?: { dtype?: D }]
): Tensor<Dims, D> {
    return zeros(...(args as any))
}

export function ones<const Dims extends number[], D extends DType = "float32">(...dims: Dims): Tensor<Dims, D>
export function ones<const Dims extends number[], D extends DType = "float32">(
    ...args: [...dims: Dims, opts: { dtype?: D }]
): Tensor<Dims, D>
export function ones<const Dims extends number[], D extends DType = "float32">(
    ...args: [...dims: Dims, opts?: { dtype?: D }]
): Tensor<Dims, D> {
    const { dims, dtype } = parseDimsAndOpts<D>(args as unknown as DimsArgs<D>)
    const shape = [...dims] as number[]
    assertValidShapeDims(shape)
    const size = sizeOfShape(shape)
    const data = new Array(size).fill(1)
    return new Tensor(data, shape as unknown as Dims, dtype)
}

// Uniform in [0, 1)
export function uniform<const Dims extends number[], D extends FloatDType = "float32">(...dims: Dims): Tensor<Dims, D>
export function uniform<const Dims extends number[], D extends FloatDType = "float32">(
    ...args: [...dims: Dims, opts: { dtype?: D }]
): Tensor<Dims, D>
export function uniform<const Dims extends number[], D extends FloatDType = "float32">(
    ...args: [...dims: Dims, opts?: { dtype?: D }]
): Tensor<Dims, D> {
    const { dims, dtype } = parseDimsAndOpts<D>(args as unknown as DimsArgs<D>)
    if (!isFloatDType(dtype)) {
        throw new Error(`uniform only supports float dtypes, got ${dtype}`)
    }
    const shape = [...dims] as number[]
    assertValidShapeDims(shape)
    const size = sizeOfShape(shape)
    const data = new Array(size)
    for (let i = 0; i < size; i++) data[i] = Math.random()
    return new Tensor(data, shape as unknown as Dims, dtype)
}

// Uniform in [low, high)
export function uniformRange<const Dims extends number[], D extends FloatDType = "float32">(
    low: number,
    high: number,
    ...dims: Dims
): Tensor<Dims, D>
export function uniformRange<const Dims extends number[], D extends FloatDType = "float32">(
    low: number,
    high: number,
    ...args: [...dims: Dims, opts: { dtype?: D }]
): Tensor<Dims, D>
export function uniformRange<const Dims extends number[], D extends FloatDType = "float32">(
    low: number,
    high: number,
    ...args: [...dims: Dims, opts?: { dtype?: D }]
): Tensor<Dims, D> {
    if (!Number.isFinite(low) || !Number.isFinite(high)) {
        throw new Error(`low/high must be finite numbers, got low=${low}, high=${high}`)
    }
    if (high <= low) {
        throw new Error(`high must be > low, got low=${low}, high=${high}`)
    }

    const { dims, dtype } = parseDimsAndOpts<D>(args as unknown as DimsArgs<D>)
    if (!isFloatDType(dtype)) {
        throw new Error(`uniformRange only supports float dtypes, got ${dtype}`)
    }

    const shape = [...dims] as number[]
    assertValidShapeDims(shape)
    const size = sizeOfShape(shape)
    const span = high - low
    const data = new Array(size)
    for (let i = 0; i < size; i++) data[i] = low + Math.random() * span
    return new Tensor(data, shape as unknown as Dims, dtype)
}
