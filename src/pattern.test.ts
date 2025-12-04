import { expect, test, describe } from "bun:test"
import { Op } from "./op"
import type { LazyNode, OpKind } from "./op"
import { Add, Mul, Var, Const, Cast, tryMatch, pattern } from "./pattern"

let _id = 0
function n(op: OpKind, inputs: LazyNode[] = [], attrs?: any): LazyNode {
    return new Op(op, inputs as any, attrs, { id: _id++, shape: [1], dtype: "float32" })
}

describe("tryMatch", () => {
    test("binds variables and matches Const by value", () => {
        const x = n("Neg", [n("Const", [], { value: 123 })])
        const expr = n("Add", [x, n("Const", [], { value: 0 })])

        const pat = Add(Var("x"), Const(0))
        const b = tryMatch(expr, pat)

        expect(b).not.toBeNull()
        expect(b!.x).toBe(x)
    })

    test("fails when Const value differs", () => {
        const x = n("Neg", [n("Const", [], { value: 123 })])
        const expr = n("Add", [x, n("Const", [], { value: 1 })])

        const pat = Add(Var("x"), Const(0))
        const b = tryMatch(expr, pat)

        expect(b).toBeNull()
    })

    test("matches nested patterns", () => {
        const x = n("Relu", [n("Const", [], { value: 5 })])
        const expr = n("Mul", [n("Add", [x, n("Const", [], { value: 0 })]), n("Const", [], { value: 1 })])

        const pat = Mul(Add(Var("x"), Const(0)), Const(1))
        const b = tryMatch(expr, pat)

        expect(b).not.toBeNull()
        expect(b!.x).toBe(x)
    })

    test("enforces repeated Var must be same node (Add(x,x))", () => {
        const a = n("Relu", [n("Const", [], { value: 7 })])
        const b = n("Relu", [n("Const", [], { value: 7 })])

        const same = n("Add", [a, a])
        const diff = n("Add", [a, b])

        const x = Var("x")
        const pat = Add(x, x)

        expect(tryMatch(same, pat)).not.toBeNull()
        expect(tryMatch(diff, pat)).toBeNull()
    })

    test("matches attrs (Cast to=...)", () => {
        const x = n("Relu", [n("Const", [], { value: 2 })])
        const expr = n("Cast", [x], { to: "float16" })

        const pat = Cast(Var("x"), "float16")
        const b = tryMatch(expr, pat)

        expect(b).not.toBeNull()
        expect(b!.x).toBe(x)
    })

    test("fails when input arity differs", () => {
        const expr = n("Add", [n("Const", [], { value: 1 })]) // wrong arity
        const pat = Add(Var("x"), Var("y"))

        expect(tryMatch(expr, pat)).toBeNull()
    })
})

describe("pattern(...) runner", () => {
    test("returns first matching case result", () => {
        const x = n("Relu", [n("Const", [], { value: 3 })])
        const expr = n("Add", [x, n("Const", [], { value: 0 })])

        const runner = pattern([
            [Mul(Var("a"), Var("b")), (_ctx, _ret) => "mul", "mul-case"],
            [Add(Var("a"), Const(0)), (_ctx, _ret, { a }) => `add0:${a!.id}`, "add0-case"],
        ] as const)

        const out = runner({ any: "ctx" }, expr)
        expect(out).toBe(`add0:${x.id}`)
    })

    test("supports alternative LHS patterns (Pat[])", () => {
        const a = n("Relu", [n("Const", [], { value: 1 })])
        const b = n("Relu", [n("Const", [], { value: 2 })])

        const addExpr = n("Add", [a, b])
        const mulExpr = n("Mul", [a, b])

        const runner = pattern([
            [[Add(Var("x"), Var("y")), Mul(Var("x"), Var("y"))], (_ctx, _ret, { x, y }) => `${x!.id},${y!.id}`],
        ] as const)

        expect(runner(null, addExpr)).toBe(`${a.id},${b.id}`)
        expect(runner(null, mulExpr)).toBe(`${a.id},${b.id}`)
    })

    test("returns null when no match, or uses onNoMatch", () => {
        const expr = n("Neg", [n("Const", [], { value: 1 })])

        const runner1 = pattern([[Add(Var("x"), Const(0)), () => "hit"]] as const)
        expect(runner1(null, expr)).toBeNull()

        const runner2 = pattern([[Add(Var("x"), Const(0)), () => "hit"]] as const, {
            onNoMatch: () => "miss",
        })
        expect(runner2(null, expr)).toBe("miss")
    })
})
