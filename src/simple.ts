import { tensor } from "./tensor";

const a = tensor([1, 2, 3])
const b = tensor([4, 5, 6])
const c = a.add(b)
const result = await c.realize()
console.log("Realized result:", result.list())
