import { realize } from "./realize"
import { tensor } from "./tensor";

const a = tensor([1, 2, 3])
const b = tensor([4, 5, 6])
const c = a.add(b)
const result = await realize([c])
console.log("Realized result:", result);