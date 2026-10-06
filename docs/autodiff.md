# Compiler autodiff

Puppygrad's `grad(loss, value)` builds a reverse-mode gradient graph during
compilation. Both arguments are ordinary graph values. `loss` must be a scalar
f32; `value` must be f32. The returned tensor has the same shape as `value`.

```python
x = input("features")
w = input("weights")
target = input("targets")
rate = input("learning_rate")
prediction = matmul(x, w)
error = prediction - target
loss = reduce(error * error, add, 2)
dw = grad(loss, w)
next_w = w - rate * dw
output next_w, loss
```

There is no runtime tape, `requires_grad` flag, mutable gradient buffer or hidden
optimizer. SGD above is ordinary tensor arithmetic. The generated backward
program uses the existing Pops and C backend, including fusion and memory
planning. Repeated identical gradient requests share their result. Shared paths
accumulate contributions; broadcast dimensions are summed back to the original
input shape. An unconnected target gets a correctly shaped zero tensor.

The Rust compiler API `autodiff::gradients(graph, loss, &[a, b, ...])` requests
several gradients in one reverse traversal. Requested targets may also be
intermediate graph values. Source currently uses one `grad(loss, value)` call
per target; structural graph sharing deduplicates identical generated nodes.

## Supported operations

- Addition, subtraction, multiplication, division and negation.
- `exp2`, `log2`, `sqrt`, `sin`, elementwise `max` and `where`.
- Sum, product and maximum reductions. Products handle zero factors without
  dividing by zero.
- Reshape, prepend-axis expand, transpose, flip, pad, slice and stack.
- Floating-point identity casts. Integer casts and comparisons stop derivative
  propagation; integer differentiation targets are rejected.
- Matrix and batched matrix multiplication, including shared forward paths.
  These remain primitive reshape/multiply/reduce graphs. Recognizing their
  decomposition lets the reverse pass produce contractions instead of large
  broadcast Jacobians. Explicitly requested intermediate gradients are preserved.

`max(a, b)` assigns the derivative to `b` at a tie, matching the forward
selection; consequently `max(x, 0.0)` has derivative zero at `x = 0`.
Maximum reductions distribute the derivative evenly among tied finite maxima.
`where` propagates only to the selected value branch, not its condition.
As usual, derivatives at singularities, outside function domains, or involving
nonfinite inputs are not guaranteed to be finite.

## Current boundaries

Scalar f32 losses and static shapes are required. Reduce a vector loss explicitly;
there is no source-level seeded vector-Jacobian product interface yet. Indexed
input gradients require scatter-add, which is not implemented: requested paths
through `index` produce a compile error. Sliding `window` views also require
overlapping scatter-add for their backward pass and report an explicit error. An unrelated indexed branch does not
prevent differentiating another input. Integer index gradients are not defined.

Smooth scalar second derivatives can be composed with `grad`, but the tests do
not establish general higher-order differentiation through nonsmooth operations.
There are no GPU backends or optimizer-specific language features added here.

## MNIST validation

`examples/mnist_autodiff.pup` trains the same classifier as the handwritten
`examples/mnist.pup`, using the same file-loader contract and explicit SGD updates.

```sh
cargo run --release --offline -- train examples/mnist_autodiff.pup --epochs 5 --threads 1
```

`tests/compiler_autodiff.rs` checks central finite differences for arithmetic,
broadcasting, view transforms and matrix contractions, plus explicit tie/zero
conventions, unsupported-path errors, shared and intermediate targets, and MNIST
updates against the handwritten derivatives on full and padded batches.

The five-epoch validation on the official 60,000/10,000 split (seed 42, learning
rate 0.1, batch 128, one thread) reached 95.45% test accuracy and 0.15818 test
loss. On the local i7-8700K with GCC 16.2.1 at the default `-O2`, training epochs
totaled 4.93 seconds, excluding loading, compilation and evaluation. This is a
single validation run, not a controlled performance comparison. It emitted 19
kernels with a 72,448-byte tensor arena and 1,792 bytes of declared matmul scratch
per worker; these exclude host data, caller input/output buffers and other stack
usage. Full local results are in the ignored
`.cache/train/mnist-autodiff-5epochs/` directory.
