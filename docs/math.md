# The exact DCT-domain linear operator

## 1. Orientation and normalization

We use the ML convention `W ∈ R^(M×N)` (output features by input features).
For a column activation `x`, `y = W x + b`; for row-batched activations,
`Y = X W^T + b`. If an existing API writes `Y = X W`, transpose its weight
convention before applying these formulas.

Define the **orthonormal DCT-II matrix** of size `n` by

\[
(D_n)_{k,j} = \alpha_k \cos\!\left[\frac{\pi}{n}(j+\tfrac12)k\right],
\quad
\alpha_0=\sqrt{1/n},\quad \alpha_{k>0}=\sqrt{2/n}.
\]

`D_n D_n^T = I`. Forward DCT is `D_n x`; inverse DCT is `D_n^T x_hat`
(the corresponding orthonormal DCT-III). Using an unnormalized library DCT
without correcting both axes changes the operator.

## 2. A single rectangular weight tile

Store its two-dimensional frequency coefficients:

\[
C = D_M W D_N^T,
\qquad W = D_M^T C D_N.
\]

Therefore the exact forward operation is

\[
\boxed{\hat{x}=D_Nx,\quad \hat{y}=C\hat{x},\quad y=D_M^T\hat{y}+b.}
\]

For a batch in row-major orientation:

\[
\boxed{Y=(X D_N^T) C^T D_M+b.}
\]

It is **not** an elementwise multiplication of frequency coefficients. DCT does
not diagonalize an arbitrary learned dense matrix. There remains a matrix-vector
or matrix-matrix multiply, with the same dense asymptotic multiply count unless
another, explicitly justified sparsity/structure approximation is introduced.
Both the input and output axes must be transformed; merely taking `DCT(X)` and
multiplying by an arbitrary transformed `W` is insufficient.

## 3. Independent tiles and a bounded streaming algorithm

Partition the spatial weight into row/column tiles `W_ij` of actual shape
`r_i × c_j`, and the input into segments `x_j`. Each tile has its own transform:

\[
C_{ij}=D_{r_i}W_{ij}D_{c_j}^T,\qquad
\hat C_{ij}=\delta_{ij}Q_{ij}.
\]

Then the quantized model's output row segment is

\[
\boxed{\tilde y_i=D_{r_i}^T\left(
  \sum_j\delta_{ij}Q_{ij}(D_{c_j}x_j)\right)+b_i.}
\]

For a batch this is

\[
\tilde Y_i=\left[\sum_j(X_jD_{c_j}^T)
 (\delta_{ij}Q_{ij})^T\right]D_{r_i}+b_i.
\]

The common output transform can be applied **once per row tile**, after summing
all input-tile contributions. The runtime follows exactly the file's
output-tile-major order:

```text
for each output tile i:
    output_frequency_accumulator[B, r_i] = 0
    for each input tile j:
        stream-decode Q_ij into an int8 slot
        input_frequency[B, c_j] = DCT(X_j)
        accumulator += delta_ij * input_frequency * Q_ij^T
        release the coefficient slot
    Y_i = inverse_DCT(accumulator) + b_i
```

No float32 `W_ij`, full frequency matrix or full reconstructed state dict is
created by inference. Input DCTs are recomputed for each output tile rather than
cached across the entire input; that is a deliberate memory/compute trade-off.
Boundary tiles use smaller DCT matrices, not a cropped transform of a padded
tile. `r_i = 1` or `c_j = 1` is valid and gives an identity one-dimensional DCT.

The offline encoder necessarily sees spatial source weights. Keeping that
conversion outside inference is the distinction this prototype demonstrates.

## 4. What scale fusion is actually valid?

Because `delta_ij` is a **scalar for one tile**, the runtime performs

\[
\hat y_u \mathrel{+}=\delta_{ij}\sum_v Q_{uv}\hat x_v.
\]

It converts compact integers for arithmetic, but does not multiply every weight
coefficient by `delta` or allocate an unquantized coefficient tensor. This is
algebraic factoring of a tile scale, **not** fusion into layer normalization.
Different tile steps cannot in general be replaced by one layer-wide scalar.

For a per-coefficient step matrix `S`, `(S ⊙ Q) x_hat` does not factor into
`Q x_hat` times a scalar. If `S = a b^T` is separable, diagonal pre/post scaling
is possible *in the frequency basis*. Arbitrary spatial diagonal scales or
curvature matrices do not commute with the DCT:

\[
D_n G \ne G D_n \quad \text{in general}.
\]

For example, if the intended spatial operator is
`W = A (D_M^T C D_N) B`, correct execution is
`y = A D_M^T C D_N (B x)`, with `A` and `B` explicitly defined by the encoder.
It is not correct to omit them or absorb them into an unrelated attention
constant. A diagonal spatial matrix generally becomes dense under
`D G D^T`. The symbol `G` alone does not establish a valid whitening/unwhitening
convention; that must be part of a future format's contract.

Layer normalization involves means, variances, epsilon and often a residual
path. It is not a general-purpose sink for arbitrary scales. Attention scaling,
RoPE, normalization and quantization fusion each need an operator-specific
proof and tests. V1 consequently supports only scalar steps per tile.

## 5. Nonlinear operations and derived tensors

In general `D f(x) ≠ f(D x)` for GELU, SiLU, softmax and normalization. The
linear runtime returns **spatial activations**, where callers may apply bias,
nonlinearities, residual additions, attention or other operators. There is no
claim that an entire transformer can remain in a single DCT basis.

Causal masks can be produced by an attention operator and tied embeddings can
share a tensor handle, but they are not universally derivable from compressed
weights. Head dimensions, positional encoding, architecture and tying semantics
need explicit metadata. This prototype does not invent that metadata or execute
container-supplied code.

## 6. Correctness and quantization error

Let `E_ij = delta_ij Q_ij - C_ij`. Orthogonality gives

\[
\|\tilde W_{ij}-W_{ij}\|_F=\|E_{ij}\|_F,
\qquad
\|\tilde W-W\|_F^2=\sum_{ij}\|E_{ij}\|_F^2.
\]

For rounding without clipping, ignoring floating-point transform roundoff,
`|E_uv| ≤ delta_ij/2`, hence

\[
\|\tilde W-W\|_F^2\le
\sum_{ij}r_i c_j\,\delta_{ij}^2/4,
\qquad
\|\tilde y-y\|_2\le\|\tilde W-W\|_F\|x\|_2.
\]

Two distinct tests are required:

1. **Runtime correctness:** compare streaming execution with a dense reference
   reconstructed from the *same stored quantized coefficients*. Errors should
   be only floating-point roundoff. Dense reconstruction is test-only.
2. **Quantization quality:** compare against the *original* source weight
   operator. Report RMSE/max error (and task metrics for an actual model), not
   exact equality. A correct kernel does not imply acceptable model quality.

## 7. Performance boundaries

The coefficient multiply remains a scalar double-precision reference. The DCT
transforms have both scalar and optional runtime-dispatched AVX2 kernels, not
FFT, BLAS or GPU implementations. Dense coefficient work is `O(BMN)`; tiled
`O(n^2)` transforms add work. SIMD changes throughput, not these identities or
asymptotics; see [dispatch, roundoff and benchmarks](simd.md). rANS tables, headers and DCT computation can outweigh compression,
especially for small or high-entropy tiles. DCT of arbitrary neural weights is
not guaranteed to concentrate energy. CRC and entropy decoding also cost time.

Prefetch may overlap decode/I/O with computation, but benefits depend on the
machine, tile sizes, storage cache and workload. Re-decoding on every token can
be slower than reusing resident quantized weights. Modern runtimes such as
llama.cpp already execute quantized kernels; full floating-point restoration is
not an inherent requirement of conventional inference engines.

Startup still requires opening files and reading/validating metadata. Parameters
are not the whole memory budget: input/output activations, transform plans,
attention/KV cache in a future transformer, filesystem cache and other state all
remain. “Zero initialization delay,” “tiny total RAM,” and unconditional speedup
are not guarantees of this design.
