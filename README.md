# AMAK native-execution prototype

A small, dependency-free **C++17 CPU reference runtime** for tiled, quantized
DCT-domain linear operators in a newly specified `.amak` binary container, with
optional runtime-dispatched **AVX2 block-DCT kernels**.

It streams independent rANS chunks into bounded int8 coefficient slots and
multiplies in the frequency basis. **Inference never reconstructs a full spatial
weight matrix or state dict.** Only activation vectors return to spatial space.

This repository defines an **experimental AMAK 1.0 format**, not compatibility
with an existing proprietary encoder. It is a working linear-layer prototype,
**not yet a transformer/token-generation engine**.

## Specification first

1. **[Binary header and chunk layout](docs/format-v1.md)** — exact byte offsets,
   endianness, directory/extents, scalar quantization, rANS state convention,
   checksums, reserved fields and validation limits.
2. **[DCT-domain equations](docs/math.md)** — rectangular and batched operators,
   tiling, valid scale factoring, nonlinear boundaries and quantization error.
3. **[C++ memory and execution design](docs/runtime.md)** — slot ownership,
   bounded prefetch, cancellation, memory budget, API and testing.
4. **[AVX2 dispatch and benchmarks](docs/simd.md)** — portable fallback, SIMD
   transforms, numerical tolerances and reproducible performance measurements.

For `W[M,N]`, a spatial column vector obeys `y = W x + b`. With
`C = D_M W D_N^T`, orthonormal DCT matrices give:

\[
y = D_M^T C D_N x + b.
\]

For independently quantized tiles `C_ij ≈ delta_ij Q_ij`, the implemented operator
is:

\[
\boxed{\tilde y_i=D_{r_i}^T\left(\sum_j\delta_{ij}Q_{ij}D_{c_j}x_j\right)+b_i.}
\]

A scalar tile step is applied after each dot product—no float `delta*Q` tensor
is allocated. Arbitrary per-coefficient/curvature scales cannot generally be
folded into layer normalization; v1 deliberately does not claim that fusion.

## Build and run

Needs a C++17 compiler (tested with GCC 12), Make and POSIX-style threading.
Tests additionally use Python 3's standard library; no NumPy, PyTorch, BLAS,
external compression library or model downloads are required.

```sh
make -j2
make test
./build/amak backends
./build/amak bench-dct --size 64
./build/amak demo build/demo.amak
./build/amak inspect build/demo.amak
./build/amak verify build/demo.amak
```

The demo creates deterministic synthetic random weights, packs them, compares
scalar/selected DCT backends and synchronous/two-slot execution, and reports
timing, numerical/quantization error and explicit buffer sizes. It intentionally
retains the source matrix for validation; use `run` for a path that loads no
dense source weights. Generated `.amak`, `.f32`
and build artifacts are ignored by Git.

Alternatively, with CMake 3.16+:

```sh
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake --parallel 2
(cd build/cmake && ctest --output-on-failure)
```

Sanitizer checks:

```sh
make sanitize
# Or configure CMake with -DAMAK_SANITIZE=ON (GCC/Clang).
```

## Select and benchmark the DCT backend

Inference defaults to `--dct-backend auto`: AVX2 on supported CPUs/builds,
otherwise scalar. Force `scalar` for comparison or `avx2` to require acceleration
(an unavailable AVX2 request fails cleanly). Only the **activation DCT transforms**
are vectorized; coefficient multiplication and rANS decoding remain scalar.
There is no global `-mavx2` requirement or binary-format change.

```sh
./build/amak bench-dct --size 64 --iterations 10000 --dct-backend auto
./build/amak demo build/simd-demo.amak --rows 256 --cols 384 \
  --tile-rows 64 --tile-cols 64 --batch 4 --iterations 20 --dct-backend auto
make test-scalar  # build without AVX2; verify auto fallback and error handling
```

`bench-dct` reports five-trial median kernel timings and scalar/selected errors.
`demo` measures complete warm forward calls, including per-call plan creation,
I/O, decoding, coefficient multiplication and thread setup. Kernel ratios are
**not** whole-engine speedups. See [the SIMD guide](docs/simd.md) for details and
`-DAMAK_ENABLE_AVX2=OFF` CMake builds.

## Pack and execute your own matrix

CLI raw files are **little-endian, row-major IEEE-754 float32**, with no headers.
Weights use `[output_features, input_features]`, input uses `[batch,
input_features]`, and output uses `[batch, output_features]`.

Create a small example without any Python packages:

```sh
python3 - <<'PY'
import struct
# W = [[1, 2, 3], [-1, 0, 1]], X = [[1, 0, -1]]
open('build/weights.f32', 'wb').write(struct.pack('<6f', 1, 2, 3, -1, 0, 1))
open('build/input.f32', 'wb').write(struct.pack('<3f', 1, 0, -1))
PY

./build/amak pack build/weights.f32 build/linear.amak \
  --rows 2 --cols 3 --tile-rows 2 --tile-cols 3 --name linear.weight

./build/amak run build/linear.amak build/input.f32 build/output.f32 \
  --batch 1 --prefetch 2 --tensor linear.weight

python3 - <<'PY'
import struct
print(struct.unpack('<2f', open('build/output.f32', 'rb').read()))
# Approximately (-2, -2); the difference is lossy coefficient quantization.
PY
```

`--prefetch 0` selects synchronous execution. `1..8` chooses the total number
of slots including the one being consumed; two slots allow decode and compute
to overlap. Slot count does not scale with parameter count. `pack --step DELTA`
uses a fixed positive coefficient step and rejects clipping; otherwise steps
are chosen automatically for each tile.

`inspect` validates **only** the header/directory. `verify` scans and validates
all chunk CRCs and rANS streams. Run `./build/amak --help` for all commands.
The CLI caps each raw input, source weight and output array at 16,777,216 elements.
The C++ API supports multiple named tensors, an optional spatial bias and
configurable reader limits; see [the API example](docs/runtime.md#public-api).

## Implemented scope

| Implemented | Not implemented |
|---|---|
| Versioned file/directory/chunk framing with CRC-32 | Existing proprietary `.amak` compatibility |
| Offline orthonormal block DCT + symmetric int8 quantization | Checkpoint/tokenizer/model-graph importer |
| Independent normalized byte-rANS32 encoder/decoder | Wavelets, curvature/per-channel scale schemes |
| Bounded synchronous or producer/consumer tile decoding | GPU/FFT kernels or persistent workers |
| Scalar + runtime-dispatched AVX2 DCT transforms | SIMD coefficient GEMM or attention kernels |
| Rectangular/batched linear layers and caller-supplied bias | Transformer attention, RoPE, KV cache, token generation |
| Limits, corruption checks, error propagation and tests | Serialized lazy recipes or automatic scale/norm fusion |

The source is split into `src/container.cpp` (format and stream lifecycle),
`src/rans.cpp` (codec), `src/dct.cpp` / `src/dct_kernels.cpp` (plans, scalar/SIMD
transforms and dispatch), `src/runtime.cpp` (native linear kernel), and
`src/main.cpp` / `src/benchmark.cpp` (CLI and benchmarks). Public headers are in
`include/amak/`.

## Correctness versus performance

The suite compares streaming results with a dense reference reconstructed from
**the same quantized coefficients**, separately from error against original
weights. It also includes an independent Python format/rANS/DCT implementation,
handcrafted interoperability fixtures, malformed files and concurrency tests.
SIMD checks exercise every size 1..256, unaligned tails, both activation input
precisions, scalar-only fallback and full forward accuracy.

The auditable scalar reference remains alongside optional AVX2 DCT kernels;
there is no blanket speedup claim. DCT does not turn an arbitrary dense matrix
multiply into elementwise multiplication. Entropy decoding,
headers/tables, CRC, repeated transforms and thread startup can cost more than the
bandwidth saved; small tiles can even increase file size. Quantization quality
requires real model/task evaluation.

Startup is not zero-cost. Reported coefficient buffers are not total RAM: input,
output, transform plans, metadata, thread/allocator overhead, filesystem cache and
future attention/KV state remain. Existing quantized engines also need not restore
all weights to floating point. See [performance boundaries](docs/math.md#7-performance-boundaries)
and [next stages](docs/runtime.md#next-stages-not-yet-implemented) before extending
this into a model runtime.
