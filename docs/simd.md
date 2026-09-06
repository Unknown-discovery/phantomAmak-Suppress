# Optional AVX2 block-DCT execution

## Scope and compatibility

This extends **AMAK 1.0 execution**, not its wire format. `src/dct_kernels.cpp`
contains scalar and AVX2-targeted implementations of:

- Orthonormal DCT-II: double input → double frequency output.
- Orthonormal DCT-II: float input → double frequency output, without a float-to-
  double staging array.
- Orthonormal inverse DCT: double frequency input → double spatial output.

`linear` now routes its actual input/output activation transforms through these
kernels. The int8-coefficient/double-activation multiply is **still scalar**, as
are rANS decoding and scale factoring. This is not a SIMD GEMM, FFT, GPU backend,
attention implementation or an AMAK v2 extension.

Both two-dimensional offline helpers and the encoder remain pinned to the scalar
DCT. Selecting an execution backend cannot change quantized coefficients or add
hardware-specific fields to a file. The rANS/container validation is unchanged.
The pre-optimization demo container was checked byte-for-byte against the new
writer. As before, packing across different math libraries need not be bit-identical.

## Dispatch and build safety

```cpp
#include <amak/amak.hpp>

amak::Container model("model.amak");
amak::RunOptions options;
options.batch = 1;
options.prefetch_slots = 2;
options.dct_backend = amak::DctBackend::automatic;
std::vector<float> x(static_cast<std::size_t>(model.tensors()[0].cols), 0.0F);
auto result = amak::linear(model, 0, x, options);
// result.dct_backend is the resolved backend, not "automatic".
```

| Request | Behavior |
|---|---|
| `automatic` / CLI `auto` | AVX2 when compiled and usable, otherwise scalar |
| `scalar` | Always use reference DCT kernels |
| `avx2` | Require AVX2; throw a normal error if the CPU/OS/build cannot use it |

On GCC/Clang x86 toolchains, only the intrinsic functions have
`__attribute__((target("avx2")))`. There is **no global `-mavx2`, `-mfma` or
`-march=native` flag**. Compiler CPU feature builtins check usable AVX2, including
OS-managed AVX register state. The result is cached in a thread-safe local static.
A plan selects a function table once; no feature probe happens per coefficient
or inside the transform loops. Only ordinary pointers/integers cross dispatch
boundaries. No AVX instructions execute just to construct the dispatch table.

On non-x86 or unsupported compilers (including MSVC-style toolchains), the AVX2
functions are excluded and `auto` falls back to scalar. `avx2_dct_compiled()` and
`avx2_dct_available()` distinguish build capability from usable execution.

```sh
make -j2
./build/amak backends

# Exclude the intrinsic functions entirely; also tests automatic fallback and
# normal errors for an explicit, unavailable AVX2 request.
make test-scalar
./build/scalar/amak backends

# Equivalent CMake configuration:
cmake -S . -B build/scalar-cmake -DAMAK_ENABLE_AVX2=OFF
cmake --build build/scalar-cmake --parallel 2
(cd build/scalar-cmake && ctest --output-on-failure)
```

For custom Make invocations use `AMAK_ENABLE_AVX2=0` or `1`. Use a separate
`BUILD_DIR` (or clean first) when changing compiler/build flags; Make does not
track arbitrary flag changes. Do not globally enable a newer ISA when building
an executable intended to run on older CPUs. Do not enable `-ffast-math`: it can
invalidate finite-value validation and the documented numerical behavior.

## Kernels and buffer contract

The basis is the same row-major double array `D[k,j]` in both backends:

- **Forward:** load four double lanes at a time, using two accumulators for
  eight-element groups. Float inputs use four-float loads widened directly into
  double lanes. A horizontal reduction combines the lanes, followed by a scalar
  tail. No float accumulator or intermediate quantized activation is introduced.
- **Inverse:** accumulate outer products `output[j] += D[k,j] * input[k]` with
  contiguous four-output-lane loads/stores. Four frequencies are unrolled in
  order to reduce output traffic. There is no transposed basis allocation.
- **Edges:** all sizes 1..256 are supported. Sizes need not be multiples of four
  or eight. Unaligned loads/stores are used, and scalar tails never read beyond
  the `n` live elements. The caller does not supply padded or aligned arrays.

Input and output must be disjoint valid buffers. Plans and their immutable basis
can be shared across threads; callers provide separate output buffers.
`DctPlan(n)` deliberately defaults to scalar for offline/reference uses. To
select acceleration for direct transform calls use
`DctPlan(n, DctBackend::automatic)` or an explicit backend.

The inverse result is staged in **one reusable spatial activation row tile**
before spatial bias, float32 overflow validation and output conversion. This
adds `8 * min(M, tile_rows)` bytes (at most 2048 bytes), independent of batch size
and parameter count. It is not a reconstructed weight tile. There is no extra
coefficient slot or transposed-basis copy. `activation_scratch_bytes` reports this
row tile along with both frequency activation buffers; the other memory figures
retain their previous meanings.

## Floating-point contract

Bases, dot products and inverse accumulation remain double precision. Default
GCC/Clang builds use `-ffp-contract=off`; the AVX2 path does not require FMA or
AVX-512. Forward SIMD reductions group additions differently, so scalar and AVX2
results need not be **bitwise** equal. Compare with tolerances scaled by input
magnitude and transform dimension, and compare final float32 outputs separately.

The new kernel errors are floating-point roundoff, not another quantization
scheme. The original lossy coefficient-quantization error remains. For a given
backend, synchronous and prefetched execution still consume tiles in the same
order and are tested for identical output bytes. Do not require bitwise equality
across different backends or claim that transform acceleration improves model
quality.

## Reproducible measurements

### Warm transform microbenchmark

```sh
./build/amak bench-dct --size 64 --iterations 10000 --warmup 100 \
  --dct-backend auto

for n in 3 16 31 32 64 128 256; do
  ./build/amak bench-dct --size "$n" --iterations 3000 --warmup 100
done
```

The command compares the scalar reference with the selected backend for float
forward, double forward and double inverse transforms. It performs **five
trials**, alternates scalar/selected order, warms each path, and reports median
nanoseconds per call, scalar/selected timing ratio and maximum absolute error.
Plan construction, allocation, file I/O, rANS decoding and coefficient
multiplication are excluded. Compiler fences and per-iteration checksums prevent
the measured work from being eliminated. Both timings include the small dispatch,
loop and checksum overhead. A scalar-only build can run the same benchmark, but
both columns then measure scalar code; differences are noise, not acceleration.

The input vectors and repeated warm working set are synthetic. These are not
cold-cache measurements, a resident-quantized-GEMM comparison, a process RSS
measurement or a production-model throughput result. Small transforms may not
benefit, and virtualized/shared hosts have noisy clocks and scheduling.

### End-to-end synthetic linear benchmark

```sh
./build/amak demo build/avx2-demo.amak --rows 256 --cols 384 \
  --tile-rows 64 --tile-cols 64 --batch 4 --iterations 20 --dct-backend auto
```

The demo reports scalar-DCT synchronous, selected-DCT synchronous and
selected-DCT two-slot prefetched means. Each configuration gets one untimed
warmup. Timed forward calls still include DCT plan construction, open/seek/read,
CRC/rANS work, coefficient multiplication, allocations and worker creation.
Different configurations are measured sequentially, so repeat runs and use
controlled machine conditions before drawing performance conclusions. The
reported scalar/selected output error is separate from source-weight
quantization error. The demo retains source weights and validation outputs,
so its process memory is not a native-only inference-memory measurement.

Kernel-only gains do **not** imply equal end-to-end gains: entropy decoding,
scalar coefficient dots, small tiles and per-call overhead can dominate. Measure
a real workload before changing tile sizes or assuming prefetch is faster.

## Example local measurement

One run on **2026-09-06**, on this x86-64 Intel Xeon 2.60 GHz virtual machine,
using GCC 12.2, the Make `-O2` defaults and `-ffp-contract=off`, produced the
following 64-point medians (3000 iterations, 100 warmups, five trials):

| Operation | Scalar ns/call | AVX2 ns/call | Scalar / AVX2 |
|---|---:|---:|---:|
| Double forward | 2878.67 | 532.90 | 5.40× |
| Float-input forward | 2917.25 | 624.66 | 4.67× |
| Double inverse | 3030.10 | 384.33 | 7.88× |

The largest scalar/AVX2 absolute error in those three 64-point comparisons was
`2.22e-15`. In the **same run**, the three-point float-forward and inverse ratios
were only `0.66×` and `0.49×`: the targeted path was slower at that tiny size.

The 256×384 synthetic layer above (batch 4, 64×64 tiles, 20 iterations) measured
**1.775 ms scalar synchronous**, **1.539 ms AVX2 synchronous**, and **1.458 ms
AVX2 with two-slot prefetch**. Thus the same-prefetch synchronous comparison was
only about **1.15×**, not the much larger isolated-DCT ratio. Scalar/AVX2 float32
outputs happened to be identical for this sample; this is not a general bitwise
contract. These are illustrative shared-VM results, not portable guarantees or
real language-model benchmarks. Repeat the commands on the target machine.

## Validation

The test suite covers:

- `auto`, explicit scalar, explicit AVX2 and unavailable/invalid requests.
- Every transform size **1 through 256**, both input precisions, forward/inverse,
  round trips, cancellation/mixed-scale values and subnormal float input.
- Unaligned, exactly sized buffers with no tail padding, under ASan/UBSan.
- Batched biased linear layers, irregular/max-area tiles and memory accounting.
- Scalar versus selected-backend accuracy, plus the existing independent Python
  dense/DCT/rANS reference and same-backend prefetch determinism.
- Unchanged file bytes across backend choices and input/output protection on errors.
- The new `backends`, `bench-dct` and `--dct-backend` CLI paths.
- Full scalar-only builds, with automatic fallback and explicit-AVX2 errors.

The CI matrix builds AVX2 enabled/disabled with sanitizers enabled/disabled.
Actual AVX2 execution is tested only on hosts that report it usable; the suite
prints which backend was exercised. Scalar-only testing is not a substitute for
running the baseline binary on every older CPU architecture.
