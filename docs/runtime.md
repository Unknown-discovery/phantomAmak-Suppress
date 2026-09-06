# C++ execution, memory layout and verification

## Components

```text
Offline, not part of inference:
  float32 W -> per-tile orthonormal DCT -> int8 Q + delta
            -> independent byte-rANS blocks -> .amak writer

Native linear inference:
  file header + directory -> select tensor extent
                             |
        producer thread      |       caller/consumer thread
  read one chunk header      |       spatial input X
  validate framing/CRC       |             |
  build normalized CDF       |          block DCT
  entropy-decode int8 Q -----+----> Q × transformed activation
       bounded slots         |       scalar delta after each dot
                             |             |
  prefetch next chunk        |       accumulate output frequencies
                             |             |
                             |        inverse block DCT + bias
                             |             |
                             |       spatial output Y
```

The producer prefetches **tiles within a tensor**, not an entire next layer. A
future model scheduler could overlap layers using the same independent extents;
v1 has no transformer scheduler. No arithmetic is performed directly on rANS
bits: they must first be decoded to compact integer frequency coefficients.

## Transform dispatch

`linear` uses `DctPlan` for float-input forward and double-input inverse
activation transforms. `RunOptions::dct_backend` selects `auto`, scalar or AVX2;
`RunResult::dct_backend` reports the resolved backend. Feature detection and
function-table selection happen outside the transform loops. The scalar encoder,
wire format, rANS decoder and coefficient multiply are unchanged. See
[AVX2 kernels and benchmarks](simd.md) for dispatch safety and numerical tolerances.

## On-disk versus in-memory layout

The [binary specification](format-v1.md) gives every on-disk field's byte offset.
`src/binary.hpp` implements little-endian reads/writes, IEEE-754 bit conversion,
checked size arithmetic and incremental CRC. There are no packed structs,
alignment-dependent casts, pointer fields or host-endian dumps in the file.

The in-memory public tile in `include/amak/amak.hpp` is conceptually:

```cpp
struct DecodedTile {
    uint64_t index, row_start, col_start;
    uint16_t rows, cols;
    float delta;
    std::vector<int8_t> coefficients; // Q[u * cols + v]
};
```

This is **not a stable C ABI or a wire struct**. Its vector storage is reserved
once for the maximum nominal tile area `T = tile_rows * tile_cols`; edge tiles
only change the live length. A callback receives a const reference valid **only
until the callback returns**. Retaining the pointer is an application bug;
copying a tile is possible but outside the runtime's bounded-storage guarantee.

The producer privately owns:

- One seekable `ifstream` scoped to this tensor scan.
- One 64-byte header and a reusable encoded-body buffer reserved for
  `4*256 + 2*T + 4` bytes (sparse frequency table plus worst-case payload).
- The 256-entry uint16 frequency table, 256-entry uint16 cumulative table and
  4096-entry uint8 symbol lookup used by the rANS decoder.
- The rANS uint32 state and byte cursor, restarted for every independent chunk.

A chunk is validated before being published. Dimensions, count, framing,
coordinates, finite positive step, probability model, CRC, initial/final rANS
states and exact byte consumption all have explicit checks. Opening a container
checks metadata only; `amak verify` scans and decodes all chunks without a GEMM.

## Ring lifecycle and ownership

`Container::visit_tiles(..., prefetch_slots)` accepts 0..8:

- **0:** synchronous path with one reusable decoded slot, no worker thread.
- **P > 0:** one producer and `min(P, chunk_count)` reusable slots. This count
  includes the currently consumed slot; it is not `P` extra tiles plus a hidden
  consumer copy. P=1 creates a thread but permits no tile/compute overlap.

Each slot moves through these states:

```text
FREE -> FILLING -> READY -> READING -> FREE
        producer          consumer
```

A mutex and condition variable govern state transitions. The producer does I/O
and entropy decoding outside the lock. The consumer invokes the callback and
computes outside the lock. It returns a slot to FREE only after the callback
finishes, so no producer can overwrite a tile while it is being multiplied.
Both sides traverse slots modulo the fixed slot count in file order. There is
no unbounded work queue, detached thread or spin-wait.

Producer errors are captured as `exception_ptr`, wake the consumer and are
rethrown in the caller's thread. Callback errors set cancellation and wake any
producer waiting for a free slot. All exit paths after successful thread
creation join the producer. The consumer also checks errors raised by the final
extent check *after* the last tile was made ready. A scan can consequently report
an error after some low-level callbacks ran; callbacks are not transactional.
`linear` only returns its output after the complete scan succeeds.

Separate visits, including simultaneous `linear` calls, use separate streams,
slots and activation buffers. Metadata is read-only. The underlying file must
remain immutable; neither CRC nor a length check provides a file snapshot against
a concurrent external writer. Cancellation cannot interrupt an OS-blocked disk
read; join waits for that read to return.

## Memory budget

Let `R,C` be nominal tile dimensions, `T=R*C`, `B` batch size and `P` the actual
slot count (one for synchronous mode). The main requested runtime buffers are:

| Storage | Bytes |
|---|---:|
| Decoded quantized coefficients | `P*T` |
| Encoded chunk body | `1024 + 2*T + 4` |
| Chunk header and rANS tables | `64 + 5120` |
| Frequency activation scratch | `8*B*(min(M,R) + min(N,C))` |
| Reused spatial activation row tile | `8*min(M,R)` |
| DCT basis plans | `8*sum(n*n for n in S)` |
| Caller input | `4*B*N` |
| Returned spatial output | `4*B*M` |
| Optional caller bias | `4*M` |

`S` is the unique set of nonzero dimensions among `min(M,R)`, `min(N,C)`,
`M % R`, `N % C`. At most four small bases are cached **within one call**; no
full-model transform basis is built. Inverse results use one spatial activation
row tile, shared across batch rows, before bias and float32 conversion. This adds
at most 2048 bytes, not a spatial weight tile. `activation_scratch_bytes` counts
both frequency buffers and this spatial row tile. The implementation reports
vector capacity rather than merely live element count. Allocator/container overhead, metadata,
stream buffering, thread stacks, code, filesystem cache and process runtime are
not included in these buffer figures. They are **not a process-RSS measurement**.

With two 16×16 slots, the explicit coefficient pool is 512 bytes and the encoded
buffer/header/rANS tables account for 6724 bytes with the tested standard library.
That does not imply the whole process uses 7 KB. In a complete transformer,
activations, attention workspace, KV caches and architecture state remain extra.

The DCT costs `O(n^2)` per vector in either scalar or AVX2 mode; coefficient
multiplication still costs `O(BMN)` in the scalar reference kernel. The runtime
recomputes each input tile's DCT for each output tile and performs just one
inverse DCT per output tile. A persistent worker, cross-call reusable plans,
SIMD coefficient-dot kernels, FFT DCTs, larger batches and alternative entropy
models are later optimizations, not implemented performance claims.

## Public API

```cpp
#include <amak/amak.hpp>

amak::Container model("model.amak");
std::size_t w = model.find_tensor("linear.weight");
// Replace these zeros with real activations for one batch row.
std::vector<float> x(static_cast<std::size_t>(model.tensors()[w].cols), 0.0F);
std::vector<float> bias; // optional: populate with output_features values
amak::RunOptions options;
options.batch = 1;
options.prefetch_slots = 2; // set 0 to compare the synchronous path
options.dct_backend = amak::DctBackend::automatic; // or scalar / avx2

auto result = amak::linear(model, w, x, options, bias);
// result.output is spatial [batch, output features]. Apply SiLU/GELU/etc here.
```

A `TensorSource` plus `write_container` is the separate offline encoding API and
can write multiple named matrices into one container. Source data is validated;
a complete file is staged in a uniquely created sibling directory and published
by filesystem rename. Failed encoding cleans up the stage and does not replace
an existing destination. This is not an fsync/power-loss durability guarantee;
replacement behavior follows the host filesystem. The CLI refuses output paths
that alias its inputs. Raw `.f32` output is written only after inference succeeds,
but a subsequent disk-write failure can still leave a partial raw output file.

## What the tests establish

`make test` or CTest runs:

- CRC/endian known answers, DCT round trips and Parseval tests through size 256.
- Scalar versus selected DCT kernels for every size 1..256, unaligned buffers,
  both input precisions, round trips, mixed scales and subnormal float input.
- CPU/build dispatch, explicit unavailable AVX2 requests and scalar-only builds.
- The exact unquantized rectangular DCT multiplication identity.
- rANS known states, all 256 symbols, 65536-element blocks, constant and skewed
  histograms, deterministic normalization, truncated/malformed streams.
- Multi-tensor, batched, biased streaming forward against dense weights rebuilt
  from the **same quantized coefficients**, plus source-weight error bounds.
- Irregular edge tiles, one-dimensional/scalar tiles, zero weights, fixed steps,
  underflow-scale and output-overflow cases.
- Synchronous versus 1/2/4/8-slot output equality, slot reuse and constant
  coefficient-buffer capacity across different matrix sizes.
- Callback cancellation, late producer errors, concurrent/repeated scans, invalid
  API shapes and configurable resource limits.
- Every byte-length truncation of a small container; corrupt checksums; valid-CRC
  adversarial headers, offsets, extents, metadata, models and rANS states.
- A **separate Python standard-library reader**, CDF-based rANS decoder and direct
  four-index inverse DCT reference checking C++-written files and CLI results.
- Handcrafted Python fixtures testing C++ reader interoperability, including the
  signed -128 symbol that the symmetric reference encoder does not emit.
- CLI argument/input protection and preservation of existing output on encode
  failures.

`make sanitize` repeats the suite under AddressSanitizer and
UndefinedBehaviorSanitizer. These checks are not a security audit, formal
verification, a race-detector run or an end-to-end language-model accuracy test.

## Next stages (not yet implemented)

1. Benchmark real transformed layers against *resident quantized* baselines;
   measure compression, p50/p95 latency, decode/compute split, RSS and task error.
2. Decide an actual model architecture and specify bias/norm/attention/tokenizer
   metadata. Implement embeddings, RoPE, attention/KV cache and nonlinear blocks.
3. Define versioned, bounded declarative recipes for derived tensors and aliases;
   reject arbitrary executable graph payloads.
4. Introduce per-channel/curvature conventions only alongside mathematically
   justified kernels and format conformance tests.
5. Extend SIMD beyond the DCT to coefficient-dot kernels, then GPU execution
   and a persistent scheduler without changing numerical semantics. Continue
   profiling and fuzzing both accelerated and reference paths.
