# AMAK 1.0 binary format (prototype specification)

**Status:** a new, experimental format defined by this repository, not a claim of
compatibility with any existing proprietary `.amak` encoder. The implementation
is a CPU reference for dense **linear operators represented by independently
quantized, block-DCT coefficients**. It does not encode a transformer graph.

The keywords MUST, MUST NOT and SHOULD describe this version's wire contract.
All offsets and lengths are **bytes**. All integers and IEEE-754 binary32 values
are **little-endian**. Readers MUST decode fields explicitly, not cast file bytes
to C/C++ structs. Reserved fields and padding MUST be zero. Unknown versions,
flags, tensor kinds, transforms, quantizers and codecs MUST be rejected.

## 1. File organization

```text
+------------------------------+ offset 0
| file header (64 bytes)       |
+------------------------------+ offset 64
| tensor directory (N * 128)   |
+------------------------------+
| tensor 0: chunk 0, 1, ...    | contiguous, output-tile-major order
+------------------------------+
| tensor 1: chunk 0, 1, ...    |
+------------------------------+
| ...                          |
+------------------------------+ file_bytes (no trailer or alignment padding)
```

A tensor directory entry gives a seekable layer extent. Inside that extent each
chunk has an independent probability table and rANS state; no previous chunk is
needed to decode it. There is no per-chunk random-access index in v1. A forward
pass reads one tensor's extent sequentially. Startup reads the header and
directory only, not the coefficients.

### File header: 64 bytes

| Offset | Bytes | Type | Field / required value |
|---:|---:|---|---|
| 0 | 8 | bytes | Magic: `41 4d 41 4b 0d 0a 1a 0a` (`AMAK\r\n\x1a\n`) |
| 8 | 2 | u16 | Major version = 1 |
| 10 | 2 | u16 | Minor version = 0 |
| 12 | 4 | u32 | Header bytes = 64 |
| 16 | 4 | u32 | Flags = 0 |
| 20 | 4 | u32 | Tensor count, nonzero |
| 24 | 8 | u64 | Directory offset = 64 |
| 32 | 8 | u64 | Directory bytes = `tensor_count * 128` |
| 40 | 8 | u64 | Total file bytes, exactly the physical length |
| 48 | 4 | u32 | CRC-32 of the complete directory |
| 52 | 4 | u32 | CRC-32 of this header, with bytes 52..55 set to zero |
| 56 | 8 | u64 | Reserved = 0 |

CRC-32 everywhere is CRC-32/ISO-HDLC: reflected polynomial `0xedb88320`, initial
register `0xffffffff`, final XOR `0xffffffff`. For example `123456789` checksums
to `0xcbf43926`. CRC detects accidental corruption, **not malicious alteration**.

### Tensor directory entry: 128 bytes

| Offset | Bytes | Type | Field / required value |
|---:|---:|---|---|
| 0 | 48 | bytes | Unique, nonempty name: 1..47 printable ASCII bytes (`0x21..0x7e`), NUL terminator, zero padding |
| 48 | 2 | u16 | Tensor kind = 1 (linear weight) |
| 50 | 1 | u8 | Rank = 2 |
| 51 | 1 | u8 | Transform = 1 (orthonormal, separable block DCT-II) |
| 52 | 1 | u8 | Quantizer = 1 (signed int8 coefficients, scalar step per chunk, no zero point) |
| 53 | 1 | u8 | Entropy codec = 1 (byte-rANS32, 12 probability bits) |
| 54 | 2 | u16 | Flags = 0 |
| 56 | 8 | u64 | Rows `M` (output features), nonzero |
| 64 | 8 | u64 | Columns `N` (input features), nonzero |
| 72 | 4 | u32 | Nominal tile rows `R`, 1..256 |
| 76 | 4 | u32 | Nominal tile columns `C`, 1..256 |
| 80 | 8 | u64 | Chunk count = `ceil(M/R) * ceil(N/C)` |
| 88 | 8 | u64 | Absolute offset of this tensor's first chunk |
| 96 | 8 | u64 | Tensor extent in bytes, including all chunk headers |
| 104 | 24 | bytes | Reserved = 0 |

The first extent MUST start immediately after the directory. Subsequent extents
MUST start at the preceding extent's end, in directory order. The final extent
MUST end at `file_bytes`. There are no aliases, holes, overlapping extents or
empty tensors. Size arithmetic MUST be checked for overflow before allocation
or seeking. Readers MAY impose tighter resource limits than the wire format.

`W` has shape `[M, N]` in row-major order, following the usual ML linear-layer
convention, not `[input, output]`. A batch `X[B, N]` produces `Y = X W^T`.

## 2. Chunk framing

Let `J = ceil(N/C)`. For chunk ordinal `k`:

```text
row_start = floor(k/J) * R
col_start = (k mod J) * C
rows      = min(R, M - row_start)
cols      = min(C, N - col_start)
```

Chunks MUST appear in this order and cover every weight exactly once. Edge
chunks use their **actual dimensions** for the DCT; they are not zero-padded.
The frequency matrix `Q[rows, cols]` is serialized row-major (output frequency
first, input frequency second).

```text
chunk = fixed_header[64]
      + frequency_table[4 * alphabet_count]
      + rans_payload[payload_bytes]
```

### Chunk header: 64 bytes

| Offset | Bytes | Type | Field / required value |
|---:|---:|---|---|
| 0 | 4 | bytes | Magic `ACNK` (`41 43 4e 4b`) |
| 4 | 2 | u16 | Header bytes = 64 |
| 6 | 2 | u16 | Flags = 0 |
| 8 | 8 | u64 | Zero-based chunk ordinal within this tensor |
| 16 | 8 | u64 | Spatial row start |
| 24 | 8 | u64 | Spatial column start |
| 32 | 2 | u16 | Actual rows, 1..256 |
| 34 | 2 | u16 | Actual columns, 1..256 |
| 36 | 4 | f32 | Quantization step `delta`, finite and strictly positive |
| 40 | 4 | u32 | Symbol count = `rows * cols`, at most 65536 |
| 44 | 2 | u16 | Alphabet count `K`, 1..256 |
| 46 | 1 | u8 | Probability bits = 12 |
| 47 | 1 | u8 | Codec = 1 |
| 48 | 4 | u32 | Payload bytes, 4..`2 * symbol_count + 4` |
| 52 | 4 | u32 | Total chunk bytes = `64 + 4*K + payload_bytes` |
| 56 | 4 | u32 | CRC-32 of the entire chunk, with bytes 56..59 set to zero |
| 60 | 4 | u32 | Reserved = 0 |

Before allocating payload storage, the reader MUST check all bounds against the
validated tensor extent and the maximum tile size. It MUST validate the chunk's
CRC before entropy decoding. It MUST also validate the fields and entropy model;
a valid checksum is not evidence that a file is well-formed.

### Sparse normalized frequency table

Each entry is `(symbol: u8, reserved: u8 = 0, frequency: u16)`. Entries MUST be
strictly increasing by symbol; frequencies MUST be positive and sum to 4096.
Symbols not listed have zero frequency. Cumulative frequency `c_s` is the sum of
frequencies for symbols strictly smaller than `s`.

A byte symbol represents signed coefficient `q = int(symbol) - 128`. Thus symbol
128 is zero. The reader accepts `q` in `[-128,127]`; the reference encoder uses
symmetric `[-127,127]`. The reconstructed *frequency* coefficient is
`C_hat[u,v] = delta * q[u,v]`. A float weight tile is never required in inference.

For reproducibility the reference encoder normalizes histogram counts as follows:

1. Give each observed symbol one frequency unit.
2. With `A = 4096 - K` and total count `T`, give symbol `s` another
   `floor(A * count_s / T)` units.
3. Give the remaining units, one each, to the largest remainders
   `(A * count_s) mod T`; break ties by ascending symbol.

Other encoders MAY choose any valid table. A single-symbol chunk has frequency
4096 and a four-byte payload; there is no special zero-tile codec.

### Byte-rANS32 bitstream

Constants: `p = 12`, `F = 1 << p`, `L = 1 << 23`. Arithmetic is unsigned; valid
states fit in 32 bits. Encoding starts with state `x = L` and processes symbols
in **reverse** coefficient order:

```text
x_max = ((L >> p) << 8) * f_s
while x >= x_max:
    emitted.push_back(x & 255)
    x >>= 8
x = ((x / f_s) << p) + (x % f_s) + c_s
```

The payload is the final state as **four little-endian bytes**, followed by the
emitted bytes in reverse order. There is no terminal marker. The initial decoder
state MUST lie in `[L, 256*L)`.

For each of exactly `symbol_count` output coefficients, in forward order:

```text
slot = x & (F - 1)
find s such that c_s <= slot < c_s + f_s
output int(s) - 128
x = f_s * (x >> p) + slot - c_s
while x < L:
    x = (x << 8) | next_payload_byte()
```

Reading beyond the payload is an error. After the last symbol, `x` MUST equal
`L` and all payload bytes MUST have been consumed. The maximum two renormalizing
bytes per symbol gives the framing bound `2*T + 4`. A 4096-byte symbol lookup
table avoids searching the CDF for every coefficient.

Two small conformance examples (payload bytes only, excluding table and header):

- Any nonempty block of `q = 0`: table `(symbol=128, frequency=4096)`;
  payload `00 00 80 00` (state `L`, no renormalization bytes).
- `q = [0, 1]`: table `(128, 2048), (129, 2048)`;
  payload `00 10 00 02` (state `0x02001000`, no renormalization bytes).

The tests also exercise renormalization, full alphabets and maximum-sized blocks.

## 3. Quantization and capabilities

For each spatial weight tile `W_ij`, the encoder computes
`C_ij = D_rows W_ij D_cols^T`. By default it selects a float32
`delta = max(abs(C_ij)) / 127` (rounded to float32, clamped upward to the smallest
positive float32 if necessary); an all-zero tile uses `delta = 1`. A caller may
supply a fixed positive step. The stored coefficient is
`q = round(C_ij / delta)`, with halfway cases rounded away from zero. The encoder
MUST reject coefficients whose rounded value is outside `[-127,127]`, rather
than silently clip. A step that cannot be represented as finite, positive f32
is an encoding error. Nonfinite source weights are rejected.

DCT arithmetic uses double precision in this reference encoder. Encoders need
not produce byte-identical quantized files across math libraries; readers follow
the stored values, not the source weights. This is lossy quantization, not a
lossless restoration of the original checkpoint.

V1 deliberately does **not** serialize biases, curvature matrices, per-channel
scales, wavelets, attention configuration, executable graphs, tied-weight aliases,
KV caches, tokenizers or lazy tensor recipes. A caller can add a spatial bias
through the runtime API. Weight tying can reuse the same tensor handle at the
application level. More complex metadata needs a new, specified format version;
reserved fields are not an unvalidated escape hatch for arbitrary code.

## 4. Prototype safety limits

The default reader caps the directory at 4096 tensors, each dimension at
1,048,576, and the number of chunks per tensor at 16,777,216. Tile dimensions
are always capped at 256. The execution API separately caps each input/output
array at 16,777,216 elements by default, and prefetch at 8 slots. Dimension,
chunk-count and activation limits are configurable through the API.

Header and directory checks happen at open; chunk checks happen on consumption.
`inspect` is therefore **not** a full-file integrity scan. Each scan uses its own
file stream, and the source file MUST remain unchanged during a reader's lifetime.
Untrusted models still warrant process-level memory, CPU-time and file-access
limits. No executable data is read from the container.
