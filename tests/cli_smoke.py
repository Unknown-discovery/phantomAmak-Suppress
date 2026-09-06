#!/usr/bin/env python3
"""CLI integration + an independent stdlib-only AMAK/rANS/DCT reference.

Dense reconstruction here is intentionally TEST-ONLY. The C++ runtime has no
full-weight restoration path. Temporary fixtures are generated, not checked in.
"""
import bisect
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib


EXE = Path(sys.argv[1]).resolve()
MAGIC = b"AMAK\r\n\x1a\n"
LOWER = 1 << 23


def run(*args, fail=False):
    result = subprocess.run([str(EXE), *map(str, args)], capture_output=True,
                            text=True, timeout=30)
    expected = 1 if fail else 0
    assert result.returncode == expected, (args, result.returncode, result.stdout, result.stderr)
    if fail:
        assert result.stderr.strip(), args
    return result.stdout


def crc_with_zero(data, offset):
    copy = bytearray(data)
    copy[offset:offset + 4] = b"\0" * 4
    return zlib.crc32(copy)


def decode_rans(entries, payload, count):
    """Use a CDF binary search, independently of the C++ 4096-entry lookup."""
    starts, total = [], 0
    for symbol, frequency in entries:
        starts.append(total)
        total += frequency
    assert total == 4096
    state, = struct.unpack_from("<I", payload)
    assert LOWER <= state < LOWER * 256
    cursor = 4
    result = []
    for _ in range(count):
        slot = state % 4096
        index = bisect.bisect_right(starts, slot) - 1
        symbol, frequency = entries[index]
        result.append(symbol - 128)
        state = frequency * (state // 4096) + slot - starts[index]
        while state < LOWER:
            assert cursor < len(payload)
            state = state * 256 + payload[cursor]
            cursor += 1
    assert state == LOWER and cursor == len(payload)
    return result


def parse_container(path):
    data = Path(path).read_bytes()
    (magic, major, minor, header_bytes, flags, count, directory_offset,
     directory_bytes, file_bytes, directory_crc, header_crc, reserved) = struct.unpack_from(
        "<8sHHIIIQQQIIQ", data)
    assert (magic, major, minor, header_bytes, flags, reserved) == (MAGIC, 1, 0, 64, 0, 0)
    assert file_bytes == len(data)
    assert directory_offset == 64 and directory_bytes == count * 128
    assert header_crc == crc_with_zero(data[:64], 52)
    assert directory_crc == zlib.crc32(data[64:64 + directory_bytes])
    cursor = 64 + directory_bytes
    tensors = []
    for index in range(count):
        entry = data[64 + index * 128:64 + (index + 1) * 128]
        name, padding = entry[:48].split(b"\0", 1)
        assert name and not any(padding)
        fields = struct.unpack_from("<HBBBBHQQIIQQQ24s", entry, 48)
        kind, rank, transform, quantizer, codec, flags, rows, cols, tr, tc, chunks, offset, length, reserved = fields
        assert (kind, rank, transform, quantizer, codec, flags) == (1, 2, 1, 1, 1, 0)
        assert not any(reserved)
        assert 1 <= tr <= 256 and 1 <= tc <= 256
        assert chunks == ((rows + tr - 1) // tr) * ((cols + tc - 1) // tc)
        assert offset == cursor
        tiles = []
        for ordinal in range(chunks):
            h = data[cursor:cursor + 64]
            assert h[:4] == b"ACNK"
            assert struct.unpack_from("<HH", h, 4) == (64, 0)
            k, row, col = struct.unpack_from("<QQQ", h, 8)
            r, c, delta, symbols, alphabet, pbits, codec, payload_bytes, total, crc, reserved = struct.unpack_from(
                "<HHfIHBBIIII", h, 32)
            column_tiles = (cols + tc - 1) // tc
            assert (k, row, col) == (ordinal, (ordinal // column_tiles) * tr, (ordinal % column_tiles) * tc)
            assert (r, c) == (min(tr, rows - row), min(tc, cols - col))
            assert math.isfinite(delta) and delta > 0 and symbols == r * c
            assert 1 <= alphabet <= 256 and (pbits, codec, reserved) == (12, 1, 0)
            assert 4 <= payload_bytes <= 2 * symbols + 4
            assert total == 64 + 4 * alphabet + payload_bytes
            chunk = data[cursor:cursor + total]
            assert len(chunk) == total and crc == crc_with_zero(chunk, 56)
            entries = []
            for a in range(alphabet):
                symbol, zero, frequency = struct.unpack_from("<BBH", chunk, 64 + a * 4)
                assert zero == 0 and frequency > 0
                assert not entries or entries[-1][0] < symbol
                entries.append((symbol, frequency))
            q = decode_rans(entries, chunk[64 + 4 * alphabet:], symbols)
            tiles.append((row, col, r, c, delta, q))
            cursor += total
        assert cursor == offset + length
        tensors.append((name.decode("ascii"), rows, cols, tiles))
    assert cursor == len(data)
    return tensors


def dct_basis(n, k, j):
    return math.sqrt((1 if k == 0 else 2) / n) * math.cos(math.pi * (j + 0.5) * k / n)


def dense_reference(tensor):
    """Direct four-index inverse formula, not the C++ separable implementation."""
    _, rows, cols, tiles = tensor
    weights = [0.0] * (rows * cols)
    squared_bound = 0.0
    for row, col, r, c, delta, q in tiles:
        squared_bound += r * c * delta * delta / 4
        for i in range(r):
            for j in range(c):
                weights[(row + i) * cols + col + j] = sum(
                    dct_basis(r, u, i) * delta * q[u * c + v] * dct_basis(c, v, j)
                    for u in range(r) for v in range(c))
    return weights, squared_bound


def write_floats(path, values):
    Path(path).write_bytes(struct.pack("<" + "f" * len(values), *values))
    # Return the actual f32 values, not their pre-rounding Python double values.
    return read_floats(path)


def read_floats(path):
    data = Path(path).read_bytes()
    assert len(data) % 4 == 0
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


def handcrafted(path, cols, entries, payload):
    """Known rANS states, built from the wire spec without calling the C++ writer."""
    table = b"".join(struct.pack("<BBH", s, 0, f) for s, f in entries)
    chunk = bytearray(64)
    struct.pack_into("<4sHHQQQHHfIHBBIIII", chunk, 0,
                     b"ACNK", 64, 0, 0, 0, 0, 1, cols, 0.5,
                     cols, len(entries), 12, 1, len(payload), 64 + len(table) + len(payload), 0, 0)
    chunk.extend(table + payload)
    struct.pack_into("<I", chunk, 56, zlib.crc32(chunk))
    entry = bytearray(128)
    entry[:7] = b"fixture"
    struct.pack_into("<HBBBBHQQIIQQQ", entry, 48, 1, 2, 1, 1, 1, 0,
                     1, cols, 1, cols, 1, 192, len(chunk))
    header = bytearray(struct.pack("<8sHHIIIQQQIIQ", MAGIC, 1, 0, 64, 0, 1,
                                  64, 128, 192 + len(chunk), zlib.crc32(entry), 0, 0))
    struct.pack_into("<I", header, 52, zlib.crc32(header))
    Path(path).write_bytes(header + entry + chunk)


def main():
    capabilities = run("backends")
    available = "AVX2 DCT available: yes" in capabilities
    selected = "avx2" if available else "scalar"
    assert f"Automatic DCT backend: {selected}" in capabilities
    assert "Coefficient multiply: scalar; file format: AMAK 1.0" in capabilities
    for backend in ("scalar", "auto", "avx2"):
        text = run("bench-dct", "--size", 7, "--iterations", 5, "--warmup", 1,
                   "--dct-backend", backend, fail=backend == "avx2" and not available)
        if backend == "avx2" and not available:
            continue
        assert f"Selected DCT backend: {selected if backend == 'auto' else backend}" in text
        assert "not an end-to-end speedup" in text
        rows = [line.split() for line in text.splitlines() if line.startswith(("f64 forward", "f32 forward", "f64 inverse"))]
        assert len(rows) == 3
        for row in rows:
            assert float(row[2]) >= 0 and float(row[3]) >= 0  # clocks may be coarse; never require a speed ratio
            assert 0 <= float(row[5]) < 1e-10
    for options in (("--size", 0), ("--size", 257), ("--iterations", 0),
                    ("--warmup", 1000001), ("--dct-backend", "unknown")):
        run("bench-dct", *options, fail=True)
    run("backends", "unexpected", fail=True)
    with tempfile.TemporaryDirectory(prefix="amak-cli-") as directory:
        root = Path(directory)
        weights_path, model = root / "weights.f32", root / "model.amak"
        x_path, y_path = root / "x.f32", root / "y.f32"
        rows, cols, batch = 7, 11, 3
        source = write_floats(weights_path, [math.sin(i * 0.53) * 0.2 for i in range(rows * cols)])
        inputs = write_floats(x_path, [math.cos(i * 0.37) for i in range(batch * cols)])
        pack_options = ("--rows", rows, "--cols", cols, "--tile-rows", 3, "--tile-cols", 4, "--name", "test.weight")
        run("pack", weights_path, model, *pack_options)
        assert "W=[7,11]" in run("inspect", model)
        assert "validated" in run("verify", model)
        tensor, = parse_container(model)
        assert tensor[:3] == ("test.weight", rows, cols)
        dense, bound = dense_reference(tensor)
        assert sum((a - b) ** 2 for a, b in zip(dense, source)) <= bound + 1e-12
        expected = [sum(inputs[b * cols + c] * dense[r * cols + c] for c in range(cols))
                    for b in range(batch) for r in range(rows)]
        packed_bytes = model.read_bytes()
        for backend in ("scalar", "auto", "avx2"):
            if backend == "avx2" and not available:
                run("run", model, x_path, y_path, "--batch", batch, "--dct-backend", backend, fail=True)
                continue
            result_bytes = None
            for prefetch in (0, 1, 2, 8):
                text = run("run", model, x_path, y_path, "--batch", batch, "--prefetch", prefetch,
                           "--tensor", "test.weight", "--dct-backend", backend)
                assert f"DCT backend: {selected if backend == 'auto' else backend}" in text
                output = read_floats(y_path)
                assert len(output) == batch * rows
                assert all(math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-6) for a, b in zip(output, expected))
                # Bitwise equality applies within a backend, not across different reductions.
                if result_bytes is not None:
                    assert y_path.read_bytes() == result_bytes
                result_bytes = y_path.read_bytes()
        assert model.read_bytes() == packed_bytes
        run("run", model, x_path, y_path, "--batch", batch, "--dct-backend", "unknown", fail=True)

        repeat = root / "repeat.amak"
        run("pack", weights_path, repeat, *pack_options)
        assert repeat.read_bytes() == model.read_bytes()
        run("pack", weights_path, repeat, *pack_options, "--step", 0.025)
        fixed, = parse_container(repeat)
        step = struct.unpack("<f", struct.pack("<f", 0.025))[0]
        assert all(tile[4] == step for tile in fixed[3])

        # Reader interoperability: signed -128, then the two-symbol golden state.
        fixture = root / "fixture.amak"
        handcrafted(fixture, 1, [(0, 4096)], b"\x00\x00\x80\x00")
        run("verify", fixture)
        write_floats(x_path, [2, -3])
        run("run", fixture, x_path, y_path, "--batch", 2)
        assert read_floats(y_path) == [-128.0, 192.0]
        handcrafted(fixture, 2, [(128, 2048), (129, 2048)], b"\x00\x10\x00\x02")
        run("verify", fixture)
        write_floats(x_path, [1, 0, 0, 1])
        run("run", fixture, x_path, y_path, "--batch", 2)
        assert all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(read_floats(y_path), [0.5 / math.sqrt(2), -0.5 / math.sqrt(2)]))

        # Failures must be normal errors, not hangs or crashes, and protect inputs.
        original_file = model.read_bytes()
        for bad_options in (("--rows", -1, "--cols", 11),
                            ("--rows", "18446744073709551616", "--cols", 11),
                            ("--rows", 7, "--cols", 0),
                            (*pack_options, "--unknown", 1),
                            (*pack_options, "--rows", 7),
                            (*pack_options, "--step", "nan"),
                            (*pack_options, "--step", "1e-99"),
                            (*pack_options, "--step", 0),
                            ("--rows", 7, "--cols", 11, "--tile-cols", 257),
                            ("--rows", 1048576, "--cols", 100)):
            run("pack", weights_path, model, *bad_options, fail=True)
            assert model.read_bytes() == original_file
        run("nonsense", fail=True)
        run("pack", weights_path, model, "--rows", fail=True)
        run("run", model, x_path, y_path, "--prefetch", 9, fail=True)
        run("run", model, x_path, y_path, "--batch", -1, fail=True)
        run("run", model, x_path, y_path, "--tensor", "missing", fail=True)
        run("run", model, x_path, y_path, "--batch", 1, fail=True)  # wrong input length
        write_floats(x_path, [math.nan] * cols)
        run("run", model, x_path, y_path, fail=True)
        write_floats(weights_path, [math.nan] * (rows * cols))
        run("pack", weights_path, model, *pack_options, fail=True)
        assert model.read_bytes() == original_file
        run("pack", weights_path, weights_path, *pack_options, fail=True)
        run("run", model, x_path, model, fail=True)
        run("run", model, x_path, x_path, fail=True)
        alias = root / "alias.f32"
        os.link(x_path, alias)
        run("run", model, x_path, alias, fail=True)

        # inspect is intentionally lazy; verify and both execution modes catch data CRC damage.
        corrupt = root / "corrupt.amak"
        damaged = bytearray(original_file)
        damaged[-1] ^= 1
        corrupt.write_bytes(damaged)
        assert "Only header/directory" in run("inspect", corrupt)
        run("verify", corrupt, fail=True)
        write_floats(x_path, inputs)
        sentinel = y_path.read_bytes()
        for prefetch in (0, 2):
            run("run", corrupt, x_path, y_path, "--batch", batch, "--prefetch", prefetch, fail=True)
            assert y_path.read_bytes() == sentinel

        demo = root / "demo.amak"
        text = run("demo", demo, "--rows", 9, "--cols", 13, "--tile-rows", 4,
                   "--tile-cols", 5, "--batch", 3, "--iterations", 1)
        assert "Synchronous/prefetch outputs: identical" in text
        assert "includes quantization error" in text
        assert parse_container(demo)[0][1:3] == (9, 13)
        assert f"Selected DCT backend: {selected}" in text
        assert "Scalar/selected max absolute difference:" in text
        before = demo.read_bytes()
        scalar_demo = run("demo", demo, "--rows", 9, "--cols", 13, "--tile-rows", 4,
                          "--tile-cols", 5, "--batch", 3, "--iterations", 1, "--dct-backend", "scalar")
        assert "Selected DCT backend: scalar" in scalar_demo
        assert demo.read_bytes() == before  # backend-independent scalar offline encoding
        if not available:
            run("demo", demo, "--dct-backend", "avx2", fail=True)
            assert demo.read_bytes() == before
        assert "not a transformer runner" in run("--help")
    print("[pass] CLI, backend dispatch/benchmarks, independent binary/rANS/DCT reference, and input protection")


if __name__ == "__main__":
    main()
