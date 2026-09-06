#pragma once

namespace amak {

// Execution policy only: never serialized into an AMAK file.
enum class DctBackend { automatic, scalar, avx2 };

const char* dct_backend_name(DctBackend backend) noexcept;
bool avx2_dct_compiled() noexcept;
// Includes the compiler's CPU/OS AVX-state availability check, not just CPUID.
bool avx2_dct_available() noexcept;
// auto falls back to scalar. Explicit avx2 throws if unavailable in this build
// or on the executing machine. Invalid enum values also throw.
DctBackend resolve_dct_backend(DctBackend requested);

} // namespace amak
