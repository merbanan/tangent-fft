# Lane4 x86 ISA variants

## Coverage

The common harness exposes every packed floating-point x86 generation
supported by the Ryzen 9 3900X test host:

| Harness name | Build boundary | Representation |
|---|---|---|
| `lane4-c` | C11, automatic vectorization disabled | four scalar complex lanes |
| `lane4-sse` | SSE, explicitly no SSE2 or AVX | four real and four imaginary floats |
| `lane4-sse2` | SSE2, explicitly no AVX | four real and four imaginary floats |
| `lane4-sse3` | SSE3, explicitly no AVX | four real and four imaginary floats |
| `lane4-ssse3` | SSSE3, explicitly no AVX | four real and four imaginary floats |
| `lane4-sse4.1` | SSE4.1, explicitly no AVX | four real and four imaginary floats |
| `lane4-sse4.2` | SSE4.2, explicitly no AVX | four real and four imaginary floats |
| `lane4-avx` | AVX, explicitly no AVX2 or FMA | four interleaved complex floats |
| `lane4-avx-fma` | AVX and FMA, explicitly no AVX2 | four interleaved complex floats |
| `lane4-avx2` | AVX2, explicitly no FMA | four interleaved complex floats |
| `lane4-avx2-fma` | AVX2 and FMA | four interleaved complex floats |

MMX is not included because it has no packed floating-point arithmetic.
SSE4a, F16C, AES, SHA, and the packed-integer extensions do not add a useful
single-precision FFT primitive. The host has no AVX-512.

Every entry is selected explicitly by the public `fft_algorithm` enum.
`fft_plan_supports` uses runtime CPU feature detection, so a function compiled
for a newer ISA is never entered on a CPU lacking that ISA.

## Structure

All versions apply the same outer decomposition:

```text
N = 4 M
F_r[q] = DFT_M(x[4m+r])
X[q + pM] = DFT_4(F_r[q] W_N^(rq))[p]
```

The portable/SSE implementation uses a simple radix-2 inner schedule to keep
the C and 128-bit versions compact and auditable. Four inner transforms still
share every butterfly. Four adjacent output frequencies are transposed,
twiddled, passed through one vertical FFT4, and stored in natural order.

The 256-bit implementations use the tuned mixed-radix-4 schedule:
register-resident FFT4/FFT8 leaves, fused FFT16/FFT32 parents, stage-major
twiddles, fixed 64/128 paths, and the fused transpose/FFT4 finish. The source
is compiled four times, with complex multiplication selected at compile time:

```text
without FMA: mul + mul + addsub
with FMA:    mul + fmaddsub
```

AVX2 adds integer operations and gathers but no new packed-float add,
multiply, shuffle, or FMA operation. Consequently the AVX/AVX2 versions of
this float-only kernel are expected to be nearly identical. SSE3 through
SSE4.2 likewise add no operation that improves the split real/imag inner
loop; their separately bounded objects intentionally converge to the SSE
instruction sequence. Keeping the entries separate verifies this rather than
assuming that a newer feature flag is faster.

## Reproduction

Build flags for every object are visible in `Makefile`. Run:

```sh
make clean
make
make test
taskset -c 2 ./fft_harness --bench --min-power 4 --max-power 13 \
  --target-ms 1000 --csv benchmark-simd.csv
```

The scalar source contains no intrinsics and is built with
`-fno-tree-vectorize`. Disassembly checks confirm that the SSE objects contain
no VEX instructions, the non-FMA AVX objects contain no FMA instructions, and
the C hot loop contains no packed arithmetic instructions.

The correctness harness checks all available variants against a
long-double direct DFT through 512 and against the existing radix-2
implementation through the lane4 size limit. The observed worst relative
maximum error is `3.516e-07` for the portable/SSE family and `3.238e-07` for
the AVX family.

## Ryzen 9 3900X measurements

Median microseconds from the command above; lower is better:

| N | C | SSE | SSE2 | SSE3 | SSSE3 | SSE4.1 | SSE4.2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.090 | 0.060 | 0.060 | 0.060 | 0.060 | 0.060 | 0.060 |
| 32 | 0.180 | 0.090 | 0.090 | 0.090 | 0.090 | 0.090 | 0.090 |
| 64 | 0.380 | 0.170 | 0.170 | 0.170 | 0.170 | 0.170 | 0.170 |
| 128 | 0.840 | 0.350 | 0.350 | 0.350 | 0.350 | 0.350 | 0.350 |
| 256 | 1.870 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| 512 | 4.150 | 1.640 | 1.640 | 1.630 | 1.640 | 1.630 | 1.650 |
| 1024 | 7.420 | 2.930 | 2.940 | 2.940 | 2.900 | 2.930 | 2.950 |
| 2048 | 16.210 | 6.380 | 6.370 | 6.370 | 6.310 | 6.371 | 6.380 |
| 4096 | 35.580 | 13.840 | 13.850 | 13.840 | 13.710 | 13.840 | 13.860 |
| 8192 | 76.700 | 29.590 | 29.600 | 29.610 | 29.330 | 29.600 | 29.620 |

| N | AVX | AVX+FMA | AVX2 | AVX2+FMA | FFmpeg |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.030 | 0.030 | 0.030 | 0.030 | 0.060 |
| 32 | 0.040 | 0.040 | 0.040 | 0.040 | 0.070 |
| 64 | 0.070 | 0.070 | 0.060 | 0.070 | 0.120 |
| 128 | 0.110 | 0.110 | 0.110 | 0.110 | 0.210 |
| 256 | 0.210 | 0.220 | 0.220 | 0.220 | 0.420 |
| 512 | 0.480 | 0.460 | 0.460 | 0.460 | 0.860 |
| 1024 | 0.800 | 0.780 | 0.800 | 0.790 | 1.480 |
| 2048 | 1.830 | 1.810 | 1.830 | 1.810 | 3.250 |
| 4096 | 4.310 | 4.110 | 4.260 | 4.150 | 7.650 |
| 8192 | 9.650 | 9.650 | 9.870 | 9.730 | 20.710 |

The small differences among instruction-equivalent columns are benchmark
noise and code-placement effects, not evidence that SSSE3 or AVX2 provides a
new float FFT operation. The raw output, including all original algorithms,
minimum times, sample counts, and checksums, is in `benchmark-simd.csv`.
