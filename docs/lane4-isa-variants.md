# Lane4 x86 ISA variants

## Coverage

The common harness exposes every packed floating-point x86 generation
supported by the Ryzen 9 3900X test host. Every entry supports the
unnormalized forward transform and normalized inverse transform:

| Harness name | Build boundary | Representation |
|---|---|---|
| `lane4-c` | C11, automatic vectorization disabled | four scalar complex lanes |
| `lane4-sse` | SSE1 assembly; requires SSE | four real and four imaginary floats |
| `lane4-sse2` | SSE1 assembly; requires SSE2 by API | four real and four imaginary floats |
| `lane4-sse3` | SSE1 assembly; requires SSE3 by API | four real and four imaginary floats |
| `lane4-ssse3` | SSE1 assembly; requires SSSE3 by API | four real and four imaginary floats |
| `lane4-sse4.1` | SSE1 assembly; requires SSE4.1 by API | four real and four imaginary floats |
| `lane4-sse4.2` | SSE1 assembly; requires SSE4.2 by API | four real and four imaginary floats |
| `lane4-avx` | AVX assembly, non-FMA entry | four interleaved complex floats |
| `lane4-avx-fma` | AVX/FMA assembly entry | four interleaved complex floats |
| `lane4-avx2` | AVX assembly, AVX2 runtime boundary | four interleaved complex floats |
| `lane4-avx2-fma` | AVX/FMA assembly, AVX2 runtime boundary | four interleaved complex floats |

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

The portable C implementation deliberately retains a simple radix-2 inner
schedule as an auditable scalar baseline. The 128-bit path is a separate,
FFmpeg-style x86inc/NASM implementation:

- a mixed-radix digit permutation feeds an FFT4 leaf or a register-resident
  FFT8 that occupies all sixteen XMM registers;
- upper work-array passes use linear row pointers and block-major radix-4
  butterflies;
- inner roots are stored as stage-major replicated XMM triplets, avoiding
  scalar broadcasts, index multiplication, and derived root addresses;
- exact `W8`, `-i`, and `W8^3` positions use sign folds and reduced
  multiplications;
- the final four-row transpose, finish twiddles, FFT4, and interleaved stores
  are one assembly kernel.

The public plan-aware transform scheduler, leaves, stages, and finish are all
in `lane4_sse_stage.asm`. Planning, allocation, outer algorithm selection,
and the benchmark remain in C. SSE through SSE4.2 have distinct CPUID/API
names but are aliases of the same assembly entry point.

The 256-bit implementations use a handwritten mixed-radix-4 schedule:
register-resident FFT4/FFT8 leaves, stage-major replicated twiddles,
block-major radix-4 passes, and a fused transpose/FFT4 finish. Complete
16- and 32-point transforms remain in registers. Scalar C builds four opaque
plan types for the public ISA entries, but execution aliases two NASM bodies:

```text
without FMA: mul + mul + addsub
with FMA:    mul + fmaddsub
```

AVX2 adds integer operations and gathers but no new packed-float add,
multiply, shuffle, or FMA operation needed here. Consequently AVX and AVX2
share one body, while AVX+FMA and AVX2+FMA share another. SSE3 through
SSE4.2 likewise add no operation needed by the split real/imag assembly
kernel. Their separately named, runtime-gated entries intentionally call the
same SSE1 instruction sequence. Keeping the entries separate verifies ISA
coverage rather than assuming that a newer feature flag is faster.

## Reproduction

Build flags for every object are visible in `Makefile`. Run:

```sh
make clean
make
make test
taskset -c 2 ./fft_harness --bench --min-power 4 --max-power 13 \
  --target-ms 1000 --csv benchmark-simd.csv
taskset -c 2 ./fft_harness --bench --inverse \
  --min-power 4 --max-power 13 --target-ms 1000 \
  --csv benchmark-inverse.csv
```

The scalar planners contain no intrinsics and are built with
`-fno-tree-vectorize`. Both SSE and AVX hot paths are NASM. The first-party
tree contains no SIMD or timestamp intrinsics; serialized timestamp reads for
the cycle tools are also implemented in `analysis/x86_tsc.asm`. Disassembly
of `lane4_stage_avx` and `lane4_finish_avx` confirms that the non-FMA entry
path contains no FMA instructions. The shared object also contains the
separately gated FMA body.

The correctness harness checks all available forward and inverse variants
against a long-double direct DFT through 512 and against the existing radix-2
implementation through the lane4 size limit. The forward worst relative
maximum error is `3.516e-07` for the portable/SSE family and `3.238e-07` for
the AVX family. The corresponding inverse maxima are `1.077e-07` and
`1.115e-07`.

## Ryzen 9 3900X measurements

The SSE table below is retained from the earlier complete ISA run. Median
microseconds; lower is better:

| N | C | SSE | SSE2 | SSE3 | SSSE3 | SSE4.1 | SSE4.2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.090 | 0.040 | 0.040 | 0.040 | 0.040 | 0.040 | 0.040 |
| 32 | 0.180 | 0.060 | 0.070 | 0.060 | 0.060 | 0.060 | 0.060 |
| 64 | 0.380 | 0.100 | 0.100 | 0.100 | 0.100 | 0.100 | 0.100 |
| 128 | 0.830 | 0.200 | 0.160 | 0.160 | 0.160 | 0.160 | 0.160 |
| 256 | 1.500 | 0.320 | 0.320 | 0.320 | 0.320 | 0.320 | 0.320 |
| 512 | 3.330 | 0.710 | 0.710 | 0.710 | 0.710 | 0.710 | 0.710 |
| 1024 | 7.380 | 1.480 | 1.480 | 1.480 | 1.480 | 1.480 | 1.480 |
| 2048 | 16.150 | 3.330 | 3.330 | 3.330 | 3.340 | 3.350 | 3.340 |
| 4096 | 35.440 | 7.270 | 7.250 | 7.270 | 7.260 | 7.260 | 7.260 |
| 8192 | 76.440 | 16.300 | 16.300 | 16.300 | 16.300 | 16.300 | 16.300 |

The AVX table is the all-assembly run in `benchmark-assembly.csv`:

| N | AVX | AVX+FMA | AVX2 | AVX2+FMA | FFmpeg |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.040 | 0.040 | 0.040 | 0.040 | 0.060 |
| 32 | 0.050 | 0.050 | 0.050 | 0.050 | 0.070 |
| 64 | 0.060 | 0.060 | 0.060 | 0.060 | 0.100 |
| 128 | 0.090 | 0.100 | 0.100 | 0.100 | 0.160 |
| 256 | 0.180 | 0.180 | 0.170 | 0.180 | 0.340 |
| 512 | 0.370 | 0.370 | 0.380 | 0.370 | 0.690 |
| 1024 | 0.830 | 0.810 | 0.820 | 0.810 | 1.470 |
| 2048 | 1.820 | 1.790 | 1.950 | 1.790 | 3.200 |
| 4096 | 4.440 | 4.300 | 4.460 | 4.330 | 7.430 |
| 8192 | 10.070 | 10.170 | 10.120 | 10.060 | 21.420 |

The assembly SSE path is at or ahead of FFmpeg at 16--256 and 4096--8192. It
trails by 2.9%, 1.4%, and 3.1% at 512, 1024, and 2048 respectively. Thus it
has reached close parity, not a strict win at every size. At 8192 it is 22.3%
faster. Every 256-bit lane4 path is clearly faster than FFmpeg at every size;
`lane4-avx2-fma` is 1.40--2.13 times as fast. Small differences among
instruction-identical SSE aliases are measurement noise. The raw output,
including minimum times, sample counts, and checksums, is in
`benchmark-assembly.csv`; the earlier compiler-generated run remains in
`benchmark-simd.csv`.

### Normalized inverse

Inverse execution reuses the selected forward kernel, then performs one
in-place reverse-and-`1/N` scale pass. These medians include that pass:

| N | C | SSE2 | AVX | AVX+FMA | AVX2 | AVX2+FMA | FFmpeg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.100 | 0.070 | 0.040 | 0.050 | 0.040 | 0.040 | 0.070 |
| 32 | 0.190 | 0.070 | 0.060 | 0.060 | 0.060 | 0.060 | 0.080 |
| 64 | 0.400 | 0.120 | 0.070 | 0.070 | 0.070 | 0.070 | 0.110 |
| 128 | 0.700 | 0.190 | 0.110 | 0.120 | 0.110 | 0.110 | 0.180 |
| 256 | 1.540 | 0.370 | 0.210 | 0.210 | 0.210 | 0.210 | 0.370 |
| 512 | 3.400 | 0.780 | 0.460 | 0.450 | 0.460 | 0.450 | 0.770 |
| 1024 | 7.560 | 1.640 | 0.990 | 0.980 | 0.980 | 0.980 | 1.640 |
| 2048 | 16.500 | 3.660 | 2.150 | 2.150 | 2.160 | 2.130 | 3.550 |
| 4096 | 36.210 | 7.840 | 5.120 | 5.030 | 5.160 | 5.030 | 8.170 |
| 8192 | 78.130 | 17.830 | 11.600 | 11.410 | 11.630 | 11.400 | 22.721 |

The complete inverse output for the assembly rewrite is in
`benchmark-assembly-inverse.csv`; the earlier run remains in
`benchmark-inverse.csv`.

## What the timer includes

Each sample surrounds one public `fft_execute` or `fft_inverse_execute` call
with two `clock_gettime` calls. Plan creation, coefficient construction, and
the harness's input reset are outside the interval. The interval does include
normal API dispatch and the internal assembly leaf/stage/finish calls. FFmpeg
additionally uses an indirect AVTX transform call and
copies into and out of its private aligned buffer; those copies are included
because the common API is in-place.

Earlier SSE prototypes used `static inline` intrinsics helpers. Compiler
inspection showed that those helpers were inlined, so they introduced no
function-call overhead. The retained SSE transform uses no intrinsics or C
execution wrapper. The harness now warms every implementation independently
for 20 ms before collection; this prevents CPU-frequency transitions from
making identical aliases appear materially different. At 16--64, timer,
dispatch, and call costs are still a material part of the reported
end-to-end latency.
