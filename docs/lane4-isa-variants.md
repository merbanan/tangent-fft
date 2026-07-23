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

The scalar source contains no intrinsics and is built with
`-fno-tree-vectorize`. The SSE hot path contains no C intrinsics: it uses
FFmpeg's vendored `x86inc.asm` calling-convention/register macros and NASM.
Disassembly checks confirm that it contains no VEX instructions, the non-FMA
AVX objects contain no FMA instructions, and the C hot loop contains no
packed arithmetic instructions.

The correctness harness checks all available forward and inverse variants
against a long-double direct DFT through 512 and against the existing radix-2
implementation through the lane4 size limit. The forward worst relative
maximum error is `3.516e-07` for the portable/SSE family and `3.238e-07` for
the AVX family. The corresponding inverse maxima are `1.077e-07` and
`1.115e-07`.

## Ryzen 9 3900X measurements

Median microseconds from the command above; lower is better:

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

| N | AVX | AVX+FMA | AVX2 | AVX2+FMA | FFmpeg |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.030 | 0.040 | 0.030 | 0.040 | 0.060 |
| 32 | 0.050 | 0.050 | 0.050 | 0.040 | 0.070 |
| 64 | 0.070 | 0.070 | 0.070 | 0.070 | 0.120 |
| 128 | 0.090 | 0.100 | 0.090 | 0.090 | 0.160 |
| 256 | 0.180 | 0.180 | 0.180 | 0.180 | 0.340 |
| 512 | 0.370 | 0.370 | 0.380 | 0.370 | 0.690 |
| 1024 | 0.800 | 0.790 | 0.800 | 0.790 | 1.460 |
| 2048 | 1.850 | 1.800 | 1.840 | 1.810 | 3.230 |
| 4096 | 4.270 | 4.140 | 4.240 | 4.120 | 7.530 |
| 8192 | 9.690 | 9.711 | 9.610 | 9.490 | 20.990 |

The assembly SSE path is at or ahead of FFmpeg at 16--256 and 4096--8192. It
trails by 2.9%, 1.4%, and 3.1% at 512, 1024, and 2048 respectively. Thus it
has reached close parity, not a strict win at every size. At 8192 it is 22.3%
faster. Every 256-bit lane4 path is clearly faster than FFmpeg at every size;
`lane4-avx2-fma` is 1.5--2.2 times as fast. Small differences among
instruction-identical SSE aliases are measurement noise. The raw output,
including minimum times, sample counts, and checksums, is in
`benchmark-simd.csv`.

### Normalized inverse

Inverse execution reuses the selected forward kernel, then performs one
in-place reverse-and-`1/N` scale pass. These medians include that pass:

| N | C | SSE2 | AVX | AVX+FMA | AVX2 | AVX2+FMA | FFmpeg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.100 | 0.050 | 0.040 | 0.040 | 0.040 | 0.040 | 0.070 |
| 32 | 0.190 | 0.070 | 0.050 | 0.060 | 0.060 | 0.060 | 0.080 |
| 64 | 0.390 | 0.120 | 0.080 | 0.080 | 0.080 | 0.080 | 0.140 |
| 128 | 0.860 | 0.220 | 0.140 | 0.120 | 0.110 | 0.110 | 0.190 |
| 256 | 1.540 | 0.360 | 0.210 | 0.210 | 0.210 | 0.210 | 0.370 |
| 512 | 3.470 | 0.840 | 0.450 | 0.450 | 0.460 | 0.450 | 0.770 |
| 1024 | 7.540 | 1.640 | 0.960 | 0.950 | 0.960 | 0.960 | 1.630 |
| 2048 | 16.450 | 3.650 | 2.150 | 2.120 | 2.160 | 2.390 | 3.570 |
| 4096 | 36.070 | 7.880 | 5.460 | 4.770 | 4.880 | 4.740 | 8.190 |
| 8192 | 77.740 | 17.480 | 11.020 | 10.880 | 11.090 | 10.760 | 22.110 |

The complete sixteen-implementation inverse output is in
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
