# Lane2 SSE FFT

## Factorization and SIMD layout

For `N=2M`, split the input into its even and odd residue classes:

```text
F0[q] = DFT_M(x[2m])
F1[q] = DFT_M(x[2m+1])
T[q]  = W_N^q F1[q]
X[q]  = F0[q] + T[q]
X[q+M]= F0[q] - T[q].
```

One baseline-SSE register holds two complete interleaved complex values:

```text
[F0.re, F0.im, F1.re, F1.im].
```

Every packed operation in the inner FFT therefore evaluates the same
butterfly in the even and odd transforms. Unlike lane4-SSE, a logical row
uses one register rather than separate real and imaginary registers.

## Planned execution

`lane2_sse_plan_create` performs all allocation and coefficient generation
outside the timed transform:

1. build the mixed radix-2/radix-4 input permutation;
2. store upper-stage roots in execution order;
3. duplicate each root's real component and store its imaginary component as
   `[-wi,+wi,-wi,+wi]`;
4. pack two distinct final twiddles into each finish-table entry;
5. allocate aligned scratch rows.

For an interleaved value vector `v`, complex multiplication is:

```text
swapped = shuffle(v, 0xb1)
result  = v * wr + swapped * [-wi,+wi,-wi,+wi].
```

This needs only baseline SSE instructions. The assembly object contains no
AVX/VEX encoding.

Execution is entirely in `lane2_sse_stage.asm`:

- even inner levels start with an FFT4 leaf;
- odd inner levels start with a register-resident FFT8 leaf;
- upper transforms use block-major radix-4 stages;
- two adjacent upper butterflies are unrolled;
- the exact `W8`, `-i`, and `W8^3` positions use constants and sign folds;
- hot entries are aligned to 64-byte boundaries.

The FFT8 leaf replaces an earlier FFT2 leaf followed by a scratch-array
radix-4 parent. It keeps all eight rows in the sixteen XMM registers and
writes only completed FFT8 results.

## Fused finish

Two adjacent scratch rows initially contain:

```text
[F0[q],   F1[q]]
[F0[q+1], F1[q+1]].
```

Two 64-bit-lane moves transpose them into:

```text
A = [F0[q], F0[q+1]]
B = [F1[q], F1[q+1]].
```

`B` is multiplied by two packed, frequency-specific twiddles. One packed add
and subtract then evaluate both final FFT2s. The results are stored directly
in the two natural-order output halves.

The normalized inverse API reuses this forward kernel and applies the
repository's common reverse-and-scale identity.

## Measurements

Ryzen 9 3900X wall-clock medians from the checked-in `benchmark.csv`,
in microseconds:

| N | lane2 SSE | lane4 SSE | FFmpeg native | FFmpeg SSE |
|---:|---:|---:|---:|---:|
| 16 | 0.040 | 0.040 | 0.060 | 0.080 |
| 32 | 0.060 | 0.060 | 0.070 | 0.150 |
| 64 | 0.080 | 0.140 | 0.100 | 0.290 |
| 128 | 0.160 | 0.160 | 0.160 | 0.610 |
| 256 | 0.330 | 0.320 | 0.330 | 1.320 |
| 512 | 0.760 | 0.710 | 0.690 | 2.870 |
| 1024 | 1.580 | 1.480 | 1.480 | 6.310 |
| 2048 | 3.660 | 3.330 | 3.210 | 13.850 |
| 4096 | 7.800 | 7.120 | 7.650 | 30.530 |
| 8192 | 20.030 | 16.490 | 21.161 | 71.210 |

The native FFmpeg column auto-dispatches to AVX codelets on this CPU. The
FFmpeg SSE column is a separately created AVTX plan with its feature mask
capped at SSE4.2, making the ISA-matched comparison explicit.

The small-size wall timer has 0.01-microsecond resolution. The more precise
`make lane2-cycles` run measured:

| N | lane2 SSE | lane4 SSE | FFmpeg AVTX |
|---:|---:|---:|---:|
| 16 | 70.346 | 82.593 | 126.956 |
| 32 | 146.773 | 151.632 | 164.032 |
| 64 | 297.637 | 299.484 | 381.457 |
| 128 | 705.700 | 664.852 | 573.539 |
| 256 | 1201.022 | 1156.370 | 1218.370 |
| 512 | 2845.872 | 2642.225 | 2549.535 |
| 1024 | 5998.842 | 5585.499 | 5448.695 |
| 2048 | 13905.959 | 12753.973 | 12302.574 |
| 4096 | 29734.332 | 27060.230 | 28162.750 |
| 8192 | 76531.703 | 62844.578 | 81130.297 |

Lane2 is the fastest 128-bit path at 16 and 32 and is effectively tied with
lane4 at 64. At larger sizes, lane4's four-transform control and coefficient
amortization generally outweighs lane2's more compact register layout.
Lane2 remains faster than FFmpeg at 16--64, 256, and 8192 in this cycle run,
but it is not a universal replacement for lane4-SSE.

## Validation

The common harness passes forward and inverse direct-DFT comparisons through
512 and cross-checks the available lane2 range against radix-2. Its worst
small-size relative maximum errors are:

```text
forward  3.238e-07
inverse  1.059e-07
```

The same suite passes under AddressSanitizer and UndefinedBehaviorSanitizer.
