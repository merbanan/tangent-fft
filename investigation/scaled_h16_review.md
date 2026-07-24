# Scaled H16 hybrid: feasibility review and benchmark

## Verdict

The H8 straight-line circuit in the proposal is mathematically correct under
its stated physical input contract `(a, 2b, ..., 2h)`. The H16 construction
from two such H8 circuits is also correct and uses 62 real operations after
its inputs have acquired the required scales, versus 64 additions for the
ordinary H16 network.

The proposal is nevertheless unlikely to beat the existing lane4 or FFmpeg
FFT for the target sizes:

- The local arithmetic saving is only 2/64, or 3.125%.
- An isolated H16 must first double 14 inputs. That makes it 76 operations,
  not 62, unless the scales arrive from a larger recursion.
- `t = r6/2` cannot be represented by a metadata change and then added to
  `A`. Addition requires the two physical operands to have the same scale.
  Alman and Rao count the divide by two as an operation.
- The AVX2 structure-of-arrays mapping needs 16 coordinate vectors before
  temporaries. AVX2 has only 16 architectural YMM registers, so the proposed
  real/imaginary schedule necessarily spills or streams.
- The current lane4 stages are complex DFT4 networks with rotations and
  twiddles, not WHT networks. Replacing a vaguely “Hadamard-like” region does
  not preserve the DFT.

My pre-implementation estimate was below a 10% chance of beating the current
lane4 AVX2/FMA path as described. The benchmarks below support that estimate.
Pairing the two H8 branches is a useful local SIMD optimization, but it does
not remove the global layout and twiddle-network costs.

## Relation to the literature

The missing exact integration is important. Alman and Rao do not substitute
an H8 or H16 circuit into an arbitrary FFT butterfly. They first commute all
split-radix input additions to the front of the transform. The resulting
`H'` transformation is a permutation of a direct sum of WHTs. Those WHTs are
then followed by the recursively structured twiddle network.

Their full paper also says that their arithmetic improvement uses the H8
rigidity decomposition and that, despite trying, they did not obtain an
improved arithmetic algorithm from the H16 decomposition. Their asymptotic
DFT result is `15/4 N log2(N) + O(N)` real operations; it relies on the full
modified-split-radix reduction and the H8 WHT recursion, not on the proposed
local H16/lane4 substitution.

Primary source: Josh Alman and Kevin Rao,
[Faster Walsh-Hadamard and Discrete Fourier Transforms From Matrix
Non-Rigidity](https://arxiv.org/abs/2211.06459), STOC 2023,
[DOI 10.1145/3564246.3585188](https://doi.org/10.1145/3564246.3585188).

## Implemented algorithm

`scaled-h16-hybrid` is a complete, exact FFT rather than an unproven local
substitution. It implements:

1. The recursive split-radix input permutation.
2. The Alman–Rao `H'` partition generated at plan time.
3. One WHT on every `H'` subset.
4. The proposal's scaled-H16 circuit recursively inside those WHTs.
5. The uprooted split-radix twiddle network, including a reduced zero-twiddle
   path.

The implementation is scalar C in `h16_hybrid.c`, uses only `float` for
samples and coefficients, has no intrinsics, supports forward and normalized
inverse transforms, and keeps plan construction outside the timed region.

This is a faithful way to give the proposed H16 circuit a mathematically valid
role in a DFT. It also exposes why the scale is not free. In a radix-16 WHT
recursion, the H16 combination work has leading term
`(31/32) N log2(N)`, but the required leaf scaling contributes `O(N)` extra
work. Ignoring memory and dependency costs, the conventional arithmetic count
does not cross the ordinary WHT until roughly `log2(N) > 32`. In the FFT
sizes tested here, the largest WHT subset is only 64.

### Paired AVX2 implementation

`scaled-h16-paired-avx2` implements the corrected follow-up design in
`simd_paired_scaled_h16_design_notes.txt`. Partitions are sorted by size and
processed in groups of four. For each H16 coordinate, one YMM register holds:

```text
[P(t0), Q(t0), P(t1), Q(t1), P(t2), Q(t2), P(t3), Q(t3)]
```

Real and imaginary components use separate arrays. The handwritten
`h16_paired_avx2.asm` kernel applies the identical scaled-H8 DAG to all eight
lanes. Each component uses exactly 22 packed additions/subtractions and one
packed multiply by 0.5, with no lane-crossing instructions or spills.
Incomplete batches use the scalar circuit, so every supported size remains
available.

The producer computes the child WHTs and writes the alternating P/Q layout
directly. This avoids conversion inside the assembly kernel, but the
irregular H′ input gather and output scatter remain part of timed execution.

## Correctness

The standard harness passes:

- forward and inverse comparison with a long-double direct DFT through 512;
- cross-checks through `2^22`;
- AddressSanitizer and UndefinedBehaviorSanitizer with leak detection disabled
  because LeakSanitizer is unavailable under the execution environment's
  tracing.

Worst normalized error reported for both H16 implementations was `1.688e-7`
forward and `3.017e-8` inverse, equal to the scalar split-radix result in this
run.

## Benchmark

Host: AMD Ryzen 9 3900X, GCC 15.2.0, one process pinned to CPU 2. Times are
median microseconds. Setup, allocation, permutation generation, partition
generation, and twiddle generation are excluded.

```sh
make
taskset -c 2 ./fft_harness --bench --min-power 4 --max-power 13 \
  --target-ms 100 --csv benchmark-h16-hybrid.csv
```

| N | radix-2 | split-radix | tangent C | lane4 C | lane4 AVX2/FMA | FFmpeg native | H16 scalar | H16 paired AVX2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.130 | 0.170 | 0.110 | 0.090 | 0.030 | 0.060 | 0.180 | 0.180 |
| 32 | 0.290 | 0.280 | 0.170 | 0.140 | 0.040 | 0.060 | 0.270 | 0.270 |
| 64 | 0.540 | 0.600 | 0.400 | 0.310 | 0.060 | 0.100 | 0.530 | 0.530 |
| 128 | 1.220 | 1.310 | 0.810 | 0.670 | 0.100 | 0.170 | 1.030 | 1.020 |
| 256 | 2.700 | 2.840 | 1.760 | 1.500 | 0.180 | 0.340 | 2.160 | 2.140 |
| 512 | 5.920 | 6.190 | 3.730 | 3.320 | 0.370 | 0.690 | 4.580 | 4.470 |
| 1024 | 12.730 | 13.330 | 7.920 | 7.380 | 0.810 | 1.470 | 9.830 | 8.940 |
| 2048 | 27.600 | 28.770 | 16.880 | 16.140 | 1.810 | 3.200 | 21.060 | 18.820 |
| 4096 | 59.761 | 61.470 | 36.410 | 35.330 | 4.370 | 7.600 | 44.420 | 39.270 |
| 8192 | 134.390 | 131.660 | 82.660 | 76.371 | 10.320 | 21.210 | 92.191 | 81.790 |

At 8192 the paired design is 11.3% faster than the scalar H16 implementation
and 1.64 times as fast as radix-2. It also narrowly beats tangent C in this
run. It remains:

- 7.1% slower than lane4 C;
- 3.86 times slower than FFmpeg native;
- 7.93 times slower than lane4 AVX2/FMA.

The complete 28-implementation CSV, including minima, sample counts, and
checksums, is `benchmark-h16-hybrid.csv`.

## What would have to change

The experiment confirms that pairing the DAGs is worthwhile but insufficient.
A further SIMD version would need to preserve the P/Q layout across the
surrounding H′ producer and twiddle consumer so the irregular gather, scalar
packing, and scatter disappear. AVX-512's 32 registers and `vscalefps` make
deeper fusion more plausible, but the target Ryzen 3900X does not provide
AVX-512.
