# Tangent FFT comparison

This repository contains sixteen single-precision, forward, complex FFT
benchmark entries:

- iterative radix-2 Cooley–Tukey in C;
- recursive conjugate-pair split-radix in C;
- a planned, non-recursive tangent FFT in C;
- `tangent-x86-asm`, an AVX2/FMA tangent FFT for x86-64;
- `lane4-c`, a plain-C lane-factorized FFT compiled with automatic
  vectorization disabled;
- lane4 kernels for SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX,
  AVX+FMA, AVX2, and AVX2+FMA;
- FFmpeg `libavutil` AVTX, built locally from pinned source.

Every algorithm uses `float` samples, coefficients, twiddles, and scale
factors. All compute the conventional unnormalised transform

```text
X[k] = sum_j x[j] exp(-2*pi*i*j*k/N).
```

## Reproducible build and benchmark

```sh
make
make test
make bench
make ffmpeg-cycles
make lane4-experiment
```

The repository includes the pinned FFmpeg source. The build produces a local
static `libavutil`; it does not use a system FFmpeg installation or download
anything. NASM must already be available in `PATH` (or supplied with
`make NASM=/path/to/nasm`). Provenance and licenses are recorded in
`third_party/README.md`.

The default benchmark prints every requested power of two from 16 through
8192: 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, and 8192. Other ranges
remain available:

```sh
./fft_harness --bench --min-power 4 --max-power 13 --target-ms 1000
./fft_harness --bench --min-power 4 --max-power 22 --csv benchmark.csv
./fft_harness --help
```

Plan allocation and coefficient generation are excluded from execution
timings. The FFmpeg row includes the copies into and out of its aligned AVTX
buffers, matching this project's in-place public API. The harness prints all
results to the console and can also write CSV.

Each x86 lane4 entry is runtime-gated by CPUID. `tangent-x86-asm` is enabled
when the host is x86-64 with AVX2 and FMA. Portable C algorithms remain
available on other targets.

## Lane-factorized implementations

For `N=4M`, every lane4 implementation computes the four length-`M`
transforms from `x[4m+r]` in parallel. The plain-C implementation makes the
decomposition explicit. The SSE family uses separate four-float real and
imaginary vectors. The 256-bit family treats each YMM register as four
interleaved complex lanes.

The tuned `lane4-avx2-fma` kernel
computes the four length-`M` transforms from `x[4m+r]` simultaneously. Its
mixed-radix input permutation is fused into FFT4/FFT8 leaves, upper stages are
block-major radix-4, and the final twiddles feed a fused 4x4 transpose plus
vector FFT4. Complete 16- and 32-point transforms stay in registers.
Fused vector FFT16 leaves and fixed 64/128 paths carry child rows across
work-array boundaries.

This schedule prioritizes regular full-width SIMD and memory locality over
minimum scalar operation count. On this host it beats tangent assembly and
FFmpeg at every tested power of two from 16 through 8192. The derivation,
prototype history, cycle results, limitations, and research references are in
[`docs/lane-factorized-fft.md`](docs/lane-factorized-fft.md).
The per-ISA organization, compiler boundaries, and comparative measurements
are in [`docs/lane4-isa-variants.md`](docs/lane4-isa-variants.md).
A detailed derivation, execution walkthrough, and literature lineage are in
[`description.md`](description.md).

## Tangent implementation structure

Bernstein's four transform families—`newfft`, `newfftS`, `newfftS2`, and
`newfftS4`—are represented in the reusable plan. The hot paths do not recurse.
Plan construction:

1. computes the conjugate-pair input permutation;
2. flattens the recursive tree into compact schedules grouped by level and
   transform family;
3. precomputes all `float` tangent factors and output-scale ratios.

The portable tangent path executes the flat schedule in post-order. The
assembly path goes further:

- the input permutation is fused into fixed FFT8/FFT16 leaf kernels;
- leaf data stays in XMM/YMM registers until its transform is complete;
- a dedicated register-resident FFT32 kernel removes the small-size call tree;
- a fixed FFT64 carries its final FFT16 child into the parent, removing four
  stores and four reloads;
- four complex samples are processed per YMM register;
- FMA implements general and reduced tangent complex products;
- larger nodes are batched by level and transform family, with no recursive
  calls, per-butterfly branches, or per-node copies.

This organization follows the useful structural ideas in FFmpeg's
`libavutil/x86/tx_float.asm`: pre-shuffled input, fixed register-resident small
kernels, split-radix quarter streams, and batched upper stages. The tangent
arithmetic and code here are independently implemented.

The scale recurrence is

```text
s[N,k] = s[N/4,k mod N/4] *
         (cos(2*pi*(k mod N/4)/N), when k mod N/4 <= N/8;
          sin(2*pi*(k mod N/4)/N), otherwise).
```

It changes a general twiddle into a form based on `1+i*tan` or `cot-i`,
reducing real multiplications. The arithmetic counts reported by the harness
are:

```text
radix-2:     5*N*lg(N) - 10*N + 16                    (N >= 4)
split-radix: 4*N*lg(N) -  6*N + 8                     (N >= 2)
tangent:    (34/9)*N*lg(N) - (124/27)*N - 2*lg(N)
             - (2/9)*(-1)^lg(N)*lg(N)
             + (16/27)*(-1)^lg(N) + 8                 (N >= 2)
```

## Current Ryzen 9 3900X results

The checked-in `benchmark.csv` comes from:

```sh
./fft_harness --bench --min-power 4 --max-power 13 \
  --target-ms 1000 --csv benchmark.csv
```

Median execution times in microseconds:

| N | radix-2 | tangent C | tangent-x86-asm | lane4-avx2-fma | FFmpeg AVTX |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.110 | 0.090 | 0.050 | 0.030 | 0.050 |
| 32 | 0.230 | 0.190 | 0.050 | 0.040 | 0.060 |
| 64 | 0.530 | 0.420 | 0.090 | 0.060 | 0.110 |
| 128 | 1.230 | 0.850 | 0.190 | 0.090 | 0.180 |
| 256 | 2.710 | 1.830 | 0.350 | 0.170 | 0.340 |
| 512 | 6.010 | 3.850 | 0.710 | 0.370 | 0.690 |
| 1024 | 13.010 | 8.260 | 1.430 | 0.790 | 1.470 |
| 2048 | 28.210 | 17.530 | 3.080 | 1.790 | 3.230 |
| 4096 | 60.930 | 37.960 | 6.950 | 4.150 | 7.590 |
| 8192 | 134.980 | 85.740 | 18.470 | 9.800 | 21.700 |

Cycle-accurate comparisons are recorded in the lane-factorized report.

The vendored FFmpeg FMA3 split-radix leaves are also patched with exact
sign-folding reductions. A five-pair cycle A/B against the unmodified pinned
source measured median gains of 0.17–0.96% from 64 through 8192 on this host.
`make ffmpeg-cycles` runs the longer-batch FFmpeg-only cycle harness.

The investigation of FFmpeg's x86 FFT optimization TODOs, including retained
and rejected assembly experiments, is in
[`docs/ffmpeg-todo-investigation.md`](docs/ffmpeg-todo-investigation.md).

Results vary with CPU, frequency policy, compiler, and system load; regenerate
`benchmark.csv` on the target machine rather than treating these values as
portable constants.

## Validation and FFmpeg's own harness

`make test` compares every path, for three input families and every power of
two through 512, against a direct DFT accumulated in `long double`. It then
cross-checks through `2^22`. The current worst relative error for
`tangent-x86-asm` is `1.413e-07`.

FFmpeg does have an FFT test/benchmark harness:
`third_party/ffmpeg/tests/checkasm/av_tx.c`. It checks AVTX implementations
against the C reference and uses checkasm's `bench_new` timing facility. This
project uses its own common harness so radix-2, split-radix, tangent, and
FFmpeg are measured through the same API and reporting path.

The tangent implementation follows Daniel J. Bernstein, *The tangent FFT*
(2007), and the equivalent scaled conjugate-pair presentation by Steven G.
Johnson and Matteo Frigo, *A Modified Split-Radix FFT With Fewer Arithmetic
Operations* (2007).
