# Tangent FFT comparison

This repository contains thirty single-precision complex FFT benchmark
entries, each supporting forward and normalized inverse transforms:

- iterative radix-2 Cooley–Tukey in C;
- recursive conjugate-pair split-radix in C;
- a planned, non-recursive tangent FFT in C;
- `tangent-x86-asm`, an AVX2/FMA tangent FFT for x86-64;
- tangent FFT assembly kernels exposed at SSE, SSE2, SSE3, SSSE3, SSE4.1,
  and SSE4.2 runtime boundaries;
- `lane4-c`, a plain-C lane-factorized FFT compiled with automatic
  vectorization disabled;
- a complete FFmpeg-style x86inc/NASM lane4 SSE execution path, exposed at
  SSE, SSE2, SSE3, SSSE3, SSE4.1, and SSE4.2 runtime boundaries;
- `lane2-sse`, a baseline-SSE assembly FFT that packs two complete
  interleaved complex values in each XMM register;
- `lane2-neon`, an AArch64 NEON assembly port with the same two-complex
  register geometry and ARM-specific FMA/sign scheduling;
- `lane4-neon-fused`, the AArch64 fused lane4-SoA split-radix/Stockham
  kernel with dedicated 16-, 32-, and 64-point leaves;
- `lane8-avx2-fma`, a handwritten-assembly `N=8M` decomposition with eight
  split-complex residue transforms per YMM pair;
- `hw-sse-auto`, a measured lane2/lane4 SSE assembly crossover dispatcher;
- handwritten lane4 assembly kernels for AVX, AVX+FMA, AVX2, and AVX2+FMA;
- FFmpeg `libavutil` AVTX, built locally from pinned source, in both native
  auto-dispatch and reproducible SSE4.2-and-below configurations;
- `scaled-h16-hybrid`, an experimental scalar-C FFT that combines the
  Alman–Rao split-radix uprooting reduction with the investigation's scaled
  H16 WHT circuit;
- `scaled-h16-paired-avx2`, which batches four H16 transforms and executes
  their paired P/Q H8 branches in handwritten AVX2 assembly.

Every algorithm uses `float` samples, coefficients, twiddles, and scale
factors. All compute the conventional unnormalised transform

```text
X[k] = sum_j x[j] exp(-2*pi*i*j*k/N).
```

The inverse API computes

```text
x[j] = (1/N) sum_k X[k] exp(+2*pi*i*j*k/N).
```

## Reproducible build and benchmark

```sh
make
make test
make bench
make ffmpeg-cycles
make tangent-cycles
make lane2-cycles
make lane8-profile
make aarch64-asm-check
make aarch64-qemu-test
make ffmpeg-aarch64-qemu-test
make aarch64-instruction-counts
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
./fft_harness --bench --inverse --min-power 4 --max-power 13
./fft_harness --bench --min-power 4 --max-power 22 --csv benchmark.csv
./fft_harness --help
```

Plan allocation and coefficient generation are excluded from execution
timings. Both FFmpeg rows include the copies into and out of their aligned
AVTX buffers, matching this project's in-place public API. `ffmpeg-avtx`
uses FFmpeg's native CPU dispatch; `ffmpeg-sse` creates the AVTX plan with
AVX disabled and the x86 feature mask capped at SSE4.2, then restores the
previous process mask. The harness prints all results to the console and can
also write CSV.

Each x86 lane2, lane4, and tangent-SSE entry is runtime-gated by CPUID.
`tangent-x86-asm` is enabled when the host is x86-64 with AVX2 and FMA.
Portable C algorithms remain available on other targets.

## Lane-factorized implementations

For `N=2M`, `lane2-sse` computes the even and odd length-`M` residue
transforms together in one XMM register. It uses a mixed-radix permutation,
register-resident FFT4/FFT8 leaves, block-major radix-4 upper stages, and a
fused two-row transpose/twiddle/FFT2 finish. It is genuine baseline SSE
assembly and contains no AVX/VEX instructions. Its implementation and
measurements are described in
[`docs/lane2-sse.md`](docs/lane2-sse.md).

The AArch64 `lane2-neon` port uses the same factorization with NEON `REV64`,
`FMUL`, and `FMLA` products. It carries its selective sign mask across all
private stages and has a freestanding QEMU correctness target. The code-level
comparison with FFmpeg NEON and LLVM-MCA estimates are in
[`docs/lane2-neon.md`](docs/lane2-neon.md).

The fused lane4-SoA layout, Stockham destination writes, and the NEON/SSE/AVX
16/32/64 leaf families are described in
[`docs/fused-lane4-soa.md`](docs/fused-lane4-soa.md).

For `N=4M`, every lane4 implementation computes the four length-`M`
transforms from `x[4m+r]` in parallel. The plain-C implementation makes the
decomposition explicit. The SSE family uses separate four-float real and
imaginary vectors; its FFT4/FFT8 leaves, radix-4 stages, and fused finish are
FFmpeg-style x86inc/NASM assembly. The 256-bit family treats each YMM register
as four interleaved complex lanes.

The tuned `lane4-avx2-fma` kernel
computes the four length-`M` transforms from `x[4m+r]` simultaneously. Its
mixed-radix input permutation is fused into FFT4/FFT8 leaves, upper stages are
block-major radix-4, and the final twiddles feed a fused 4x4 transpose plus
vector FFT4. Complete 16- and 32-point transforms stay in registers.
Scalar C is restricted to allocation, permutation, and float coefficient
generation; all AVX execution is handwritten NASM. The first-party tree uses
no compiler SIMD or timestamp intrinsics.

This schedule prioritizes regular full-width SIMD and memory locality over
minimum scalar operation count. On this host it beats tangent assembly and
FFmpeg at every tested power of two from 16 through 8192. The derivation,
prototype history, cycle results, limitations, and research references are in
[`docs/lane-factorized-fft.md`](docs/lane-factorized-fft.md).
The per-ISA organization, compiler boundaries, and comparative measurements
are in [`docs/lane4-isa-variants.md`](docs/lane4-isa-variants.md).
A detailed derivation, execution walkthrough, and literature lineage are in
[`description.md`](description.md).
The separate lane8 AVX and size-adaptive SSE top-level designs, their local
machine-operation bounds, assembly layout, and measured limits are in
[`docs/hardware-top-level-fft.md`](docs/hardware-top-level-fft.md).
Measured top-level candidates that were not retained are recorded in
[`docs/rejected-top-level-experiments.md`](docs/rejected-top-level-experiments.md).
The mathematical review, exact H16 integration, validation, and complete
16-through-8192 benchmark are in
[`investigation/scaled_h16_review.md`](investigation/scaled_h16_review.md).

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
- fixed FFT128 and FFT256 trees remove level/family dispatch and offset loops;
- execution-ordered normal factors are preduplicated into real and imaginary
  streams, replacing two hot-loop shuffles with one load;
- scheduled transforms keep AVX state live across assembly batches and issue
  one `vzeroupper` at the C boundary;
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
  --target-ms 100 --csv benchmark.csv
```

Median execution times in microseconds:

| N | radix-2 | tangent C | tangent SSE3 | tangent AVX2/FMA | lane2 SSE | lane4 AVX2/FMA | FFmpeg native | FFmpeg SSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.140 | 0.110 | 0.070 | 0.040 | 0.040 | 0.040 | 0.060 | 0.080 |
| 32 | 0.310 | 0.220 | 0.120 | 0.060 | 0.060 | 0.050 | 0.070 | 0.150 |
| 64 | 0.670 | 0.490 | 0.220 | 0.100 | 0.080 | 0.060 | 0.100 | 0.290 |
| 128 | 1.250 | 0.800 | 0.360 | 0.170 | 0.160 | 0.090 | 0.160 | 0.610 |
| 256 | 2.760 | 1.760 | 0.760 | 0.340 | 0.330 | 0.180 | 0.330 | 1.320 |
| 512 | 6.060 | 3.700 | 1.560 | 0.730 | 0.760 | 0.370 | 0.690 | 2.870 |
| 1024 | 13.190 | 7.920 | 3.260 | 1.420 | 1.580 | 0.780 | 1.480 | 6.310 |
| 2048 | 28.520 | 16.880 | 6.930 | 3.010 | 3.660 | 1.790 | 3.210 | 13.850 |
| 4096 | 61.950 | 36.380 | 15.330 | 6.940 | 7.800 | 4.160 | 7.650 | 30.530 |
| 8192 | 136.866 | 82.650 | 35.660 | 18.450 | 20.030 | 9.830 | 21.161 | 71.210 |

The exhaustive 26-implementation by 10-size table is in
[`docs/benchmark-results.md`](docs/benchmark-results.md); `benchmark.csv`
contains the same complete run plus sample counts, minima, arithmetic counts,
speedups, and checksums.

The wall-clock timer is quantized at 0.01 µs for the smallest transforms.
The longer-batch `make tangent-cycles` run is more discriminating: tangent
AVX is faster than FFmpeg at 16, 64, and 1024–8192; tied within 0.2% at
128/256; 7.3% slower at 32; and 5.7% slower at 512 on this run.

The vendored FFmpeg FMA3 split-radix leaves are also patched with exact
sign-folding reductions. A five-pair cycle A/B against the unmodified pinned
source measured median gains of 0.17–0.96% from 64 through 8192 on this host.
`make ffmpeg-cycles` runs the longer-batch FFmpeg-only cycle harness.
`make tangent-cycles` reports the tangent AVX and representative SSE paths
beside FFmpeg through the same public in-place API.
`make lane2-cycles` compares lane2-SSE, lane4-SSE, and FFmpeg with serialized
TSC measurements.

The investigation of FFmpeg's x86 FFT optimization TODOs, including retained
and rejected assembly experiments, is in
[`docs/ffmpeg-todo-investigation.md`](docs/ffmpeg-todo-investigation.md).
The follow-up tangent work is documented in
[`docs/tangent-assembly-optimization.md`](docs/tangent-assembly-optimization.md);
unsuccessful experiments are collected separately in
[`docs/rejected-tangent-experiments.md`](docs/rejected-tangent-experiments.md).

Results vary with CPU, frequency policy, compiler, and system load; regenerate
`benchmark.csv` on the target machine rather than treating these values as
portable constants.

## Validation and FFmpeg's own harness

`make test` compares every forward and inverse path, for three input families
and every power of two through 512, against a direct DFT accumulated in
`long double`. It then cross-checks both directions through `2^22`.

FFmpeg does have an FFT test/benchmark harness:
`third_party/ffmpeg/tests/checkasm/av_tx.c`. It checks AVTX implementations
against the C reference and uses checkasm's `bench_new` timing facility. This
project uses its own common harness so radix-2, split-radix, tangent, and
FFmpeg are measured through the same API and reporting path.

The tangent implementation follows Daniel J. Bernstein, *The tangent FFT*
(2007), and the equivalent scaled conjugate-pair presentation by Steven G.
Johnson and Matteo Frigo, *A Modified Split-Radix FFT With Fewer Arithmetic
Operations* (2007).
