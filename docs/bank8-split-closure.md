# Bank8 split-closure FFT

## Result

This investigation produced a new SIMD execution schedule, implemented as
`bank8-avx2-fma`, and a conservative deployment policy,
`banked-avx2-auto`.

The raw algorithm is a conventional `N=8M` Cooley--Tukey factorization with
an implementation-specific combination that was not present elsewhere in
this repository:

- two native four-complex AoS YMM banks are advanced together;
- every inner twiddle load is shared by both banks;
- the input needs no AoS-to-SoA conversion;
- both the digit-reversed leaves and the natural-input outer transform use a
  34-instruction split-radix FFT8 closure;
- the recursion, permutation, and coefficient generation are completely
  outside the timed assembly path.

On the Ryzen 9 3900X, raw bank8 is 10.1% faster than lane4 AVX2/FMA at
`N=64`, equal at 128, and 3.7% faster at 8192. It loses by 1.2--3.1% at
256--4096 and by 10.6% at 32. On the loaded Skylake i7-6500U, median paired
runs favor bank8 from 32 through 1024 and lane4 from 2048 upward. Bank8 beats
native FFmpeg AVTX at every measured size on both machines.

This is a new schedule and data layout in this project. It is not a new DFT
identity or a new arithmetic-complexity bound. A literature search did not
find this exact dual-AoS-bank/shared-root/split-closure combination, but that
is not evidence sufficient for a claim of worldwide algorithmic novelty.

## Factorization

Let

```text
N = 8M
n = 8m + r,       0 <= r < 8
k = q + Mp,       0 <= p < 8
W_L = exp(-2*pi*i/L).
```

Define eight length-`M` transforms:

```text
F_r[q] = sum(m=0..M-1) x[8m+r] W_M^(mq).
```

Then

```text
X[q+Mp] = sum(r=0..7) F_r[q] W_N^(rq) W_8^(rp).
```

The arithmetic is therefore eight inner FFTs, seven outer twiddle products
per `q`, and one FFT8. The implementation is differentiated by how those
operations are grouped into machine registers and memory passes.

## Dual-bank layout

One 64-byte work row is:

```text
bank A ymm = [F0.re,F0.im, F1.re,F1.im, F2.re,F2.im, F3.re,F3.im]
bank B ymm = [F4.re,F4.im, F5.re,F5.im, F6.re,F6.im, F7.re,F7.im]
```

The public complex AoS input already has exactly this layout. A leaf reads
two adjacent YMM vectors, so the six input-conversion instructions paid by
the older split-complex lane8 implementation disappear.

The two banks represent independent transforms at the same inner frequency.
Consequently they use identical radix-4 roots. The upper stage loads a
replicated real and imaginary coefficient once and applies it to the
corresponding value in both banks:

```text
load wr, wi
rotate bankA value by (wr,wi)
rotate bankB value by (wr,wi)
```

The work array still occupies exactly `8N` bytes, the same as lane4:

```text
lane4: (N/4) rows * 32 bytes
bank8: (N/8) rows * 64 bytes.
```

Bank8 halves the row count, permutation entries, loop positions, and root
records for comparable stages. It does not reduce the sample traffic of a
complete radix-4 pass.

## Split-radix FFT8 closure

The first prototype used a direct radix-2 FFT8 requiring about 46 vector
instructions. The retained natural-input codelet uses one FFT4 and two
FFT2s. For inputs `x0` through `x7`, define:

```text
E = FFT4(x0,x2,x4,x6)
A0 = x1+x5       A1 = x1-x5
B0 = x3+x7       B1 = x3-x7

T0 = A0+B0
T2 = -i(A0-B0)
C  = A1+B1
D  = A1-B1
T1 = sqrt(1/2) * ( D-iC)
T3 = sqrt(1/2) * (-D-iC)

Xk   = Ek+Tk
Xk+4 = Ek-Tk,       0 <= k < 4.
```

Lifted over four independent frequencies in YMM lanes, this takes 34 vector
instructions. It is used in two places:

1. The outer closure receives residues in natural order directly.
2. An FFT8 leaf receives mixed-radix digit-reversed inputs in
   `[0,4,1,5,2,6,3,7]` order. Merely changing the eight load positions to
   `[0,1,2,3,4,5,6,7]` lets the same natural-input split-radix codelet replace
   the old leaf without changing the permutation table.

Replacing both closures was important. On Zen 2, the leaf change reduced the
observed bank8 cost around 3% at 256, 4% at 1024, and 2% at 4096.

## Finish

Four adjacent work rows are loaded as eight YMM registers. Each bank is
transposed independently, producing eight residue columns with four
frequencies per vector. Columns 1 through 7 receive their prepacked
`W_N^(rq)` factors. The split-radix FFT8 then produces eight natural output
blocks, each stored as four contiguous complex values.

The finish has more arithmetic than two lane4 FFT4 closures: it has seven
outer products rather than six and a larger vertical codelet. This explains
why bank8 is not unconditionally faster even though its leaves, loop count,
and root consumption are smaller.

## Planning and execution

The scalar C planner creates:

- a mixed-radix input permutation;
- stage-major replicated float roots;
- four-frequency outer-factor packets;
- one 64-byte-aligned dual-bank work array.

All transform samples, coefficients, tables, and SIMD arithmetic are
single-precision `float`. Sizes, pointers, counters, and timing data use the
integer or timer types required by the platform.

The execution path is handwritten NASM. It has no recursion, allocation,
coefficient generation, transform-kind branch, or intrinsic. The public
entry performs:

```text
paired FFT4 or split-radix FFT8 leaves
zero or more paired radix-4 stages
dual transpose + seven roots + split-radix FFT8 finish
```

The current assembly object's text is 2,366 bytes, compared with 3,731 bytes
for `lane4_avx_stage.o` and 6,215 bytes for the older split-complex
`lane8_avx_stage.o`. Object size includes constants and alignment padding and
is not a source-line comparison.

## Correctness

The complete repository harness passes:

- forward and normalized inverse comparison with a long-double direct DFT
  through 512;
- cross-checks through `2^22`;
- every power of two exposed by the project.

Worst normalized bank8 error in the recorded run was:

```text
forward  2.760e-7
inverse  9.473e-8
```

## Ryzen 9 3900X cycles

Each table entry is the median paired ratio from three independent runs.
Every run used 41 samples, alternated measurement order, included the public
in-place API, and was pinned to CPU 2. A ratio below one favors bank8.
Absolute cycles are the median of the three per-run candidate medians; they
are included as a scale indicator, while the paired ratio is the more robust
comparison.

| N | bank8 cycles | bank8 / lane4 AVX2-FMA | bank8 / FFmpeg |
|---:|---:|---:|---:|
| 32 | 73.02 | 1.1063 | 0.5592 |
| 64 | 128.03 | 0.8989 | 0.4273 |
| 128 | 279.39 | 0.9992 | 0.4270 |
| 256 | 609.83 | 1.0239 | 0.5007 |
| 512 | 1,355.07 | 1.0213 | 0.5292 |
| 1024 | 2,973.43 | 1.0120 | 0.5470 |
| 2048 | 6,760.73 | 1.0186 | 0.5507 |
| 4096 | 16,535.01 | 1.0312 | 0.5752 |
| 8192 | 36,290.89 | 0.9627 | 0.4500 |

## Intel i7-6500U cycles

The laptop was simultaneously doing unrelated work. Results therefore use
the median paired ratio from three complete runs, each containing the same
41 alternating samples as the Zen test. Absolute cycles are intentionally
not used to infer cross-machine speed.

| N | bank8 / lane4 AVX2-FMA | bank8 / FFmpeg |
|---:|---:|---:|
| 32 | 0.9701 | 0.5383 |
| 64 | 0.8244 | 0.4422 |
| 128 | 0.8514 | 0.5360 |
| 256 | 0.9480 | 0.5381 |
| 512 | 0.9513 | 0.5826 |
| 1024 | 0.9424 | 0.5786 |
| 2048 | 1.0866 | 0.7933 |
| 4096 | 1.1546 | 0.8219 |
| 8192 | 1.1546 | 0.8097 |

The sharp 1024/2048 reversal is why the deployment entry does not assume one
global bank width.

## Adaptive policy

`banked-avx2-auto` keeps selection outside the hot assembly:

```text
Skylake:  bank8 for 32..1024, lane4 otherwise
Zen 2:    bank8 for 64, 128, and 8192, lane4 otherwise
other:    bank8 for 64 and 128, lane4 otherwise
```

These are deliberately narrow policies based on the two measured cores.
They are not claims about all processors from either vendor. Raw
`bank8-avx2-fma` remains available for retesting and future dispatch updates.

## Rejected variants

### Split-complex lane8

The older `lane8-avx2-fma` reduces the direct complex-product instruction
count but pays an AoS-to-SoA conversion at every input row. Bank8 is
consistently about 15--46% faster because its two banks already match the
public layout.

### Direct radix-2 FFT8

The initial 46-instruction closure was correct but left bank8 slower than
lane4 at almost every size. The 34-instruction split-radix closure is
retained.

### Register-resident N=32

A fully fused N=32 leaf removed the scratch store/reload but was about 6%
slower than the generic bank path. The longer dependency graph outweighed
the 512 bytes of avoided work traffic, so the leaf and its dead code were
removed.

### Compact 24-byte roots

Replacing each 192-byte replicated root record with six scalar floats made
the table eight times smaller and required the same number of explicit load
instructions via `vbroadcastss`. On pinned Zen 2 it regressed the 8192-point
transform by about 7--9%. The broadcast/permutation pressure outweighed the
cache-footprint reduction.

### Repeated memory-source roots

Using six memory-source multiply/FMA instructions instead of two explicit
loads plus six register-source arithmetic instructions saved six decoded
instructions per paired butterfly, but read every coefficient twice. It
regressed the important 1024 and 4096 comparisons by roughly 3--6 percentage
points and was reverted.

### Independent scratch names

Assigning different architectural scratch registers to the three rotations
did not give a repeatable gain. Modern register renaming already removes the
false write-after-read and write-after-write name dependencies.

## Static throughput audit

`analysis/bank8_mca.s` isolates shared roots, repeated memory-source roots,
and compact broadcasts. LLVM-MCA models all three at a two-cycle block
throughput on both Zen 2 and Skylake. The shared form has a shorter modeled
loaded-coefficient path than repeated memory operands. The model does not
predict the end-to-end compact-root regression, which depends on surrounding
shuffle pressure and cache behavior; this is why measured full transforms
decide retention.

## Reproduction

```sh
make
./fft_harness --test
make bank8-cycles
make bank8-mca
```

`analysis/bank8_cycles` reports all powers of two from 32 through 8192 and
compares bank8 with lane4 AVX, lane4 AVX2/FMA, the older lane8, and native
FFmpeg.

## Credit and relation to prior work

The algebraic foundation is Cooley--Tukey. The FFT8 closure is split radix.
Daniel J. Bernstein's *The tangent FFT* motivated the project's continued
separation of arithmetic count from memory-access cost:
<https://cr.yp.to/arith/tangentfft-20070809.pdf>.

FFTW's codelet and planner work is the established precedent for scheduling
small DFT DAGs for registers and selecting different decompositions by
machine:

- Matteo Frigo, *A Fast Fourier Transform Compiler*,
  <https://fftw.org/pub/fftw/pldi99.pdf>.
- Matteo Frigo and Steven G. Johnson,
  *The Design and Implementation of FFTW3*,
  <https://fftw.org/fftw-paper-ieee.pdf>.

FFmpeg's `libavutil/tx` x86 assembly supplied the practical coding style,
instruction-selection baseline, and comparison target. This implementation
does not copy an FFmpeg transform routine; it uses the vendored FFmpeg build
through the repository's common in-place API for measurement.
