# Lane-factorized FFT for AVX2

## Outcome

The resulting algorithm is `lane4-radix4`, a float-only, forward complex FFT
for power-of-two sizes. On the test host it is faster than this repository's
split-radix C, tangent C, tangent AVX2 assembly, and patched FFmpeg AVTX paths
at every measured size from 16 through 8192. It remains faster through its
current implementation limit of 131072.

This is not a new asymptotic FFT or a claim of a new arithmetic identity. It
is a Cooley--Tukey/four-step decomposition selected and scheduled around the
four complex-float lanes in an AVX2 register. Its advantage comes from data
layout, full-width independent operations, fixed register codelets, regular
loops, and a fused transpose/butterfly finish rather than a lower scalar
operation count.

## Derivation

Let

```text
N = 4 M
n = 4 m + r,       0 <= r < 4
k = q + M p,       0 <= p < 4
W_L = exp(-2 pi i / L).
```

Substitution in the DFT gives

```text
X[q + M p]
  = sum_r W_N^(r q) W_4^(r p)
      sum_m x[4 m + r] W_M^(m q).
```

Define four length-`M` transforms

```text
F_r[q] = sum_m x[4 m + r] W_M^(m q).
```

Then the computation is:

1. compute the four `F_r` transforms in parallel;
2. form `A[q,r] = F_r[q] W_N^(r q)`;
3. for every `q`, compute the length-4 DFT of `A[q,:]`;
4. place its result at `X[q + M p]`.

The useful layout follows directly from the original interleaved complex
array. One unaligned 256-bit load fetches

```text
[x[4m+0], x[4m+1], x[4m+2], x[4m+3]]
```

where each item is one interleaved complex float. Consequently, the four
independent `M`-point transforms are one vector-valued transform. Every add,
subtract, and inner-transform twiddle multiplication uses the entire YMM
register. There is no initial AoS-to-SoA conversion.

## Schedule

### 1. Fuse permutation into register leaves

The inner transform uses iterative DIT radix-4 stages. Its mixed-radix digit
permutation is computed once in the plan. Execution gathers each 32-byte input
group in the order needed by either:

- a register-resident FFT4 when `log2(M)` is even; or
- a fused radix-2/radix-4 FFT8 when `log2(M)` is odd.

The leaf writes natural-order vector outputs to the work array. This removes
a separate permutation copy and a separate first-stage work-array pass.

### 2. Use block-major, full-width radix-4 stages

All upper inner stages are ordinary radix-4 butterflies over four YMM
registers. Each scalar twiddle is broadcast because all four lanes are at the
same inner frequency. Twiddle triplets are stored stage-major in exact
execution order, so the hot loop consumes a linear 24-byte stream instead of
deriving the `k`, `2k`, and `3k` root addresses. Two independent butterflies
are software-unrolled together. The stage loop is:

```text
for each contiguous radix-4 block
    for each frequency within the block
        load four vectors
        apply three twiddles
        radix-4 butterfly
        store four vectors
```

There is no recursive call tree, transform-family dispatch, per-node copy, or
per-butterfly transform-kind branch.

### 3. Fuse the final transpose and FFT4

The first prototype performed four horizontal, shuffle-heavy FFT4s and then
transposed the four result rows. The faster identity is:

```text
four row FFT4s, then transpose
    ==
transpose the input rows, then one vector FFT4 over the columns.
```

Four consecutive `q` rows are first transposed into four `r` columns.
Column zero has no twiddle. Columns one through three are multiplied by
prepacked factors and one ordinary vector radix-4 butterfly then produces
four vectors containing consecutive `q` outputs for `p=0,1,2,3`. Those
vectors can be stored directly in natural output order.

The factor table stores already-duplicated real and imaginary vectors.
Execution therefore uses two loads instead of loading an interleaved factor
and issuing two duplication shuffles. This trades plan memory for fewer
shuffle-port operations in the important cache-resident size range.

This fusion was the largest single improvement. It changes four cross-lane
FFT4 networks plus an output transpose into one transpose plus one full-width
FFT4. It also illustrates the main design rule: transpose when it turns
shuffle-bound horizontal arithmetic into independent vertical SIMD
arithmetic.

### 4. Keep complete tiny transforms in registers

The 16- and 32-point paths retain the whole decomposition in the AVX2
register file. The 32-point path holds the eight rows from its vector FFT8 and
feeds them directly to two fused finish blocks.

For larger even-log inner transforms, four FFT4 children and their first
radix-4 parent form one straight-line vector FFT16 leaf. GCC carries most rows
in registers and emits three short-lived stack spills instead of a full
16-row work-array store/reload round trip. Odd-log transforms store three FFT8
children, retain the fourth child's eight rows, and consume those rows
directly in the FFT32 parent.

The fixed 64- and 128-point paths carry completed child rows further into the
final transpose/butterfly. Hot entry points are explicitly aligned so adding
fixed codelets cannot accidentally move the general loop onto a poor
instruction-fetch boundary.

## Prototype progression

The retained experiment in `analysis/lane4_experiment.c` records the useful
failures as well as the successful path:

1. lane-factorized radix-2 proved the layout and DFT identity, but its
   cache-unfriendly full-array passes became very slow at 8192;
2. block-major radix-4 reduced stage traffic and loop overhead;
3. fusing digit permutation with FFT4/FFT8 leaves removed one complete pass;
4. register-resident 16/32 leaves removed small-transform scratch traffic;
5. transpose-before-butterfly fusion removed most of the finishing shuffles;
6. column-major factors removed the always-trivial `r=0` multiply;
7. stage-major twiddles and two-way unrolling removed upper-loop address and
   control overhead;
8. FFT16/FFT32 parent fusion carried register rows across the first work
   boundary;
9. fixed 64/128 paths carried rows into the final transform.

This progression is important: merely putting a conventional iterative
radix-2 FFT in YMM registers was not sufficient. The memory traversal and the
position of the transpose mattered more than the arithmetic count.

Several plausible variants were measured and rejected:

- two radix-8 stages replacing three radix-4 stages saved one work-array pass
  but lost 2--8% at 1024--8192 because its denser codelet and seven external
  twiddles dominated while data remained cache-resident;
- branching out exact `W8`, `-i`, `W8^3` stage frequencies saved arithmetic
  but lost 3--8% by disrupting the uniform two-way loop;
- four-way stage unrolling, two-way finish unrolling, and `restrict` hints
  all regressed;
- compressing each leaf's permutation to one base offset reduced metadata
  but merely exchanged table loads for address-generation instructions;
- retaining one row from every FFT8 child caused code-size/register-pressure
  regressions; retaining all eight rows from only the final child won.

## Cycle results

Command:

```sh
make lane4-experiment
```

The harness pins execution to CPU 2, warms each routine, measures 31
serialized-TSC batches, and reports the median cycles per transform. These
results were measured on an AMD Ryzen 9 3900X with GCC 15.2.0:

| N | lane4-radix4 | tangent-x86-asm | FFmpeg AVTX | vs tangent | vs FFmpeg |
|---:|---:|---:|---:|---:|---:|
| 16 | 63.5 | 135.9 | 125.2 | 2.14x | 1.97x |
| 32 | 88.3 | 143.7 | 161.7 | 1.63x | 1.83x |
| 64 | 122.6 | 232.9 | 294.1 | 1.90x | 2.40x |
| 128 | 262.3 | 754.7 | 570.2 | 2.88x | 2.17x |
| 256 | 582.5 | 1432.1 | 1205.7 | 2.46x | 2.07x |
| 512 | 1313.1 | 2710.0 | 2540.8 | 2.06x | 1.94x |
| 1024 | 2861.8 | 5429.0 | 5473.0 | 1.90x | 1.91x |
| 2048 | 6611.9 | 11568.5 | 12337.1 | 1.75x | 1.87x |
| 4096 | 14854.3 | 26430.3 | 28873.9 | 1.78x | 1.94x |
| 8192 | 35629.2 | 71335.8 | 81392.7 | 2.00x | 2.28x |

Relative to the preceding committed lane4 implementation, the retained
changes reduce cycle medians by roughly 17--31% from 64 through 8192. The
common wall-clock harness separately copies randomized input before each timed
execution and also shows a win at every listed size. The exact numbers vary
with CPU frequency and system load; ratios are more stable than
sub-microsecond wall-clock readings.

## Correctness and numerical behavior

`make test` performs three input families at every power of two through 512
against a direct DFT accumulated in `long double`, then randomized
cross-checks through `2^22`. The production lane path is currently enabled
through 131072 and therefore participates in those cross-checks up to that
size.

The worst relative maximum error observed for `lane4-radix4` is
`3.238e-07`. Transform samples, roots, finish factors, and all arithmetic in
the algorithm are `float`; wider types are used only by the test reference
and timers.

An AddressSanitizer/UndefinedBehaviorSanitizer build also passes the complete
suite:

```sh
make debug
ASAN_OPTIONS=detect_leaks=0 ./fft_harness --test
```

Leak detection is disabled in that command only because LeakSanitizer cannot
run under the surrounding ptrace environment.

## Memory and implementation limits

For `N=4M`, the lane plan currently allocates approximately:

```text
8N bytes  vector work array
12N bytes split real/imag final twiddle vectors
~2N bytes stage-major inner twiddle triplets
 N bytes  mixed-radix permutation
```

or about `23N` bytes plus allocator and plan overhead. Plan construction is
outside benchmark timing. Execution is in-place at the API boundary but uses
the `8N`-byte work array.

The implementation requires x86-64 AVX2 and FMA and is currently exposed for
`16 <= N <= 131072`. Portable algorithms remain available elsewhere. It
supports only power-of-two complex forward transforms and natural-order
output. There is no inverse or real-input specialization yet.

At substantially larger sizes, cache blocking or a recursive six-step
version of the inner vector transform may reduce work-array traffic. On
AVX-512, the natural generalization is eight complex lanes (`N=8M`) with an
FFT8 finish, but the transpose/butterfly fusion must be regenerated for the
wider register file rather than mechanically widening this code.

## Influences and distinctions

The design search used these established ideas:

- FFTW's generated register codelets and separation of planning from
  execution;
- PFFFT's treatment of one SIMD vector as several parallel scalar FFTs;
- four/six-step FFT transpositions and cache-conscious traversal;
- FFTS and FFmpeg's fixed small kernels and precomputed execution schedules;
- the earlier tangent work in this repository, especially the lesson that
  low arithmetic count does not compensate for irregular dataflow.

The final decomposition and C intrinsics implementation were written for this
repository; source was not copied from PFFFT or FFTS.

Primary references:

- D. J. Bernstein, [The tangent FFT](https://cr.yp.to/arith/tangentfft-20070809.pdf)
- M. Frigo and S. G. Johnson,
  [The Design and Implementation of FFTW3](https://fftw.org/fftw-paper-ieee.pdf)
- M. Frigo et al.,
  [Cache-Oblivious Algorithms](https://www.fftw.org/~athena/abstracts/abstract12.html)
- [PFFFT source repository](https://github.com/marton78/pffft)
- [FFTS source repository](https://github.com/anthonix/ffts)
