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
same inner frequency. The stage loop is:

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

After the four lane-dependent twiddle vectors have been applied, four
consecutive `q` rows are transposed into four `r` columns. One ordinary vector
radix-4 butterfly then produces four vectors containing consecutive `q`
outputs for `p=0,1,2,3`. Those vectors can be stored directly in natural
output order.

This fusion was the largest single improvement. It changes four cross-lane
FFT4 networks plus an output transpose into one transpose plus one full-width
FFT4. It also illustrates the main design rule: transpose when it turns
shuffle-bound horizontal arithmetic into independent vertical SIMD
arithmetic.

### 4. Keep complete tiny transforms in registers

The 16- and 32-point paths retain the whole decomposition in the AVX2
register file. The 32-point path holds the eight rows from its vector FFT8 and
feeds them directly to two fused finish blocks. It does not materialize the
intermediate work array.

## Prototype progression

The retained experiment in `analysis/lane4_experiment.c` records the useful
failures as well as the successful path:

1. lane-factorized radix-2 proved the layout and DFT identity, but its
   cache-unfriendly full-array passes became very slow at 8192;
2. block-major radix-4 reduced stage traffic and loop overhead;
3. fusing digit permutation with FFT4/FFT8 leaves removed one complete pass;
4. register-resident 16/32 leaves removed small-transform scratch traffic;
5. transpose-before-butterfly fusion removed most of the finishing shuffles.

This progression is important: merely putting a conventional iterative
radix-2 FFT in YMM registers was not sufficient. The memory traversal and the
position of the transpose mattered more than the arithmetic count.

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
| 16 | 60.5 | 136.6 | 127.8 | 2.26x | 2.11x |
| 32 | 74.4 | 115.4 | 131.4 | 1.55x | 1.77x |
| 64 | 177.5 | 235.9 | 299.3 | 1.33x | 1.69x |
| 128 | 332.2 | 754.0 | 574.8 | 2.27x | 1.73x |
| 256 | 795.2 | 1427.1 | 1213.5 | 1.79x | 1.53x |
| 512 | 1634.1 | 2704.0 | 2549.7 | 1.65x | 1.56x |
| 1024 | 3853.6 | 5400.3 | 5442.2 | 1.40x | 1.41x |
| 2048 | 8182.2 | 11446.2 | 12271.0 | 1.40x | 1.50x |
| 4096 | 19764.9 | 26650.3 | 29091.4 | 1.35x | 1.47x |
| 8192 | 43130.6 | 70592.1 | 81238.1 | 1.64x | 1.88x |

The common wall-clock harness separately copies randomized input before each
timed execution. It also shows a win at every listed size. The exact numbers
vary with CPU frequency and system load; the cycle ratios above are more
stable than sub-microsecond wall-clock readings.

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
8N bytes  packed final twiddle vectors
2N bytes  inner roots
 N bytes  mixed-radix permutation
```

or about `19N` bytes plus allocator and plan overhead. Plan construction is
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
