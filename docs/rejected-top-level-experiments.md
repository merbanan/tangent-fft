# Top-level FFT experiments that did not win

This report is intentionally separate from the retained implementation. It
records correct ideas that lost on the Ryzen 9 3900X so the same schedules do
not need to be rediscovered and retested.

## Vertical split-radix across SIMD lanes

The first candidate used four AVX residue transforms or two SSE residue
transforms as the leaves of a vertical conjugate-pair split-radix tree. It
reduced the scalar arithmetic count and preserved independent SIMD lanes.
The implementation was correct, but its irregular quarter transforms,
level-group scheduling, and additional work-array traffic were much more
expensive than regular radix-4 lane execution.

Median cycles:

| N | AVX vertical split | SSE vertical split | lane4 AVX | lane4 SSE | FFmpeg |
|---:|---:|---:|---:|---:|---:|
| 16 | 95 | 120 | 64 | 83 | 125 |
| 64 | 294 | 539 | 159 | 299 | 385 |
| 256 | 1,372 | 2,586 | 589 | 1,156 | 1,213 |
| 1024 | 5,326 | 9,838 | 2,855 | 5,550 | 5,437 |
| 4096 | 26,041 | 47,522 | 15,126 | 26,960 | 28,791 |
| 8192 | 57,342 | 107,871 | 36,627 | 60,807 | 81,924 |

At 8192 the lower-flop AVX schedule was 56% slower than lane4. Disassembly
showed inlined vector arithmetic and FMA generation; the loss was structural,
not a missed compiler instruction.

## Regular upper radix-8 on interleaved AVX rows

A mixed-radix prototype consumed three FFT levels per pass with a
register-resident FFT8. It was meant to reduce full work-array passes and
move the leading arithmetic coefficient closer to split radix. It sometimes
beat the otherwise identical radix-4 prototype, but not consistently:

| N | radix-4 prototype cycles | radix-8 prototype cycles |
|---:|---:|---:|
| 256 | 1,088 | 1,033 |
| 512 | 2,256 | 2,480 |
| 1024 | 5,128 | 4,967 |
| 2048 | 10,749 | 10,415 |
| 4096 | 24,671 | 25,339 |
| 8192 | 53,531 | 52,711 |

The gains on favorable `log2(N) mod 3` sizes were too small to offset the
losses on the other sizes. Seven general twiddle streams and the FFT8
register schedule also left less freedom than the four-row radix-4 kernel.
No intrinsic version of this experiment is retained; the production research
continued directly in NASM.

### Full-assembly retry with compact roots

A second implementation made the radix-8 pass entirely FFmpeg-style NASM
and combined it with the compact-root representation described below. It was
correct for forward and inverse transforms through the full cross-check
range, but the result was worse than the earlier prototype:

| N | production lane4 cycles | assembly radix-8 cycles | penalty |
|---:|---:|---:|---:|
| 128 | 267 | 336 | 26% |
| 256 | 591 | 701 | 19% |
| 512 | 1,324 | 1,562 | 18% |
| 1024 | 2,856 | 3,755 | 31% |
| 2048 | 6,650 | 8,068 | 21% |
| 4096 | 15,109 | 18,822 | 25% |
| 8192 | 35,886 | 45,316 | 26% |

The pass does remove a work-array traversal when the exponent admits the
factorization. However, seven twiddle chains must coexist with the
register-resident FFT8. On Zen 2 that lengthens the critical path and raises
shuffle/load-port pressure more than the eliminated traversal saves.

## Lower-bound AoS rotation with expanded coefficients

The most promising AVX retry preserved the production lane4 interleaved
layout and changed each common coefficient to replicated vectors. A generic
rotation then reached the direct AoS/FMA instruction bound:

```text
vpermilps       swapped, value, 0xb1
vmulps          swapped, signed_im
vfmaddps        value, value, real, swapped
```

This is three vector arithmetic instructions and no representation
conversion, versus two broadcasts plus three arithmetic instructions in the
production compiler output. The complete execution path, including FFT4/FFT8
leaves, stages, transpose, final FFT4, and stores, was handwritten NASM.

The first 64-byte coefficient record tied near 512 but lost at larger sizes
because the last-stage table no longer fit in L1. A compact 36-byte record
kept one signed-imaginary vector and one scalar real component, requiring one
broadcast and four instructions. Three rotations were dependency-balanced
across independent YMM registers, and the first FFT4 parent was fused with
its leaves.

Five serialized-TSC runs showed no retained crossover:

| N | production lane4 cycles | compact-root assembly cycles |
|---:|---:|---:|
| 16 | 52 | 54 |
| 32 | 72 | 76 |
| 64 | 128 | 149 |
| 128 | 267 | 285 |
| 256 | 588 | 648 |
| 512 | 1,316 | 1,354 |
| 1024 | 2,875 | 3,094 |
| 2048 | 6,649 | 6,655 |
| 4096 | 14,968 | 16,677 |
| 8192 | 35,904 | 37,480 |

The lesson is that a local instruction lower bound is insufficient when it
requires a larger coefficient stream. The production loop's compact
24-byte triplet and two-butterfly compiler scheduling win globally.

## Production-stage assembly and exact midpoint

Two attempts operated directly on the winning production representation:

1. a dependency-balanced assembly radix-4 stage consuming the compact scalar
   triplet table;
2. a C/AVX exact-root specialization for the one `k=previous/2` butterfly in
   every block (`W8`, `-i`, `W8^3`).

Both were correct. The assembly stage regressed roughly 2--4%. The midpoint
specialization saved seven vector operations at that butterfly but split the
compiler's paired hot loops; the complete transform regressed roughly
0.3--2.6%. Neither change is retained.

## Four-frequency lane8 finalizer

The first lane8 assembly finalizer used the 256-bit representation for the
inner FFT, then evaluated four final frequencies at a time in XMM registers
to make the eight output blocks contiguous. It was correct but dominated the
runtime:

```text
N=8192 base       11,149 cycles
       stages     20,034 cycles
       finish     29,049 cycles
       complete   12.48 microseconds
```

It was replaced by the retained eight-frequency YMM finalizer. The replacement
reduced the finish to about 17,515 cycles and the complete transform to
10.08--10.68 microseconds. The XMM finalizer remains only as the necessary
`N=32` fallback.

## Replacing lane4 with lane8

Lane8 reduces eight generic split-complex rotations from six interleaved AVX
instructions to four split/FMA instructions. That local result did pan out.
Replacing lane4 globally did not:

| N | lane4 AVX2/FMA us | lane8 AVX2/FMA us | lane8 penalty |
|---:|---:|---:|---:|
| 64 | 0.060 | 0.070 | 17% |
| 128 | 0.090 | 0.120 | 33% |
| 256 | 0.180 | 0.230 | 28% |
| 512 | 0.370 | 0.450 | 22% |
| 1024 | 0.790 | 0.980 | 24% |
| 2048 | 1.820 | 1.990 | 9% |
| 4096 | 4.190 | 4.830 | 15% |
| 8192 | 9.920 | 10.680 | 8% |

The public input is interleaved complex AoS. Lane4 can load four residue
classes directly into one YMM. Lane8 must split eight complex values into
real and imaginary YMM vectors with two loads, two shuffles, and two
cross-128-bit permutations for every input row. That conversion and the
larger FFT8 finish consume the inner-stage instruction savings.

Lane8 is retained as a public assembly algorithm because it is correct,
reproducible, and materially faster than FFmpeg in the important range. It is
not selected over the faster lane4 path.

## Upper radix-8 for SSE split rows

An SSE lane4 work row already occupies two XMM registers. Eight radix-8 input
rows therefore consume all sixteen architectural XMM registers before
twiddle temporaries, sums, or address-independent outputs are available.
The unavoidable spills eliminate the saved work-array pass. The
register-balanced radix-4 upper stage was retained.

## Practical conclusion

Scalar operation count alone did not predict the winner. The successful
schedule must simultaneously account for:

- input representation conversion;
- architectural register count;
- shuffle-port pressure;
- coefficient load streams;
- number of complete work-array passes;
- whether output stores are naturally contiguous.

For the tested AoS API and Zen 2 core, four interleaved complex lanes are the
best AVX top-level mapping, four split lanes are the best general SSE mapping,
and two interleaved SSE lanes remain useful only for the smallest sizes.
