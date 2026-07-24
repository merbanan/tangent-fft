# FFmpeg SIMD-history optimization audit

Date: 2026-07-24

## Scope and method

This pass inspected the history of FFmpeg's x86, AArch64, and legacy ARM FFT
implementations, translated every applicable idea to the tangent and
lane-factorized kernels, and tested each change in isolation.  The vendored
FFmpeg snapshot is `80eb9e99b93491753fc9d5bf1f8718c361131998`.  At the time
of the audit it was four upstream commits behind; none changed an FFT.

The x86 machine was an AMD Ryzen 9 3900X (Zen 2).  Cycle tests were pinned to
CPU 2, warmed before measurement, and report the median of 31 serialized-TSC
batches.  ARM changes were cross-assembled, correctness-tested with QEMU, and
modeled with LLVM-MCA 20.1.8.  QEMU and LLVM-MCA are not substitutes for
native ARM cycle measurements, so ARM changes were retained only when they
were correctness-neutral, instruction-count-neutral, and had a clear
in-order scheduling rationale.

The most useful historical FFmpeg changes were:

| Commit | Author | Lesson evaluated here |
|---|---|---|
| `0938ff9701` | Lynne | allocate independent load addresses to independent temporary registers |
| `27cffd16aa` | Lynne | use FMA selectively when it shortens the useful dependency chain |
| `82a68a8771`, `9e94c35941` | Lynne | gather profitability is CPU- and dispatch-dependent |
| `11ab1e409f` | Loren Merritt | move permutation work out of the arithmetic loop |
| `25e4f8aaee` | Zuxy Meng | unroll only when the register file can expose independent work |
| `90c17a05aa` | Lynne | replace integer multiply with addressing instructions when the result is measurable |
| `f932b89ea3` | Lynne | schedule NEON loads, signs, and complex products around the ISA's actual primitives |

FFmpeg and its contributors are credited for these structural ideas.  The
lane and tangent implementations and the experiments below are independent
code in this repository.

## Decision summary

| Experiment | Result | Decision |
|---|---|---|
| lane4 AVX FMA FFT16/FFT32 leaves | 14--17% slower at the affected small sizes | rejected |
| lane4 AVX FMA FFT8 base | neutral/noisy | rejected |
| lane4 AVX `imul` to `lea` plus shift | better static model, less than 0.5% and unstable in complete FFTs | rejected |
| lane4 SSE shared row offset | 0.3--2.0% faster from 128 through 8192 | retained |
| NEON two-address gather scheduling | large A53 model improvement, neutral on wide cores | retained |
| NEON grouped `ST1` stores | conflicting core models and FFmpeg A53 experience | rejected |
| AVX2 `vgatherdpd` permutation | 13--117% slower than scalar load/insert | rejected |
| tangent gather load interleaving | mixed by size and not repeatable | rejected |
| unused production gather routine | no callers; 64-byte text cost | removed |
| Armv8.3 `FCMLA` layout | fewer uops and half the root traffic; no native timing | analysis prototype only |

The retained production changes add no text: the lane4 SSE object remains
3,948 bytes, lane2 NEON remains 2,240 bytes, and lane4 NEON remains 2,616
bytes.  Removing the unused tangent gather entry reduces its assembly object
from 50,306 to 50,242 text bytes.

## Ideas already present or not transferable

Several history lessons were already part of the current design:

- permutation is fused into FFT4/FFT8 leaves, so there is no arithmetic-loop
  shuffle left to hoist;
- tangent FFT64 and the NEON fixed leaves already use private register
  contracts to carry child state; larger AVX parents exhaust the 16-register
  YMM file, making AVX-512's 32-register file the next plausible target;
- the fixed 16/32/64 leaves are already straight-line and the general stages
  already expose multiple independent products; further unrolling increases
  the live set and instruction-cache footprint without removing data traffic;
- AArch64 keeps the alternating-sign mask resident in `v31` and implements
  add/sub behavior with `REV64` plus `EOR`, avoiding the repeated constant
  reload that makes a literal x86 `ADDSUBPS` translation unattractive;
- FFmpeg's SVE/SME and Arm32 history contains feature-dispatch and state
  handling patterns, but no wider FFT schedule that can be transplanted into
  the NEON lane geometry;
- software prefetch remains size- and core-specific for this cache-resident
  16--8192 range and was not enabled without native cache-miss data.

## x86 experiments

### FMA leaves

FFmpeg's FMA conversion works because it folds sign and addition operations
without lengthening the limiting chain.  Applying FMA more broadly to the
lane4 FFT16 and FFT32 leaves reduced the architectural arithmetic count but
made the dependency chain more serial on Zen 2:

| N | original lane4 AVX | FMA-leaf candidate | change |
|---:|---:|---:|---:|
| 16 | about 47.4 cycles | about 53.9 cycles | 14% slower |
| 32 | about 69.2 cycles | about 81.2 cycles | 17% slower |

An FMA FFT8 base produced only measurement noise.  All three candidates were
reverted.  This is an example where the minimum instruction count is not the
minimum latency schedule.

### Integer address generation

The general-stage expression formerly using an integer multiply was
rewritten as:

```asm
lea rax, [rbx + rbx*2 - 3]
shl rax, 6
```

LLVM-MCA reduced the local modeled throughput from 1.0 to 0.8 cycles on Zen
2, 1.0 to 0.5 on Skylake, and 2.0 to 1.0 on Alder Lake.  Complete FFT A/B
runs changed by less than 0.5% and changed sign by size.  The original,
smaller expression was restored.

### One offset instead of four SSE row updates

The SSE upper stage now keeps the four row bases fixed and advances one byte
offset.  Every butterfly replaces four integer pointer additions with one
offset addition.  Repeated paired medians were:

| N | four advancing pointers | shared offset | change |
|---:|---:|---:|---:|
| 128 | 554.1 cycles | 549.0 cycles | 0.9% faster |
| 256 | 1,183.4 | 1,166.5 | 1.4% faster |
| 512 | 2,580.3 | 2,571.4 | 0.3% faster |
| 1024 | 5,462.0 | 5,432.5 | 0.5% faster |
| 2048 | 12,473.6 | 12,313.7 | 1.3% faster |
| 4096 | 26,552.8 | 26,260.2 | 1.1% faster |
| 8192 | 60,372.9 | 59,196.6 | 2.0% faster |

The code size is unchanged and the complete correctness suite passes, so this
change is retained.

### Gather

`analysis/gather_cycles.asm` compares the exact four-complex
`vgatherdpd` pattern with scalar `vmovq`/`vpinsrq` loads.  Both write the same
bit-reversal permutation and the C driver checks their outputs byte for byte.

| N | `vgatherdpd` cycles | scalar cycles | gather penalty |
|---:|---:|---:|---:|
| 16 | 41.65 | 19.64 | 112.1% |
| 64 | 141.80 | 71.15 | 99.3% |
| 256 | 546.00 | 276.66 | 97.4% |
| 1024 | 2,182.91 | 1,125.33 | 94.0% |
| 4096 | 10,510.08 | 5,767.61 | 82.2% |
| 8192 | 44,259.57 | 39,040.25 | 13.4% |

Gather is therefore not used on Zen 2.  The formerly exported
`tangent_x86_permute` implementation had no callers and was removed.  An
independent tangent experiment that merely interleaved all low-half loads
before high-half inserts was also rejected: N=128 sometimes improved, while
N=16, 64, and 512 regressed, and broader runs were neutral.

The microbenchmark is reproducible with:

```sh
make gather-cycles
```

## ARM experiments

### Two independent gather addresses

The lane2 and lane4 FFT4/FFT8 gather leaves formerly reused one integer
address register for every address-generation/load pair.  The retained
schedule alternates between two address registers, so the next address can
be generated while the previous load is issued.  It executes the same
instructions and produces the same values.

For the isolated load region, LLVM-MCA gives:

| Core | lane2 original | lane2 paired | lane4 original | lane4 paired |
|---|---:|---:|---:|---:|
| Cortex-A53 | 3,904 | 2,704 | 6,304 | 5,104 |
| Cortex-A76 | unchanged | unchanged | unchanged | unchanged |
| Neoverse N1 | unchanged | unchanged | unchanged | unchanged |
| Apple M1 | unchanged | unchanged | unchanged | unchanged |

Values are modeled cycles per 100 region iterations.  This is a 30.7%
lane2 and 19.0% lane4 reduction in the dependent load region on the in-order
A53 model.  Wide out-of-order cores schedule both forms equally.  Because
the production instruction count and code size do not change, the paired
schedule is retained.

### Grouped stores

Replacing four lane4 `STP` stores with two four-register `ST1` stores, and
two lane2 `STP` stores with one `ST1`, passed correctness and reduced dynamic
instruction counts in base-4 paths.  The model was not portable:

| Core | paired `STP` throughput | grouped `ST1` throughput |
|---|---:|---:|
| Cortex-A53 | 4.0 cycles | 4.0 cycles |
| Cortex-A76 | 16.0 | 16.0 |
| Neoverse N1 | 8.0 | 4.0 |
| Apple M1 | 2.0 | 4.0 |

FFmpeg's source also records `STP` as slightly faster on A53 while noting the
theoretical Apple benefit of a structure store.  Without native measurements
that justify a per-core dispatch split, the grouped stores were reverted.

### Armv8.3 complex multiply-add

The analysis-only `FCMLA` region stores each root as
`[wr,wi,wr,wi]`.  Three radix-4 legs then read 48 coefficient bytes instead
of the baseline 96.  Two rotations per leg form the complex products without
reloading the missing add/sub sign mask.

LLVM-MCA results for the complete general and finish regions are:

| Core model | general uops | general throughput | finish uops | finish throughput |
|---|---:|---:|---:|---:|
| Apple M1/M3 | 38 to 35 | 6.3 to 6.3 | 19 to 18 | 3.2 to 3.0 |
| Cortex-A710/A720 | 51 to 46 | 11.5 to 11.5 | 29 to 26 | 4.5 to 4.5 |
| Neoverse V2 | 51 to 46 | 4.8 to 4.0 | 29 to 26 | 1.8 to 1.6 |
| AmpereOne | 37 to 34 | 9.5 to 9.5 | 17 to 17 | 4.3 to 4.3 |

Uops are per iteration; throughput is modeled cycles.  Neoverse V2 has the
clearest arithmetic-throughput gain, while the other models primarily gain
coefficient bandwidth and front-end margin.  A production version needs a
second coefficient layout plus runtime `HWCAP_FCMA` dispatch.  It is not
enabled until it can be timed on native Armv8.3 hardware.

## Final retained x86 result

The precise cycle harness covers the four AVX lane4 entries, lane4 SSE, and
native FFmpeg through every power of two from 16 to 8192:

```sh
make lane4-cycles
```

The final Zen 2 medians were:

| N | lane4 SSE | best lane4 AVX family | FFmpeg native |
|---:|---:|---:|---:|
| 16 | 54.9 | 47.3 | 103.5 |
| 32 | 124.9 | 69.1 | 135.5 |
| 64 | 251.0 | 145.4 | 316.0 |
| 128 | 554.7 | 287.0 | 582.3 |
| 256 | 1,167.7 | 605.9 | 1,235.3 |
| 512 | 2,537.1 | 1,298.1 | 2,470.2 |
| 1024 | 5,340.6 | 2,923.8 | 5,273.4 |
| 2048 | 12,230.0 | 6,630.1 | 11,907.1 |
| 4096 | 26,100.6 | 16,050.6 | 27,855.0 |
| 8192 | 59,243.9 | 36,325.0 | 78,380.2 |

The SSE path beats native FFmpeg at 16--256 and 4096--8192, but remains
2.7%, 1.3%, and 2.7% slower at 512, 1024, and 2048 respectively.  The AVX
lane4 family is faster than native FFmpeg at every tested size on this host.
The wall-clock all-implementation run is recorded in
`benchmark-history-audit.csv`.

## Validation

- `./fft_harness --test`: direct long-double forward/inverse validation
  through 512 and cross-checks through `2^22`, passed.
- `make aarch64-qemu-test`: lane2 NEON and fused lane4 NEON, every power of
  two from 16 through 8192, passed.
- `make gather-cycles`: gather and scalar permutation outputs compare equal
  at every tested size.
- `git diff --check`: no whitespace errors.

The remaining actionable experiment is a separately dispatched FCMLA kernel
on native Armv8.3 hardware.  The other historical ideas have either already
been incorporated structurally or now have a measured rejection.
