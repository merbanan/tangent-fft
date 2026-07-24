# FFmpeg ARM/AArch64 optimization audit

## Scope

This audit covered the complete vendored `libavutil/aarch64` and
`libavutil/arm` directories, not only the FFT file:

- AArch64 transform assembly and dispatch in `tx_float_neon.S` and
  `tx_float_init.c`;
- AArch64 float DSP, pixel, CRC, CPU-feature, SVE, and SME sources;
- Arm32 NEON, VFP, Armv6 pixel, CPU-feature, and float-DSP sources;
- the shared assembly macros, ABI helpers, timers, and architecture headers.

The transform-specific comparison is against the pinned FFmpeg source under
`third_party/ffmpeg`. Non-FFT files were inspected for reusable low-level
patterns such as pointer updates, load scheduling, register lifetime, and
loop organization.

## Useful patterns found in FFmpeg

### Address generation belongs in memory operations

FFmpeg's DSP and pixel loops use post-index loads/stores heavily. This removes
standalone pointer additions from the loop body and lets a single instruction
both transfer data and advance a stream. Paired `LDP`/`STP` operations are
used whenever two adjacent registers share an address stream.

Applied here:

- lane2's four output-row pointers advance in their `STR Q` instructions;
- lane2's first coefficient `LDP Q` advances the whole 96-byte packet and the
  next two pairs use negative fixed offsets;
- lane4's work and packed-root streams use post-index paired operations;
- lane4 reuses the fourth stage row's post-indexed store pointer as the next
  block pointer instead of constructing the same address separately;
- lane4's FFT8 permutation map uses four `LDP W` instructions instead of
  eight serialized scalar loads.

### Consume the permutation that the ISA naturally produces

The FFmpeg transform code assigns semantic values to the registers produced
by `ZIP`, `TRN`, and `UZP`; it does not first copy them into a cosmetically
ordered register set. Lane4's 4x4 transpose now follows the same rule. Its
eight `TRN` instructions return columns in their native destination mapping,
and the radix/store consumers use that mapping directly. Three `MOV`
instructions were removed from every transpose.

### Match coefficient layout to paired loads

FFmpeg places coefficients that are consumed together next to each other.
The NEON lane4 planner now builds an execution-ordered stream of:

```text
W^(1k).re[4], W^(1k).im[4],
W^(2k).re[4], W^(2k).im[4],
W^(3k).re[4], W^(3k).im[4]
```

Each four-frequency finish group consumes this with three post-index
`LDP Q` instructions. The N=16 and N=32 constant tables use the same
real/imaginary pairing. This changes layout and address work, not the number
of coefficient bytes read by the transform.

### Carry state across private leaves

FFmpeg's fixed transforms exploit private calling relationships: values
remain in caller-saved registers when the exact callee is known. Lane4's
N=32 and N=64 paths now carry work, output, and constant pointers in
`x15-x17` across their private FFT16 calls, whose register contract is under
this project's control. This removes the former four-register save/restore.
The generic wrapper similarly keeps `inner_size` in `x23`, avoiding repeated
plan loads between private stages.

Lane2 already applies this pattern to its selective sign mask and exact
diagonal constants in `v28-v31`.

### Expose independent work to the scheduler

FFmpeg loads independent inputs or constants before a dependent arithmetic
chain and interleaves unrelated combine work with reloads. Lane4 now loads
all three coefficient pairs before the three independent complex products
in its fixed split and finish paths. This does not reduce instruction count,
but avoids imposing unnecessary load-use serialization on in-order cores and
gives wider cores more independent operations.

The FFT4/FFT8 gather leaves apply the same principle to integer address
generation. They alternate between two address registers instead of forming
every address and load through one dependent register. The instruction count
and code size are unchanged. LLVM-MCA's Cortex-A53 model reduces the isolated
lane2 load region from 3,904 to 2,704 cycles per 100 iterations and the lane4
region from 6,304 to 5,104. Cortex-A76, Neoverse N1, and Apple M1 models are
neutral, so the schedule is retained as an in-order improvement without a
wide-core cost.

## Resulting instruction reductions

The counts below come from QEMU's one-instruction-per-translation-block mode,
filtered to the assembly transform address range. They exclude planning,
reference calculations, and syscalls.

| N | lane2 before | lane2 after | lane4 before | lane4 after |
|---:|---:|---:|---:|---:|
| 16 | 95 | 95 | 90 | 80 |
| 32 | 410 | 392 | 312 | 278 |
| 64 | 829 | 791 | 700 | 605 |
| 128 | 2,052 | 1,902 | 1,351 | 1,215 |
| 256 | 4,309 | 3,999 | 3,014 | 2,777 |
| 512 | 10,324 | 9,406 | 6,503 | 5,978 |
| 1024 | 21,685 | 19,807 | 14,422 | 13,496 |
| 2048 | 50,372 | 45,422 | 30,999 | 28,921 |
| 4096 | 105,157 | 95,087 | 67,878 | 64,199 |
| 8192 | 238,548 | 213,630 | 144,679 | 136,392 |

At N=8192 this is a 10.4% instruction reduction for lane2 and 5.7% for
lane4. Lane4's larger small-size reductions come from the packed fixed roots,
native transpose mapping, and private-call register contract.

Static lane4 text decreased from 2,684 to 2,360 bytes; its fixed read-only
tables remain 256 bytes. Lane2 text decreased from 2,304 to 2,176 bytes.

## Ideas evaluated but not enabled

### `FCMLA`

Armv8.3 complex arithmetic can express a complex rotation more compactly. The
analysis model stores each root as `[wr,wi,wr,wi]` and uses the 0- and
90-degree `FCMLA` forms. This halves generic radix-4 coefficient traffic from
96 to 48 bytes and reduces modeled uops:

| Core model | general uops / throughput | FCMLA | finish uops / throughput | FCMLA |
|---|---:|---:|---:|---:|
| Apple M1/M3 | 38 / 6.3 | 35 / 6.3 | 19 / 3.2 | 18 / 3.0 |
| Cortex-A710/A720 | 51 / 11.5 | 46 / 11.5 | 29 / 4.5 | 26 / 4.5 |
| Neoverse V2 | 51 / 4.8 | 46 / 4.0 | 29 / 1.8 | 26 / 1.6 |
| AmpereOne | 37 / 9.5 | 34 / 9.5 | 17 / 4.3 | 17 / 4.3 |

Throughput is in modeled cycles. `FCMLA` is not part of the baseline
AArch64/NEON contract. A production implementation needs a second
coefficient table and runtime `HWCAP_FCMA` dispatch, and its benefit must be
confirmed on native hardware. The analysis regions remain in
`analysis/lane2_neon_mca.s`; the baseline kernel is unchanged.

### Grouped `ST1` stores

Replacing lane4's four `STP` operations with two four-register `ST1`
operations, and lane2's two `STP` operations with one `ST1`, passed QEMU
correctness and reduced dynamic instruction count in base-4 paths. It did not
have one portable performance result: LLVM-MCA favors `ST1` on Neoverse N1,
ties on Cortex-A53/A76, and favors `STP` on Apple M1. FFmpeg's source also
records a slight A53 advantage for `STP`. The grouped form was reverted
pending native, per-core measurements.

### Compact roots expanded by `LD2R`

Storing one `{wr,wi}` pair per coefficient cuts table traffic, but execution
must duplicate lanes and construct the alternating imaginary signs. With L1
hits, LLVM-MCA predicts worse upper-butterfly throughput:

| Core model | expanded roots | compact `LD2R` roots |
|---|---:|---:|
| Cortex-A53 | 19.0 cycles | 20.0 cycles |
| Cortex-A76 | 20.0 cycles | 21.7 cycles |
| Neoverse N1 | 13.5 cycles | 18.0 cycles |
| Apple M1/M3 | 6.3 cycles | 9.3 cycles |

The expanded stream remains appropriate for the requested cache-resident
16--8192 range. A size-gated compact stream would need native cache-miss
measurements.

### Software prefetch

The active data and coefficient streams are sequential after the gather
leaf, and the important sizes are small enough that a fixed prefetch distance
can easily be neutral or harmful. FFmpeg does not establish one portable
distance across Cortex, Neoverse, and Apple cores. No prefetch was added
without native miss and cycle data.

### Extra loop unrolling

FFmpeg unrolls when it exposes multiple accumulators or eliminates a costly
small-loop boundary. Lane4's generic butterflies already have a large live
vector set. Further unrolling duplicates a sizable body, increases I-cache
footprint, and does not reduce coefficient or data traffic. The observed
branch savings alone do not justify it.

### Reconstructing roots arithmetically

Deriving later roots from earlier roots replaces loads with dependent
floating-point operations and accumulates error. FFmpeg's own kernels favor
loaded constants for these cache-resident transforms. The execution-ordered
paired layout retains that choice.

### SVE/SME and Arm32 specialization

FFmpeg's SVE/SME files primarily demonstrate feature detection and scalable
state handling; they do not contain an FFT schedule that can be transplanted.
A scalable-vector FFT needs a new lane geometry, permutation plan, and
tail policy. Arm32 NEON has only half the architectural vector-register file
available to AArch64 code and a different ABI, so the register-resident
lane4 leaves cannot be mechanically reused.

## Validation and interpretation

The production lane2/lane4 assembly passes the independent double-precision
reference test at every power of two from 16 through 8192 under
QEMU AArch64. The exact vendored FFmpeg leaves pass N=16 through N=256.
The normal host test suite also passes.

QEMU instruction traces prove executed control-flow counts and catch assembly
integration errors; they do not predict native cycles, cache behavior, or
front-end throughput. The next meaningful optimization gate is native
measurement on at least an in-order Cortex, an out-of-order Cortex or
Neoverse, and an Apple core. `FCMLA` should be tested as a separately
dispatched variant at that stage.
