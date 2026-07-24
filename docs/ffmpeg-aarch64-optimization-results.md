# FFmpeg AArch64 FFT optimization and lane4 comparison

## Implemented FFmpeg changes

Two changes were made in
`third_party/ffmpeg/libavutil/aarch64/tx_float_neon.S`.

### Direct page-relative FFT8 constant loads

The old FFT8 and FFT8_X2 expansion used:

```asm
adrp/add address
ld1 vector, [address]
```

`LOAD_TAB8` now uses `ADRP` plus an `LDR Q` with the page offset in the load.
The Apple `@PAGE/@PAGEOFF` relocation spelling and ELF/COFF `:lo12:` spelling
are both handled. This saves one instruction at every expanded FFT8:

| Function | Old instructions | New instructions | Saved |
|---|---:|---:|---:|
| FFT8 natural | 59 | 58 | 1 |
| FFT8 pre-shuffled | 42 | 41 | 1 |
| FFT16 natural | 138 | 137 | 1 |
| FFT16 pre-shuffled | 105 | 104 | 1 |
| FFT32 natural | 326 | 324 | 2 |
| FFT32 pre-shuffled | 261 | 259 | 2 |
| split-radix natural body | 5,672 | 5,668 | 4 |
| split-radix pre-shuffled body | 5,551 | 5,547 | 4 |

The complete object text decreased from 48,940 to 48,860 bytes: 80 bytes.
The reduction is slightly larger than the sum of function-body changes
because less alignment padding is required.

### FFT64 reload scheduling

The apparent FFT64 “store then immediately reload” is not a removable
round-trip. The four `LDP` pairs recover older FFT32 values after all 32 NEON
registers have been reused by two FFT16 children. Keeping all eight Q
registers live would require GPR moves or another spill and costs more
instructions than the four paired loads.

The safe optimization moves two reload pairs into registers that become dead
after the first combine and carries them across the independent second
combine. The other two pairs remain after that combine. Memory traffic and
instruction count are unchanged, but two loads are no longer immediately on
the third combine's dependency path.

This corrects the earlier assumption that four stores and four loads could
simply be deleted.

## First-party NEON changes derived from the audit

The complete FFmpeg ARM/AArch64 source audit is recorded in
[`arm-optimization-audit.md`](arm-optimization-audit.md). The following
changes were applied to the first-party lane kernels:

- lane2 folds four row-pointer advances into post-index stores and folds the
  96-byte coefficient advance into the first paired coefficient load;
- lane4 stores each four-frequency real/imaginary coefficient pair together,
  so three post-index `LDP Q` instructions consume the entire finish packet;
- lane4 consumers use the native `TRN1`/`TRN2` destination mapping, removing
  three register copies from every 4x4 transpose;
- paired permutation loads replace eight serialized scalar index loads in
  the FFT8 gather leaf;
- independent coefficient pairs are loaded before their arithmetic users;
- the generic wrapper carries the transform size through private calls, and
  the fixed N=32/N=64 paths carry private state in caller-saved registers
  instead of saving four callee-saved registers.

The lane4 assembly body decreased from 2,684 to 2,360 text bytes (671 to 590
AArch64 instructions), while its fixed coefficient data remained 256 bytes.
The lane2 text decreased from 2,304 to 2,176 bytes.

## Correctness

`make ffmpeg-aarch64-qemu-test` builds the exact modified FFmpeg assembly in a
freestanding AArch64 executable. It supplies libavutil-compatible cosine
tables and split-radix maps, then checks N=16 through N=256 against a
double-precision reference FFT.

The production lane2 and lane4 kernels are checked by
`make aarch64-qemu-test` at every power of two from N=16 through N=8192.

## Executed AArch64 instruction counts

QEMU's one-instruction-per-translation-block mode counts only instructions
whose program counters are inside the transform symbol. Planner work,
reference calculation, and syscall code are excluded.

```sh
make aarch64-instruction-counts \
  AARCH64_LD=/path/to/ld.lld \
  QEMU_AARCH64=/path/to/qemu-aarch64
```

| N | lane2 NEON | fused lane4 NEON | FFmpeg natural input | FFmpeg pre-shuffled |
|---:|---:|---:|---:|---:|
| 16 | 95 | 80 | 137 | 104 |
| 32 | 392 | 278 | 324 | 259 |
| 64 | 791 | 605 | 770 | 649 |
| 128 | 1,902 | 1,215 | 1,773 | 1,528 |
| 256 | 3,999 | 2,777 | 4,013 | 3,528 |
| 512 | 9,406 | 5,978 | 9,014 | 8,041 |
| 1024 | 19,807 | 13,496 | 19,977 | 18,036 |
| 2048 | 45,422 | 28,921 | 43,918 | 40,033 |
| 4096 | 95,087 | 64,199 | 95,705 | 87,940 |
| 8192 | 213,630 | 136,392 | 207,214 | 191,681 |

All entries are dynamic traces of the executed assembly. Lane4 uses 34.2%
fewer transform instructions than FFmpeg's natural-input path and 28.8%
fewer than its pre-shuffled path at N=8192.

Relative to the previous first-party NEON code, lane4 removes 10 instructions
at N=16, 95 at N=64, and 8,287 at N=8192. Lane2 is unchanged at its already
fused N=16 leaf and removes 24,918 instructions at N=8192.

Instruction count is not a cycle result. It does not model issue width,
load-use latency, cache behavior, or the throughput differences between
Cortex, Neoverse, and Apple cores. It does show that the fused SoA schedule is
structurally competitive rather than relying on a favorable instruction-cost
assumption.

## Current performance interpretation

- N=16 is the clearest NEON win candidate: the natural-input lane4 leaf is 80
  instructions, versus 95 for the fused lane2 leaf and 137 for FFmpeg.
- N=32 trades extra Stockham traffic for natural-order input/output. It beats
  FFmpeg's natural-input count but remains 7.3% above the pre-shuffled leaf.
- N=64 now beats both FFmpeg entry modes in instruction count.
- From N=128 onward, the lane-parallel radix-4 stages amortize the layout work
  and use fewer instructions than both FFmpeg entry modes.
- A real ARM cycle benchmark is still required before claiming the lane4
  kernel is faster on a particular CPU.
