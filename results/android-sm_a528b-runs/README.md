# Samsung SM-A528B native ARM benchmark

These are native measurements from a Samsung SM-A528B reporting SoC
`SM7325`, Android 14, and Linux 5.4.254.  `/proc/cpuinfo` identifies CPUs
0-3 as Arm part `0xd05` (Cortex-A55) and CPUs 4-7 as part `0xd41`
(Cortex-A78).  Android allowed the ADB shell to use CPU 3 and CPU 6, so the
benchmark was pinned as follows:

- Cortex-A55: CPU 3, `taskset 08`, 300-1804.8 MHz policy.
- Cortex-A78: CPU 6, `taskset 40`, 691.2-2400 MHz policy.

The freestanding static AArch64 harness directly links the production
`lane2_neon_stage.S`, `lane4_neon_stage.S`, and the vendored FFmpeg
`tx_float_neon.S`.  Both native correctness programs passed before timing:

```text
PASS: lane2-neon and fused lane4-neon, N=16..8192
PASS: FFmpeg NEON FFT assembly
```

Each cell below is the median of five complete outer runs.  Each outer run
itself reports the median of 31 samples after 64 warm-up transforms.  Plan
creation and twiddle generation are outside the timed region.  All ten FFT
sizes from 16 through 8192 are included.  CPU temperatures remained between
31.6 and 35.5 degrees C in the recorded post-run snapshots.

`ffmpeg-natural` includes FFmpeg's natural-input permutation.
`ffmpeg-pre` calls its pre-permuted kernel and is therefore a useful
kernel-only lower bound, not an equivalent natural-input API comparison.
Ratios below are candidate time divided by FFmpeg time, so values below
1.0 favor the candidate.

## Cortex-A55 results

| N | lane2 ns | lane4 ns | FFmpeg natural ns | FFmpeg pre ns | lane2/natural | lane4/natural | lane4/pre |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 92 | 67 | 92 | 86 | 1.000 | 0.728 | 0.779 |
| 32 | 337 | 254 | 208 | 198 | 1.620 | 1.221 | 1.283 |
| 64 | 695 | 546 | 522 | 501 | 1.331 | 1.046 | 1.090 |
| 128 | 1730 | 1061 | 1228 | 1186 | 1.409 | 0.864 | 0.895 |
| 256 | 3648 | 2567 | 2880 | 2778 | 1.267 | 0.891 | 0.924 |
| 512 | 8765 | 5491 | 6554 | 6373 | 1.337 | 0.838 | 0.862 |
| 1024 | 19841 | 12942 | 32978 | 14415 | 0.602 | 0.392 | 0.898 |
| 2048 | 48958 | 31833 | 49895 | 32981 | 0.981 | 0.638 | 0.965 |
| 4096 | 117218 | 85071 | 191683 | 79201 | 0.612 | 0.444 | 1.074 |
| 8192 | 290064 | 199616 | 213921 | 177881 | 1.356 | 0.933 | 1.122 |

Lane4 beats FFmpeg's equivalent natural-input path at 8 of 10 sizes.  It
loses by 22.1% at N=32 and 4.6% at N=64.  Against FFmpeg's pre-permuted
lower bound it wins at N=16 and N=128 through N=2048, then loses at N=4096
and N=8192.  The worst outer-run spread was 2.224% (lane2, N=2048).

## Cortex-A78 results

| N | lane2 ns | lane4 ns | FFmpeg natural ns | FFmpeg pre ns | lane2/natural | lane4/natural | lane4/pre |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 26 | 25 | 33 | 31 | 0.788 | 0.758 | 0.806 |
| 32 | 64 | 84 | 81 | 72 | 0.790 | 1.037 | 1.167 |
| 64 | 148 | 186 | 199 | 182 | 0.744 | 0.935 | 1.022 |
| 128 | 346 | 300 | 455 | 417 | 0.760 | 0.659 | 0.719 |
| 256 | 801 | 672 | 1040 | 964 | 0.770 | 0.646 | 0.697 |
| 512 | 1806 | 1494 | 2330 | 2179 | 0.775 | 0.641 | 0.686 |
| 1024 | 4192 | 3294 | 7220 | 4868 | 0.581 | 0.456 | 0.677 |
| 2048 | 9505 | 7284 | 13472 | 10790 | 0.706 | 0.541 | 0.675 |
| 4096 | 20954 | 16293 | 26832 | 23619 | 0.781 | 0.607 | 0.690 |
| 8192 | 48851 | 35804 | 58857 | 51569 | 0.830 | 0.608 | 0.694 |

Lane4 beats FFmpeg's natural-input path at 9 of 10 sizes; N=32 is 3.7%
slower.  It also beats the pre-permuted kernel at 8 of 10 sizes, losing by
16.7% at N=32 and 2.2% at N=64.  For N=128 through N=8192, lane4 is
28.1-32.5% faster than FFmpeg's pre-permuted kernel and 34.1-54.4% faster
than its natural-input path.  The worst outer-run spread was 4.000% at N=16
because the integer-nanosecond output changed by one nanosecond; excluding
N=16 it was 2.706% (lane4, N=8192).

## Reproduction

Build and push the benchmark and native correctness programs:

```sh
make analysis/lane2_neon_qemu_test \
     analysis/ffmpeg_neon_qemu_test \
     analysis/aarch64_phone_bench
adb push analysis/lane2_neon_qemu_test \
         analysis/ffmpeg_neon_qemu_test \
         analysis/aarch64_phone_bench /data/local/tmp/
adb shell chmod 755 /data/local/tmp/lane2_neon_qemu_test \
                    /data/local/tmp/ffmpeg_neon_qemu_test \
                    /data/local/tmp/aarch64_phone_bench
adb shell /data/local/tmp/lane2_neon_qemu_test
adb shell /data/local/tmp/ffmpeg_neon_qemu_test
```

Run and summarize the repeated cluster-pinned campaign:

```sh
RUNS=5 COOL_SECONDS=10 sh scripts/run_android_phone_bench.sh
python3 scripts/summarize_android_phone_bench.py \
    results/android-sm_a528b-runs
```

The ten `a55-run*.txt` and `a78-run*.txt` files in this directory are the
unaltered raw console captures.
