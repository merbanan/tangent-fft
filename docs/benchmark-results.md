# Complete benchmark results

This is the complete forward-transform run checked into `benchmark.csv`.
Times are median microseconds per transform. Plan construction and coefficient
generation are excluded; all implementations use the same public in-place
API. Both FFmpeg configurations include copies into and out of AVTX's aligned
buffers.

```sh
make
make test
taskset -c 2 ./fft_harness --bench --min-power 4 --max-power 13 \
  --target-ms 100 --csv benchmark.csv
```

The run was made on the Ryzen 9 3900X host on 2026-07-23. The wall-clock
timer has 0.01-microsecond resolution, so differences among the smallest
transforms should be treated as coarse. Results vary with CPU, frequency
policy, compiler, and system load.

`ffmpeg-avtx` uses FFmpeg's native CPU auto-dispatch, which selects AVX
codelets on this host. `ffmpeg-sse` creates a separate AVTX plan with AVX
disabled and the x86 feature mask capped at SSE4.2. `hw-sse-auto` contains
only legacy SSE instructions. It is 2.00x to 4.32x faster than
`ffmpeg-sse` over this size range.

| implementation | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `radix-2` | 0.140 | 0.310 | 0.670 | 1.250 | 2.760 | 6.060 | 13.190 | 28.520 | 61.950 | 136.866 |
| `split-radix` | 0.180 | 0.350 | 0.750 | 1.330 | 2.840 | 6.200 | 13.410 | 28.790 | 61.830 | 131.921 |
| `tangent` | 0.110 | 0.220 | 0.490 | 0.800 | 1.760 | 3.700 | 7.920 | 16.880 | 36.380 | 82.650 |
| `tangent-x86-asm` | 0.040 | 0.060 | 0.100 | 0.170 | 0.340 | 0.730 | 1.420 | 3.010 | 6.940 | 18.450 |
| `tangent-sse` | 0.080 | 0.120 | 0.220 | 0.370 | 0.760 | 1.580 | 3.350 | 7.130 | 15.730 | 36.600 |
| `tangent-sse2` | 0.080 | 0.120 | 0.220 | 0.370 | 0.770 | 1.580 | 3.340 | 7.130 | 15.750 | 36.610 |
| `tangent-sse3` | 0.070 | 0.120 | 0.220 | 0.360 | 0.760 | 1.560 | 3.260 | 6.930 | 15.330 | 35.660 |
| `tangent-ssse3` | 0.070 | 0.120 | 0.220 | 0.360 | 0.770 | 1.560 | 3.250 | 6.930 | 15.330 | 35.670 |
| `tangent-sse4.1` | 0.070 | 0.120 | 0.220 | 0.360 | 0.760 | 1.560 | 3.220 | 6.930 | 15.320 | 35.670 |
| `tangent-sse4.2` | 0.070 | 0.120 | 0.220 | 0.360 | 0.770 | 1.560 | 3.260 | 6.930 | 15.320 | 35.651 |
| `lane4-c` | 0.090 | 0.180 | 0.540 | 0.670 | 1.500 | 3.330 | 7.430 | 16.210 | 35.450 | 76.721 |
| `lane4-sse` | 0.040 | 0.060 | 0.140 | 0.160 | 0.320 | 0.710 | 1.480 | 3.330 | 7.120 | 16.490 |
| `lane4-sse2` | 0.040 | 0.060 | 0.100 | 0.160 | 0.320 | 0.710 | 1.480 | 3.330 | 7.130 | 16.490 |
| `lane4-sse3` | 0.050 | 0.060 | 0.100 | 0.160 | 0.320 | 0.710 | 1.480 | 3.330 | 7.130 | 16.490 |
| `lane4-ssse3` | 0.040 | 0.060 | 0.100 | 0.160 | 0.540 | 0.710 | 1.480 | 3.330 | 7.130 | 16.490 |
| `lane4-sse4.1` | 0.040 | 0.060 | 0.080 | 0.160 | 0.320 | 0.710 | 1.480 | 3.330 | 7.130 | 16.500 |
| `lane4-sse4.2` | 0.040 | 0.060 | 0.080 | 0.160 | 0.320 | 0.710 | 1.480 | 3.330 | 7.130 | 16.470 |
| `lane4-avx` | 0.030 | 0.050 | 0.050 | 0.090 | 0.180 | 0.390 | 0.790 | 1.830 | 4.290 | 9.960 |
| `lane4-avx-fma` | 0.040 | 0.050 | 0.050 | 0.090 | 0.180 | 0.370 | 0.780 | 1.800 | 4.210 | 9.770 |
| `lane4-avx2` | 0.040 | 0.050 | 0.050 | 0.090 | 0.180 | 0.370 | 0.800 | 1.830 | 4.300 | 9.990 |
| `lane4-avx2-fma` | 0.040 | 0.050 | 0.060 | 0.090 | 0.180 | 0.370 | 0.780 | 1.790 | 4.160 | 9.830 |
| `lane2-sse` | 0.040 | 0.060 | 0.080 | 0.160 | 0.330 | 0.760 | 1.580 | 3.660 | 7.800 | 20.030 |
| `ffmpeg-avtx` | 0.060 | 0.070 | 0.100 | 0.160 | 0.330 | 0.690 | 1.480 | 3.210 | 7.650 | 21.161 |
| `lane8-avx2-fma` | 0.040 | 0.070 | 0.070 | 0.110 | 0.230 | 0.440 | 0.960 | 1.970 | 4.850 | 10.330 |
| `hw-sse-auto` | 0.040 | 0.060 | 0.080 | 0.160 | 0.320 | 0.710 | 1.470 | 3.330 | 7.130 | 16.470 |
| `ffmpeg-sse` | 0.080 | 0.150 | 0.290 | 0.610 | 1.320 | 2.870 | 6.310 | 13.850 | 30.530 | 71.210 |

The CSV is the authoritative machine-readable result and also records the
minimum time, sample count, theoretical arithmetic count where applicable,
speedup versus radix-2, and output checksum.
