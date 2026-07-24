#!/bin/sh
#
# Run the freestanding AArch64 FFT benchmark through ADB on representative
# Cortex-A55 and Cortex-A78 cores.  The caller may override:
#   BENCH_BINARY  Android path to the benchmark
#   OUTPUT_DIR    host path for raw result files
#   RUNS          number of outer repetitions
#   COOL_SECONDS  delay between repetitions

set -eu

bench_binary=${BENCH_BINARY:-/data/local/tmp/aarch64_phone_bench}
output_dir=${OUTPUT_DIR:-analysis/android-sm_a528b-runs}
runs=${RUNS:-5}
cool_seconds=${COOL_SECONDS:-10}

mkdir -p "$output_dir"

run=1
while [ "$run" -le "$runs" ]; do
    for cluster in a55 a78; do
        if [ "$cluster" = a55 ]; then
            mask=08
            policy=0
            core=3
        else
            mask=40
            policy=4
            core=6
        fi

        output_file="$output_dir/$cluster-run$run.txt"
        echo "START cluster=$cluster core=$core run=$run"
        adb shell "
            echo META_cluster=$cluster
            echo META_core=$core
            echo BEFORE_freq_khz=\$(cat /sys/devices/system/cpu/cpufreq/policy$policy/scaling_cur_freq)
            echo BEFORE_cpuss0_mC=\$(cat /sys/class/thermal/thermal_zone44/temp)
            echo BEFORE_cpuss1_mC=\$(cat /sys/class/thermal/thermal_zone45/temp)
            taskset $mask $bench_binary
            echo AFTER_freq_khz=\$(cat /sys/devices/system/cpu/cpufreq/policy$policy/scaling_cur_freq)
            echo AFTER_cpuss0_mC=\$(cat /sys/class/thermal/thermal_zone44/temp)
            echo AFTER_cpuss1_mC=\$(cat /sys/class/thermal/thermal_zone45/temp)
        " >"$output_file"
        tail -n 4 "$output_file"
        echo "DONE cluster=$cluster core=$core run=$run"
        sleep "$cool_seconds"
    done
    run=$((run + 1))
done
