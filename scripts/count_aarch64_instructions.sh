#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

read -r -a aarch64_cc <<< "${AARCH64_CC:-clang --target=aarch64-linux-gnu}"
aarch64_ld="${AARCH64_LD:-}"
if [ -z "$aarch64_ld" ]; then
    aarch64_ld="$(command -v ld.lld || true)"
fi
if [ -z "$aarch64_ld" ]; then
    echo "error: ld.lld was not found; set AARCH64_LD" >&2
    exit 1
fi
qemu_aarch64="${QEMU_AARCH64:-qemu-aarch64}"
task_tmp="$(mktemp -d)"
trap 'rm -rf "$task_tmp"' EXIT

common_cflags=(
    -O2 -std=c11 -ffreestanding -fno-stack-protector -nostdlibinc
    -Ianalysis/freestanding -I. -nostdlib -static
    "--ld-path=$aarch64_ld" -Wl,-e,_start
)

run_filtered()
{
    local binary="$1"
    local symbol="$2"
    local trace="$3"
    local start size end end_hex

    read -r start size < <(
        llvm-nm -S "$binary" |
            awk -v name="$symbol" '$4 == name { print $1, $2 }'
    )
    end=$((16#$start + 16#$size - 1))
    printf -v end_hex '%x' "$end"
    "$qemu_aarch64" -one-insn-per-tb -d exec,nochain \
        -dfilter "0x$start..0x$end_hex" -D "$trace" "$binary" >/dev/null
    wc -l < "$trace"
}

run_filtered_span()
{
    local binary="$1"
    local first_symbol="$2"
    local end_symbol="$3"
    local trace="$4"
    local start end end_hex

    start="$(
        llvm-nm -S "$binary" |
            awk -v name="$first_symbol" '$4 == name { print $1 }'
    )"
    end="$(
        llvm-nm -S "$binary" |
            awk -v name="$end_symbol" '$4 == name { print $1 }'
    )"
    printf -v end_hex '%x' "$((16#$end - 1))"
    "$qemu_aarch64" -one-insn-per-tb -d exec,nochain \
        -dfilter "0x$start..0x$end_hex" -D "$trace" "$binary" >/dev/null
    wc -l < "$trace"
}

compile_lane()
{
    local n="$1"
    local lane2="$2"
    local output="$3"

    "${aarch64_cc[@]}" "${common_cflags[@]}" \
        -DHAVE_TANGENT_AARCH64_ASM=1 -DHAVE_TANGENT_X86_ASM=0 \
        -DTEST_LANE2="$lane2" -DTEST_MIN_N="$n" -DTEST_MAX_N="$n" \
        -o "$output" analysis/lane2_neon_qemu_test.c \
        lane2_neon.c lane2_neon_stage.S lane4_portable.c \
        lane4_neon_stage.S
}

compile_ffmpeg()
{
    local n="$1"
    local natural="$2"
    local output="$3"

    "${aarch64_cc[@]}" "${common_cflags[@]}" \
        -Ithird_party/ffmpeg -I.ffmpeg-build \
        -DTEST_MIN_N="$n" -DTEST_MAX_N="$n" \
        -DTEST_FFMPEG_NATURAL="$natural" \
        -o "$output" analysis/ffmpeg_neon_qemu_test.c \
        third_party/ffmpeg/libavutil/aarch64/tx_float_neon.S
}

ffmpeg_symbol()
{
    local n="$1"
    local natural="$2"

    if [ "$n" -eq 16 ]; then
        if [ "$natural" -eq 1 ]; then
            echo ff_tx_fft16_float_neon
        else
            echo ff_tx_fft16_ns_float_neon
        fi
    elif [ "$n" -eq 32 ]; then
        if [ "$natural" -eq 1 ]; then
            echo ff_tx_fft32_float_neon
        else
            echo ff_tx_fft32_ns_float_neon
        fi
    elif [ "$natural" -eq 1 ]; then
        echo ff_tx_fft_sr_float_neon
    else
        echo ff_tx_fft_sr_ns_float_neon
    fi
}

printf '%8s %12s %12s %15s %18s\n' \
    N lane2-neon lane4-neon ffmpeg-natural ffmpeg-preshuffled

for n in 16 32 64 128 256 512 1024 2048 4096 8192; do
    lane2_binary="$task_tmp/lane2-$n"
    lane4_binary="$task_tmp/lane4-$n"
    natural_binary="$task_tmp/ffmpeg-natural-$n"
    ns_binary="$task_tmp/ffmpeg-ns-$n"

    compile_lane "$n" 1 "$lane2_binary"
    compile_lane "$n" 0 "$lane4_binary"
    compile_ffmpeg "$n" 1 "$natural_binary"
    compile_ffmpeg "$n" 0 "$ns_binary"

    lane2_count="$(
        run_filtered_span "$lane2_binary" lane2_neon_execute \
            lane2_neon_code_end "$task_tmp/lane2-$n.trace"
    )"
    lane4_count="$(
        run_filtered "$lane4_binary" lane4_neon_execute \
            "$task_tmp/lane4-$n.trace"
    )"
    natural_count="$(
        run_filtered "$natural_binary" "$(ffmpeg_symbol "$n" 1)" \
            "$task_tmp/ffmpeg-natural-$n.trace"
    )"
    ns_count="$(
        run_filtered "$ns_binary" "$(ffmpeg_symbol "$n" 0)" \
            "$task_tmp/ffmpeg-ns-$n.trace"
    )"

    printf '%8d %12d %12d %15d %18d\n' \
        "$n" "$lane2_count" "$lane4_count" "$natural_count" "$ns_count"
done
