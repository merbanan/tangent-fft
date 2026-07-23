#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ffmpeg_source="$project_root/third_party/ffmpeg"
build_dir="$project_root/.ffmpeg-build"
nasm_executable=${NASM:-nasm}

if ! command -v "$nasm_executable" >/dev/null 2>&1; then
    echo "NASM not found: $nasm_executable" >&2
    exit 1
fi
mkdir -p "$build_dir"

if [ ! -f "$build_dir/config.h" ]; then
    cd "$build_dir"
    "$ffmpeg_source/configure" \
        --disable-programs \
        --disable-doc \
        --disable-network \
        --disable-autodetect \
        --disable-avcodec \
        --disable-avdevice \
        --disable-avfilter \
        --disable-avformat \
        --disable-swresample \
        --disable-swscale \
        --disable-shared \
        --enable-static \
        --enable-pic \
        --x86asmexe="$nasm_executable"
fi

make -C "$build_dir" -j4 libavutil/libavutil.a
