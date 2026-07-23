# Bundled FFmpeg comparison dependency

`ffmpeg/` is a source snapshot from the official FFmpeg repository at commit:

```text
80eb9e99b93491753fc9d5bf1f8718c361131998
```

The benchmark uses FFmpeg's public `libavutil/tx.h` API with
`AV_TX_FLOAT_FFT`. Only a static `libavutil` is built; FFmpeg programs, codecs,
formats, filters, networking, documentation, and unrelated libraries are
disabled.

The vendored tree is a build-minimal snapshot of that commit: `libavutil`, its
x86 assembly, required headers/configuration metadata, licenses, and
`tests/checkasm/av_tx.c`. The root FFmpeg Makefile is reduced only by omitting
program, documentation, and test-suite includes that are absent from this
snapshot.

`ffmpeg/libavutil/x86/tx_float.asm` carries a local, benchmarked patch on top
of that commit. It folds exact sign-XOR/add sequences into FMA3 operations only
for the FFT16 leaves inside the 64-point-and-larger split-radix path. The
standalone FFT8/FFT16/FFT32 arithmetic paths and the non-FMA path remain
upstream. Details, correctness results, and the baseline comparison are in
`../docs/ffmpeg-todo-investigation.md`.

FFmpeg is primarily licensed under LGPL-2.1-or-later. The snapshot retains the
upstream license notices alongside the included source, including
`ffmpeg/COPYING.LGPLv2.1` and `ffmpeg/LICENSE.md`.

The x86 SIMD routines require NASM. The build uses `nasm` from `PATH`, or the
command supplied as `make NASM=/path/to/nasm`.
