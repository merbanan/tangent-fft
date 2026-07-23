CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?= -lm -pthread

TARGET := fft_harness
CORE_OBJECTS := fft.o ffmpeg_fft.o lane4_portable.o
OBJECTS := $(CORE_OBJECTS) harness.o
FFMPEG_LIB := .ffmpeg-build/libavutil/libavutil.a
NASM ?= nasm
HOST_ARCH := $(shell uname -m)

ifeq ($(HOST_ARCH),x86_64)
CPPFLAGS += -DHAVE_TANGENT_X86_ASM=1
LANE4_X86_OBJECTS := lane4_avx.o lane4_avx_fma.o \
	lane4_avx2.o lane4_fft.o lane4_sse_stage.o lane4_avx_stage.o
LANE2_X86_OBJECTS := lane2_sse.o lane2_sse_stage.o
CORE_OBJECTS += tangent_x86_kernel.o tangent_sse_stage.o \
	$(LANE4_X86_OBJECTS) $(LANE2_X86_OBJECTS) lane8_avx.o lane8_avx_stage.o
OBJECTS += tangent_x86_kernel.o tangent_sse_stage.o \
	$(LANE4_X86_OBJECTS) $(LANE2_X86_OBJECTS) lane8_avx.o lane8_avx_stage.o
else
CPPFLAGS += -DHAVE_TANGENT_X86_ASM=0
endif

CPPFLAGS += -I. -Ithird_party/ffmpeg -I.ffmpeg-build

.PHONY: all clean test bench debug ffmpeg ffmpeg-cycles tangent-cycles \
	lane2-cycles lane8-profile
.NOTPARALLEL: debug

all: $(TARGET)

$(TARGET): $(OBJECTS) $(FFMPEG_LIB)
	$(CC) $(LDFLAGS) -o $@ $(OBJECTS) $(FFMPEG_LIB) $(LDLIBS)

fft.o: fft.c fft.h ffmpeg_fft.h tangent_x86_asm.h lane4_fft.h \
	lane4_portable.h tangent_sse_asm.h lane2_sse.h
ffmpeg_fft.o: ffmpeg_fft.c ffmpeg_fft.h fft.h \
	third_party/ffmpeg/libavutil/tx.h third_party/ffmpeg/libavutil/cpu.h
harness.o: harness.c fft.h

lane4_portable.o: lane4_portable.c lane4_portable.h \
	lane4_portable_internal.h fft.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -fno-tree-vectorize -c -o $@ $<

lane2_sse.o: lane2_sse.c lane2_sse.h fft.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -fno-tree-vectorize -c -o $@ $<

lane8_avx.o: lane8_avx.c lane8_avx.h lane8_avx_internal.h fft.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -fno-tree-vectorize -c -o $@ $<

lane8_avx_stage.o: lane8_avx.asm \
	third_party/ffmpeg/libavutil/x86/x86inc.asm .ffmpeg-build/config.asm
	$(NASM) -f elf64 -g -F dwarf -Ithird_party/ffmpeg/ \
		-P.ffmpeg-build/config.asm -o $@ $<

lane4_fft.o: lane4_fft.c lane4_fft.h fft.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -fno-tree-vectorize -c -o $@ $<

lane4_avx.o: lane4_fft.c lane4_fft.h fft.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -fno-tree-vectorize \
		-DLANE4_BUILD_AVX=1 -c -o $@ $<

lane4_avx_fma.o: lane4_fft.c lane4_fft.h fft.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -fno-tree-vectorize \
		-DLANE4_BUILD_AVX_FMA=1 -c -o $@ $<

lane4_avx2.o: lane4_fft.c lane4_fft.h fft.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -fno-tree-vectorize \
		-DLANE4_BUILD_AVX2=1 -c -o $@ $<

tangent_x86_kernel.o: tangent_x86_kernel.asm
	$(NASM) -f elf64 -g -F dwarf -o $@ $<

lane4_avx_stage.o: lane4_avx.asm
	$(NASM) -f elf64 -g -F dwarf -o $@ $<

tangent_sse_stage.o: tangent_sse_stage.asm
	$(NASM) -f elf64 -g -F dwarf -o $@ $<

lane4_sse_stage.o: lane4_sse_stage.asm \
	third_party/ffmpeg/libavutil/x86/x86inc.asm .ffmpeg-build/config.asm
	$(NASM) -f elf64 -g -F dwarf -Ithird_party/ffmpeg/ \
		-P.ffmpeg-build/config.asm -o $@ $<

lane2_sse_stage.o: lane2_sse_stage.asm \
	third_party/ffmpeg/libavutil/x86/x86inc.asm .ffmpeg-build/config.asm
	$(NASM) -f elf64 -g -F dwarf -Ithird_party/ffmpeg/ \
		-P.ffmpeg-build/config.asm -o $@ $<

$(FFMPEG_LIB): scripts/build_ffmpeg.sh \
	third_party/ffmpeg/libavutil/x86/tx_float.asm
	NASM="$(NASM)" ./scripts/build_ffmpeg.sh

ffmpeg: $(FFMPEG_LIB)

test: $(TARGET)
	./$(TARGET) --test

bench: $(TARGET)
	./$(TARGET) --bench

ffmpeg-cycles: analysis/ffmpeg_cycles
	taskset -c 2 ./analysis/ffmpeg_cycles

tangent-cycles: analysis/tangent_cycles
	taskset -c 2 ./analysis/tangent_cycles

lane2-cycles: analysis/lane2_cycles
	taskset -c 2 ./analysis/lane2_cycles

analysis/x86_tsc.o: analysis/x86_tsc.asm
	$(NASM) -f elf64 -g -F dwarf -o $@ $<

analysis/tangent_cycles: analysis/tangent_cycles.c analysis/x86_tsc.o \
	$(CORE_OBJECTS) $(FFMPEG_LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ $< $(CORE_OBJECTS) \
		analysis/x86_tsc.o $(FFMPEG_LIB) $(LDLIBS)

analysis/lane2_cycles: analysis/lane2_cycles.c analysis/x86_tsc.o \
	$(CORE_OBJECTS) $(FFMPEG_LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ $< $(CORE_OBJECTS) \
		analysis/x86_tsc.o $(FFMPEG_LIB) $(LDLIBS)

analysis/ffmpeg_cycles: analysis/ffmpeg_cycles.c analysis/x86_tsc.o \
	ffmpeg_fft.o $(FFMPEG_LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ $< ffmpeg_fft.o \
		analysis/x86_tsc.o $(FFMPEG_LIB) $(LDLIBS)

lane8-profile: analysis/lane8_profile
	taskset -c 2 ./analysis/lane8_profile

analysis/lane8_profile: analysis/lane8_profile.c lane8_avx.o lane8_avx_stage.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ $< \
		lane8_avx.o lane8_avx_stage.o $(LDLIBS)

debug: CFLAGS := -O0 -g3 -std=c11 -Wall -Wextra -Wpedantic \
	-fsanitize=address,undefined
debug: LDFLAGS := -fsanitize=address,undefined
debug: clean $(TARGET)

clean:
	$(RM) $(TARGET) $(OBJECTS) analysis/ffmpeg_cycles \
		analysis/tangent_cycles analysis/lane2_cycles \
		analysis/x86_tsc.o \
		analysis/lane8_profile \
		lane4_sse.o lane4_sse2.o \
		lane4_sse3.o lane4_ssse3.o lane4_sse41.o lane4_sse42.o
