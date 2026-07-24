CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?= -lm -pthread

TARGET := fft_harness
CORE_OBJECTS := fft.o ffmpeg_fft.o h16_hybrid.o lane4_portable.o
OBJECTS := $(CORE_OBJECTS) harness.o
FFMPEG_LIB := .ffmpeg-build/libavutil/libavutil.a
NASM ?= nasm
HOST_ARCH := $(shell uname -m)
TARGET_TRIPLE ?= $(shell $(CC) -dumpmachine 2>/dev/null)
AARCH64_CC ?= clang --target=aarch64-linux-gnu
AARCH64_LD ?= $(shell command -v ld.lld 2>/dev/null)
QEMU_AARCH64 ?= qemu-aarch64

ifneq ($(findstring x86_64,$(TARGET_TRIPLE)),)
CPPFLAGS += -DHAVE_TANGENT_X86_ASM=1
LANE4_X86_OBJECTS := lane4_avx.o lane4_avx_fma.o \
	lane4_avx2.o lane4_fft.o lane4_sse_stage.o lane4_avx_stage.o
LANE2_X86_OBJECTS := lane2_sse.o lane2_sse_stage.o
CORE_OBJECTS += tangent_x86_kernel.o tangent_sse_stage.o \
	$(LANE4_X86_OBJECTS) $(LANE2_X86_OBJECTS) lane8_avx.o lane8_avx_stage.o \
	bank8_avx.o bank8_avx_stage.o h16_paired_avx2.o
OBJECTS += tangent_x86_kernel.o tangent_sse_stage.o \
	$(LANE4_X86_OBJECTS) $(LANE2_X86_OBJECTS) lane8_avx.o lane8_avx_stage.o \
	bank8_avx.o bank8_avx_stage.o h16_paired_avx2.o
else
CPPFLAGS += -DHAVE_TANGENT_X86_ASM=0
endif

ifneq ($(findstring aarch64,$(TARGET_TRIPLE)),)
CPPFLAGS += -DHAVE_TANGENT_AARCH64_ASM=1
LANE2_AARCH64_OBJECTS := lane2_neon.o lane2_neon_stage.o \
	lane4_neon_stage.o
CORE_OBJECTS += $(LANE2_AARCH64_OBJECTS)
OBJECTS += $(LANE2_AARCH64_OBJECTS)
else
CPPFLAGS += -DHAVE_TANGENT_AARCH64_ASM=0
endif

CPPFLAGS += -I. -Ithird_party/ffmpeg -I.ffmpeg-build

.PHONY: all clean test bench debug ffmpeg ffmpeg-cycles tangent-cycles \
	lane2-cycles lane4-cycles bank8-cycles lane8-profile \
	bank8-mca \
	aarch64-asm-check aarch64-qemu-test \
	gather-cycles ffmpeg-aarch64-qemu-test aarch64-instruction-counts \
	aarch64-phone-bench check-aarch64-linker
.NOTPARALLEL: debug

all: $(TARGET)

$(TARGET): $(OBJECTS) $(FFMPEG_LIB)
	$(CC) $(LDFLAGS) -o $@ $(OBJECTS) $(FFMPEG_LIB) $(LDLIBS)

fft.o: fft.c fft.h ffmpeg_fft.h h16_hybrid.h tangent_x86_asm.h lane4_fft.h \
	lane4_portable.h tangent_sse_asm.h lane2_sse.h lane2_neon.h \
	lane8_avx.h bank8_avx.h
h16_hybrid.o: h16_hybrid.c h16_hybrid.h fft.h
ffmpeg_fft.o: ffmpeg_fft.c ffmpeg_fft.h fft.h \
	third_party/ffmpeg/libavutil/tx.h third_party/ffmpeg/libavutil/cpu.h \
	| $(FFMPEG_LIB)
harness.o: harness.c fft.h

lane4_portable.o: lane4_portable.c lane4_portable.h \
	lane4_portable_internal.h fft.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -fno-tree-vectorize -c -o $@ $<

lane2_sse.o: lane2_sse.c lane2_sse.h fft.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -fno-tree-vectorize -c -o $@ $<

lane2_neon.o: lane2_neon.c lane2_neon.h fft.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -fno-tree-vectorize -c -o $@ $<

lane2_neon_stage.o: lane2_neon_stage.S
	$(CC) $(CPPFLAGS) -c -o $@ $<

lane4_neon_stage.o: lane4_neon_stage.S
	$(CC) $(CPPFLAGS) -c -o $@ $<

lane8_avx.o: lane8_avx.c lane8_avx.h lane8_avx_internal.h fft.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -fno-tree-vectorize -c -o $@ $<

lane8_avx_stage.o: lane8_avx.asm \
	third_party/ffmpeg/libavutil/x86/x86inc.asm | $(FFMPEG_LIB)
	$(NASM) -f elf64 -g -F dwarf -Ithird_party/ffmpeg/ \
		-P.ffmpeg-build/config.asm -o $@ $<

bank8_avx.o: bank8_avx.c bank8_avx.h bank8_avx_internal.h fft.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -fno-tree-vectorize -c -o $@ $<

bank8_avx_stage.o: bank8_avx.asm
	$(NASM) -f elf64 -g -F dwarf -o $@ $<

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

h16_paired_avx2.o: h16_paired_avx2.asm
	$(NASM) -f elf64 -g -F dwarf -o $@ $<

lane4_avx_stage.o: lane4_avx.asm
	$(NASM) -f elf64 -g -F dwarf -o $@ $<

tangent_sse_stage.o: tangent_sse_stage.asm
	$(NASM) -f elf64 -g -F dwarf -o $@ $<

lane4_sse_stage.o: lane4_sse_stage.asm \
	third_party/ffmpeg/libavutil/x86/x86inc.asm | $(FFMPEG_LIB)
	$(NASM) -f elf64 -g -F dwarf -Ithird_party/ffmpeg/ \
		-P.ffmpeg-build/config.asm -o $@ $<

lane2_sse_stage.o: lane2_sse_stage.asm \
	third_party/ffmpeg/libavutil/x86/x86inc.asm | $(FFMPEG_LIB)
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

lane4-cycles: analysis/lane4_cycles
	taskset -c 2 ./analysis/lane4_cycles

gather-cycles: analysis/gather_cycles
	taskset -c 2 ./analysis/gather_cycles

aarch64-asm-check:
	$(AARCH64_CC) -c lane2_neon_stage.S \
		-o analysis/lane2_neon_stage.aarch64.o

check-aarch64-linker:
	@test -n "$(AARCH64_LD)" || { \
		echo "error: ld.lld was not found; set AARCH64_LD"; \
		exit 1; \
	}

analysis/lane2_neon_qemu_test: analysis/lane2_neon_qemu_test.c \
		analysis/freestanding/math.h analysis/freestanding/stddef.h \
		analysis/freestanding/stdint.h analysis/freestanding/stdlib.h \
		lane2_neon.c lane2_neon.h lane2_neon_stage.S \
		lane4_portable.c lane4_portable.h lane4_portable_internal.h \
		lane4_neon_stage.S fft.h | check-aarch64-linker
	$(AARCH64_CC) -O2 -std=c11 -Wall -Wextra -Wpedantic \
		-ffreestanding -fno-stack-protector -nostdlibinc \
		-DHAVE_TANGENT_AARCH64_ASM=1 -DHAVE_TANGENT_X86_ASM=0 \
		-Ianalysis/freestanding -I. -nostdlib -static \
		--ld-path=$(AARCH64_LD) \
		-Wl,-e,_start -o $@ analysis/lane2_neon_qemu_test.c \
		lane2_neon.c lane2_neon_stage.S lane4_portable.c \
		lane4_neon_stage.S

aarch64-qemu-test: analysis/lane2_neon_qemu_test
	$(QEMU_AARCH64) ./analysis/lane2_neon_qemu_test

analysis/ffmpeg_neon_qemu_test: analysis/ffmpeg_neon_qemu_test.c \
		analysis/freestanding/stddef.h analysis/freestanding/stdint.h \
		third_party/ffmpeg/libavutil/aarch64/tx_float_neon.S \
		third_party/ffmpeg/libavutil/aarch64/asm.S .ffmpeg-build/config.h \
		| check-aarch64-linker
	$(AARCH64_CC) -O2 -std=c11 -Wall -Wextra -Wpedantic \
		-ffreestanding -fno-stack-protector -nostdlibinc \
		-Ianalysis/freestanding -I. -Ithird_party/ffmpeg -I.ffmpeg-build \
		-nostdlib -static --ld-path=$(AARCH64_LD) \
		-Wl,-e,_start -o $@ analysis/ffmpeg_neon_qemu_test.c \
		third_party/ffmpeg/libavutil/aarch64/tx_float_neon.S

ffmpeg-aarch64-qemu-test: analysis/ffmpeg_neon_qemu_test
	$(QEMU_AARCH64) ./analysis/ffmpeg_neon_qemu_test

analysis/aarch64_phone_bench: analysis/aarch64_phone_bench.c \
		analysis/freestanding/math.h analysis/freestanding/stddef.h \
		analysis/freestanding/stdint.h analysis/freestanding/stdlib.h \
		lane2_neon.c lane2_neon.h lane2_neon_stage.S \
		lane4_portable.c lane4_portable.h lane4_portable_internal.h \
		lane4_neon_stage.S fft.h \
		third_party/ffmpeg/libavutil/aarch64/tx_float_neon.S \
		third_party/ffmpeg/libavutil/aarch64/asm.S .ffmpeg-build/config.h \
		| check-aarch64-linker
	$(AARCH64_CC) -O3 -std=c11 -Wall -Wextra -Wpedantic \
		-ffreestanding -fno-stack-protector -nostdlibinc \
		-DHAVE_TANGENT_AARCH64_ASM=1 -DHAVE_TANGENT_X86_ASM=0 \
		-Ianalysis/freestanding -I. -Ithird_party/ffmpeg -I.ffmpeg-build \
		-nostdlib -static --ld-path=$(AARCH64_LD) \
		-Wl,-e,_start -o $@ analysis/aarch64_phone_bench.c \
		lane2_neon.c lane2_neon_stage.S lane4_portable.c \
		lane4_neon_stage.S \
		third_party/ffmpeg/libavutil/aarch64/tx_float_neon.S

aarch64-phone-bench: analysis/aarch64_phone_bench
	@echo "Push analysis/aarch64_phone_bench to /data/local/tmp with adb."

aarch64-instruction-counts:
	AARCH64_CC="$(AARCH64_CC)" AARCH64_LD="$(AARCH64_LD)" \
		QEMU_AARCH64="$(QEMU_AARCH64)" \
		./scripts/count_aarch64_instructions.sh

analysis/x86_tsc.o: analysis/x86_tsc.asm
	$(NASM) -f elf64 -g -F dwarf -o $@ $<

analysis/gather_cycles.o: analysis/gather_cycles.asm
	$(NASM) -f elf64 -g -F dwarf -o $@ $<

analysis/gather_cycles: analysis/gather_cycles.c analysis/gather_cycles.o \
	analysis/x86_tsc.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

analysis/tangent_cycles: analysis/tangent_cycles.c analysis/x86_tsc.o \
	$(CORE_OBJECTS) $(FFMPEG_LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ $< $(CORE_OBJECTS) \
		analysis/x86_tsc.o $(FFMPEG_LIB) $(LDLIBS)

analysis/lane2_cycles: analysis/lane2_cycles.c analysis/x86_tsc.o \
	$(CORE_OBJECTS) $(FFMPEG_LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ $< $(CORE_OBJECTS) \
		analysis/x86_tsc.o $(FFMPEG_LIB) $(LDLIBS)

analysis/lane4_cycles: analysis/lane4_cycles.c analysis/x86_tsc.o \
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

analysis/bank8_cycles: analysis/bank8_cycles.c $(CORE_OBJECTS) \
	analysis/x86_tsc.o $(FFMPEG_LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ analysis/bank8_cycles.c \
		$(CORE_OBJECTS) analysis/x86_tsc.o $(FFMPEG_LIB) $(LDLIBS)

bank8-cycles: analysis/bank8_cycles
	taskset -c 2 ./analysis/bank8_cycles

bank8-mca: analysis/bank8_mca.s
	llvm-mca -mcpu=znver2 -iterations=100 analysis/bank8_mca.s
	llvm-mca -mcpu=skylake -iterations=100 analysis/bank8_mca.s

debug: CFLAGS := -O0 -g3 -std=c11 -Wall -Wextra -Wpedantic \
	-fsanitize=address,undefined
debug: LDFLAGS := -fsanitize=address,undefined
debug: clean $(TARGET)

clean:
	$(RM) $(TARGET) $(OBJECTS) analysis/ffmpeg_cycles \
		analysis/tangent_cycles analysis/lane2_cycles \
		analysis/lane4_cycles analysis/bank8_cycles \
		analysis/gather_cycles analysis/gather_cycles.o \
		analysis/x86_tsc.o \
		analysis/lane8_profile \
		lane2_neon.o lane2_neon_stage.o \
		lane4_neon_stage.o \
		analysis/lane2_neon_stage.aarch64.o \
		analysis/lane2_neon_qemu_test \
		analysis/ffmpeg_neon_qemu_test \
		analysis/aarch64_phone_bench \
		lane4_sse.o lane4_sse2.o \
		lane4_sse3.o lane4_ssse3.o lane4_sse41.o lane4_sse42.o
