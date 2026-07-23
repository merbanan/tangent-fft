CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?= -lm -pthread

TARGET := fft_harness
CORE_OBJECTS := fft.o ffmpeg_fft.o
OBJECTS := $(CORE_OBJECTS) harness.o
FFMPEG_LIB := .ffmpeg-build/libavutil/libavutil.a
NASM ?= nasm
HOST_ARCH := $(shell uname -m)

ifeq ($(HOST_ARCH),x86_64)
CPPFLAGS += -DHAVE_TANGENT_X86_ASM=1
CORE_OBJECTS += tangent_x86_kernel.o lane4_fft.o
OBJECTS += tangent_x86_kernel.o lane4_fft.o
else
CPPFLAGS += -DHAVE_TANGENT_X86_ASM=0
endif

CPPFLAGS += -I. -Ithird_party/ffmpeg -I.ffmpeg-build

.PHONY: all clean test bench debug ffmpeg ffmpeg-cycles lane4-experiment
.NOTPARALLEL: debug

all: $(TARGET)

$(TARGET): $(OBJECTS) $(FFMPEG_LIB)
	$(CC) $(LDFLAGS) -o $@ $(OBJECTS) $(FFMPEG_LIB) $(LDLIBS)

fft.o: fft.c fft.h ffmpeg_fft.h tangent_x86_asm.h lane4_fft.h
ffmpeg_fft.o: ffmpeg_fft.c ffmpeg_fft.h fft.h third_party/ffmpeg/libavutil/tx.h
harness.o: harness.c fft.h

lane4_fft.o: lane4_fft.c lane4_fft.h fft.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -mavx2 -mfma -c -o $@ $<

tangent_x86_kernel.o: tangent_x86_kernel.asm
	$(NASM) -f elf64 -g -F dwarf -o $@ $<

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

analysis/ffmpeg_cycles: analysis/ffmpeg_cycles.c ffmpeg_fft.o $(FFMPEG_LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ $< ffmpeg_fft.o \
		$(FFMPEG_LIB) $(LDLIBS)

lane4-experiment: analysis/lane4_experiment
	taskset -c 2 ./analysis/lane4_experiment

analysis/lane4_experiment: analysis/lane4_experiment.c $(CORE_OBJECTS) $(FFMPEG_LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) -mavx2 -mfma $(LDFLAGS) -o $@ $< \
		$(CORE_OBJECTS) $(FFMPEG_LIB) $(LDLIBS)

debug: CFLAGS := -O0 -g3 -std=c11 -Wall -Wextra -Wpedantic \
	-fsanitize=address,undefined
debug: LDFLAGS := -fsanitize=address,undefined
debug: clean $(TARGET)

clean:
	$(RM) $(TARGET) $(OBJECTS) analysis/ffmpeg_cycles \
		analysis/lane4_experiment
