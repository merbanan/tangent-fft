CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?= -lm -pthread

TARGET := fft_harness
OBJECTS := fft.o ffmpeg_fft.o harness.o
FFMPEG_LIB := .ffmpeg-build/libavutil/libavutil.a
NASM ?= nasm
HOST_ARCH := $(shell uname -m)

ifeq ($(HOST_ARCH),x86_64)
CPPFLAGS += -DHAVE_TANGENT_X86_ASM=1
OBJECTS += tangent_x86_kernel.o
else
CPPFLAGS += -DHAVE_TANGENT_X86_ASM=0
endif

CPPFLAGS += -I. -Ithird_party/ffmpeg -I.ffmpeg-build

.PHONY: all clean test bench debug ffmpeg ffmpeg-cycles

all: $(TARGET)

$(TARGET): $(OBJECTS) $(FFMPEG_LIB)
	$(CC) $(LDFLAGS) -o $@ $(OBJECTS) $(FFMPEG_LIB) $(LDLIBS)

fft.o: fft.c fft.h ffmpeg_fft.h tangent_x86_asm.h
ffmpeg_fft.o: ffmpeg_fft.c ffmpeg_fft.h fft.h third_party/ffmpeg/libavutil/tx.h
harness.o: harness.c fft.h

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

debug: CFLAGS := -O0 -g3 -std=c11 -Wall -Wextra -Wpedantic \
	-fsanitize=address,undefined
debug: LDFLAGS := -fsanitize=address,undefined
debug: clean $(TARGET)

clean:
	$(RM) $(TARGET) $(OBJECTS) analysis/ffmpeg_cycles
