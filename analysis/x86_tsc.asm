; Serialized timestamp helpers for the cycle-analysis programs.

bits 64
default rel

section .text

global x86_tsc_start
align 16
x86_tsc_start:
    lfence
    rdtsc
    shl rdx, 32
    or  rax, rdx
    ret

global x86_tsc_end
align 16
x86_tsc_end:
    rdtscp
    lfence
    shl rdx, 32
    or  rax, rdx
    ret
