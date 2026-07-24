.intel_syntax noprefix
.text

# Isolated dual-bank complex rotation with one shared vector coefficient.
# LLVM-MCA-BEGIN bank8_shared_root
        vmovaps         ymm8, ymmword ptr [r10]
        vmovaps         ymm9, ymmword ptr [r10 + 32]
        vpermilps       ymm10, ymm1, 0xb1
        vmulps          ymm10, ymm10, ymm9
        vfmaddsub132ps  ymm1, ymm10, ymm8
        vpermilps       ymm11, ymm5, 0xb1
        vmulps          ymm11, ymm11, ymm9
        vfmaddsub132ps  ymm5, ymm11, ymm8
# LLVM-MCA-END bank8_shared_root

# Same work with two memory-source products, as in two independent lane4
# butterflies.  It has fewer instructions but reads each coefficient twice.
# LLVM-MCA-BEGIN lane4_two_roots
        vpermilps       ymm10, ymm1, 0xb1
        vmulps          ymm10, ymm10, ymmword ptr [r10 + 32]
        vfmaddsub132ps  ymm1, ymm10, ymmword ptr [r10]
        vpermilps       ymm11, ymm5, 0xb1
        vmulps          ymm11, ymm11, ymmword ptr [r10 + 32]
        vfmaddsub132ps  ymm5, ymm11, ymmword ptr [r10]
# LLVM-MCA-END lane4_two_roots

# Rejected 24-byte coefficient stream.  Its instruction count matches the
# shared-vector form, but both coefficient loads are broadcast shuffles.
# LLVM-MCA-BEGIN bank8_broadcast_root
        vbroadcastss    ymm8, dword ptr [r10]
        vbroadcastss    ymm9, dword ptr [r10 + 4]
        vpermilps       ymm10, ymm1, 0xb1
        vmulps          ymm10, ymm10, ymm9
        vfmaddsub132ps  ymm1, ymm10, ymm8
        vpermilps       ymm11, ymm5, 0xb1
        vmulps          ymm11, ymm11, ymm9
        vfmaddsub132ps  ymm5, ymm11, ymm8
# LLVM-MCA-END bank8_broadcast_root

        ret
