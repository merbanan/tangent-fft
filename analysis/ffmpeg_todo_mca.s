# Input for:
#   llvm-mca -mcpu=znver2 -iterations=100 analysis/ffmpeg_todo_mca.s
#
# The regions reproduce the instruction-level comparisons discussed in
# docs/ffmpeg-todo-investigation.md. They are throughput models, not functional
# transform kernels.

.intel_syntax noprefix

# LLVM-MCA-BEGIN vperm-duplicate
vperm2f128 ymm3, ymm3, ymm3, 0
# LLVM-MCA-END

# LLVM-MCA-BEGIN vinsert-duplicate
vinsertf128 ymm3, ymm3, xmm3, 1
# LLVM-MCA-END

# LLVM-MCA-BEGIN xor-add-rotation
vpermilps ymm6, ymm6, 0xb1
vxorps ymm7, ymm6, ymmword ptr [rip + masks]
vxorps ymm8, ymm6, ymmword ptr [rip + masks]
vaddps ymm5, ymm2, ymm7
vaddps ymm6, ymm2, ymm8
# LLVM-MCA-END

# LLVM-MCA-BEGIN xor-addsub-rotation
vpermilps ymm6, ymm6, 0xb1
vxorps ymm7, ymm6, ymmword ptr [rip + masks]
vaddsubps ymm5, ymm2, ymm7
vaddsubps ymm6, ymm2, ymm6
# LLVM-MCA-END

# LLVM-MCA-BEGIN shuffle
vshufps ymm4, ymm2, ymm3, 0xb1
# LLVM-MCA-END

# LLVM-MCA-BEGIN blend
vblendps ymm4, ymm2, ymm3, 0xcc
# LLVM-MCA-END

masks:
.zero 32
