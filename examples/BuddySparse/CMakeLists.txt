set(BUDDY_EXAMPLES_DIR ${BUDDY_MLIR_DIR}/examples/)
set(BUDDY_BINARY_DIR ${BUDDY_MLIR_DIR}/build/bin/)
set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_DIR}/build/thirdparty/riscv-gnu-toolchain)
set(CMAKE_CXX_COMPILER  ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(BUDDY_SPARSE_DIR ${BUDDY_EXAMPLES_DIR}/BuddySparse)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces/)
set(INCLUDE_DIR ${BUDDY_SPARSE_DIR}/include/)


add_custom_command(
  OUTPUT  forward.mlir subgraph0.mlir
  COMMAND export BUDDYGEMMINI_EXAMPLE_PATH=${BUDDY_SPARSE_DIR} &&
          python3 ${BUDDY_SPARSE_DIR}/buddy-matmul-import.py
  DEPENDS buddy-matmul-import.py
  COMMENT "Generating forward.mlir and subgraph0.mlir"
)

add_custom_command(
  OUTPUT  matmul.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_SPARSE_DIR}/subgraph0.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" 
            > matmul_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_SPARSE_DIR}/subgraph0.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-tensor-to-linalg
            -linalg-bufferize
            -convert-linalg-to-gemmini
            -batchmatmul-optimize
            > matmul_loops.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_SPARSE_DIR}/subgraph0.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-tensor-to-linalg
            -linalg-bufferize
            -convert-linalg-to-gemmini
            -lower-gemmini=vector=true
            > matmul_gemmini_vector.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_SPARSE_DIR}/subgraph0.mlir 
           -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-tensor-to-linalg
            -linalg-bufferize
            -convert-linalg-to-gemmini
            -lower-gemmini
            > matmul_gemmini_default.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_SPARSE_DIR}/subgraph0.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -linalg-bufferize
            -tensor-bufferize
            -func-bufferize
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini 
            -arith-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts | 
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir
          > matmul.ll
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_SPARSE_DIR}/subgraph0.mlir 
          -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
        ${BUDDY_BINARY_DIR}/buddy-opt
          -eliminate-empty-tensors
          -linalg-bufferize
          -tensor-bufferize
          -func-bufferize
          -convert-linalg-to-gemmini
          -expand-strided-metadata
          -convert-linalg-to-loops
          -convert-scf-to-cf
          -llvm-request-c-wrappers
          -lower-gemmini 
          -arith-bufferize
          -buffer-deallocation
          -finalizing-bufferize
          -convert-arith-to-llvm
          -convert-func-to-llvm
          -finalize-memref-to-llvm
          -reconcile-unrealized-casts | 
        ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
        ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O0 -mattr=+buddyext,+D -float-abi=hard -o matmul.o
        DEPENDS subgraph0.mlir forward.mlir
  COMMENT "Building matmul.o"
  VERBATIM)

add_custom_target(matmul 
  DEPENDS matmul.o
  COMMENT "Building matmul.o"
  VERBATIM)

# add_library(GemminiMatmul STATIC matmul.o)
# set_target_properties(GemminiSPMM PROPERTIES LINKER_LANGUAGE CXX)

# add_executable(buddy-gemmini-spmm-run buddy-spmm-main.cpp)
# add_dependencies(buddy-gemmini-spmm-run GemminiSPMM)
# target_include_directories(buddy-gemmini-spmm-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
# target_link_libraries(buddy-gemmini-spmm-run -static GemminiSPMM)
