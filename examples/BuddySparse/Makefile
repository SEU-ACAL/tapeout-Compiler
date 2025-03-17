BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc
INTERFACES:= ../../frontend/Interfaces



spmm-lower:
	@${BUDDY_OPT} ./spmm.mlir \
		-lower-gemmini \
		-o log.mlir

spmm-translate:
	@${BUDDY_OPT} ./spmm.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

spmm-run:
	@${BUDDY_OPT} ./spmm.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out


matmul-vector-lower:
	@${BUDDY_OPT} ./subgraph0.mlir \
		-convert-linalg-to-gemmini \
		-lower-gemmini="vector=true" \
		-o matmul_vector.mlir

matmul-cisc-lower:
	@${BUDDY_OPT} ./subgraph0.mlir \
		-convert-linalg-to-gemmini \
		-lower-gemmini="vector=false" \
		-o matmul_cisc.mlir

matmul-default-lower:
	@${BUDDY_OPT} ./subgraph0.mlir \
		-convert-linalg-to-gemmini \
		-lower-gemmini \
		-o matmul_default.mlir