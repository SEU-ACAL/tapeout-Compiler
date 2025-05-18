add_mlir_dialect(BuckyBall buckyball)
add_mlir_doc(BuckyBall BuckyBall Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS BuckyBall.td)
mlir_tablegen(BuckyBallConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(BuddyBuckyBallConversionsIncGen)
