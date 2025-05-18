add_mlir_dialect_library(BuddyBuckyBall
  BuckyBallDialect.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)
