add_mlir_dialect_library(BuddyBuckyBallTransforms
  LegalizeForLLVMExport.cpp

  LINK_LIBS PUBLIC
  BuddyBuckyBall
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
