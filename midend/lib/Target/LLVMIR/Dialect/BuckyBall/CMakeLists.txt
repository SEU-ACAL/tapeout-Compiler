add_mlir_translation_library(BuddyBuckyBallToLLVMIRTranslation
  BuckyBallToLLVMIRTranslation.cpp

  DEPENDS
  BuddyBuckyBallConversionsIncGen
  buddy_intrinsics_gen
  
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  BuddyBuckyBall
  )
