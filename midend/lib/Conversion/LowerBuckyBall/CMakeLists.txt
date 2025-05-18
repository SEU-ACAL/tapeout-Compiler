add_mlir_library(LowerBuckyBallPass
  LowerBuckyBallPass.cpp

  LINK_LIBS PUBLIC
  BuddyBuckyBallTransforms
  )
