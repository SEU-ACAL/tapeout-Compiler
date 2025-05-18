//===- BuckyBallDialect.cpp - MLIR BuckyBall dialect implementation ----------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the BuckyBall dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

#include "BuckyBall/BuckyBallDialect.h"
#include "BuckyBall/BuckyBallOps.h"
using namespace mlir;
using namespace buddy::buckyball;

#include "BuckyBall/BuckyBallDialect.cpp.inc"

#define GET_OP_CLASSES
#include "BuckyBall/BuckyBall.cpp.inc"

void BuckyBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "BuckyBall/BuckyBall.cpp.inc"
      >();
}
