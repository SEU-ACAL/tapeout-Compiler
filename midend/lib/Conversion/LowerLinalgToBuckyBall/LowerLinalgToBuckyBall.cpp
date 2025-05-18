//====- LowerLinalgToBuckyBall.cpp - Linalg Dialect Lowering Pass -----------===//
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
// This file defines Linalg dialect lowering pass.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "BuckyBall/BuckyBallDialect.h"
#include "BuckyBall/BuckyBallOps.h"
using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class MatmulLowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  explicit MatmulLowering(MLIRContext *context, std::string accType)
      : OpRewritePattern(context), accType(accType) {}
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp matMulOp,
                                PatternRewriter &rewriter) const override {
    auto inputs = matMulOp.getInputs();
    auto ouputs = matMulOp.getOutputs();
    Location loc = matMulOp.getLoc();
    Value input0 = inputs[0];
    Value input1 = inputs[1];
    Value output0 = ouputs[0];
    MemRefType input0Type =  dyn_cast<MemRefType>(input0.getType());
    MemRefType biasType =
        MemRefType::get(input0Type.getShape(), rewriter.getI32Type());
    TypedAttr fillOpInputAttr = rewriter.getI32IntegerAttr(0);
    Type fillOpInsType = rewriter.getI32Type();
    if (accType == "f32") {
      biasType = MemRefType::get(input0Type.getShape(), rewriter.getF32Type());
      fillOpInputAttr = rewriter.getF32FloatAttr(0);
      fillOpInsType = rewriter.getF32Type();
    }
    llvm::APFloat scale1((float)1.0);
    llvm::APFloat scale0((float)0.0);
    Value bias = rewriter.create<memref::AllocOp>(loc, biasType);
    Value fillOpInputValue =
        rewriter.create<arith::ConstantOp>(loc, fillOpInsType, fillOpInputAttr);
    rewriter.create<linalg::FillOp>(loc, fillOpInputValue, bias);
    rewriter.replaceOpWithNewOp<buckyball::TileMatMulOp>(
        matMulOp, input0, input1, output0, bias, /*aScaleFactor = */ scale1,
        /*bScaleFactor = */ scale1, /*dScaleFactor = */ scale1, /*act = */ 0,
        /*accScale = */ scale1, /*bertScale = */ scale0);
    rewriter.create<memref::DeallocOp>(loc, bias);
    return success();
  }

private:
  std::string accType;
};

} // namespace

void populateLowerLinalgToBuckyBallConversionPatterns(RewritePatternSet &patterns,
                                                    std::string accType) {
  patterns.add<MatmulLowering>(patterns.getContext(), accType);
  patterns.add<Conv2DNchwFchwLowering>(patterns.getContext(), accType);
  patterns.add<Conv2DNhwcFhwcLowering>(patterns.getContext(), accType);
  patterns.add<Conv2DNhwcHwcfLowering>(patterns.getContext(), accType);
  patterns.add<BatchMatMulOpLowering>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// LowerLinalgToBuckyBall
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgToBuckyBallPass
    : public PassWrapper<LowerLinalgToBuckyBallPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToBuckyBallPass);
  LowerLinalgToBuckyBallPass() = default;
  LowerLinalgToBuckyBallPass(const LowerLinalgToBuckyBallPass &) {}
  StringRef getArgument() const final { return "convert-linalg-to-buckyball"; }
  StringRef getDescription() const final {
    return "convert linalg dialect to buckyball dialect";
  }
  void runOnOperation() override;
  Option<std::string> accType{*this, "acc_t",
                              llvm::cl::desc("The type of acc_t."),
                              llvm::cl::init("i32")};
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<buckyball::BuckyBallDialect, func::FuncDialect,
                    memref::MemRefDialect, linalg::LinalgDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }
};
} // namespace

void LowerLinalgToBuckyBallPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  ConversionTarget target(*context);
  target.addLegalDialect<memref::MemRefDialect, buckyball::BuckyBallDialect,
                         arith::ArithDialect, scf::SCFDialect>();
  target.addLegalOp<linalg::FillOp, linalg::YieldOp>();
  RewritePatternSet patterns(context);
  populateLowerLinalgToBuckyBallConversionPatterns(patterns, accType);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerLinalgToBuckyBallPass() {
  PassRegistration<LowerLinalgToBuckyBallPass>();
}
} // namespace buddy
} // namespace mlir
