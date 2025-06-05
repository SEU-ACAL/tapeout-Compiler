//===- LegalizeForLLVMExport.cpp - Prepare BuckyBall for LLVM translation ---===//
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

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "BuckyBall/BuckyBallDialect.h"
#include "BuckyBall/BuckyBallOps.h"
#include "BuckyBall/Transform.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace buddy::buckyball;

namespace {
int64_t getNumberFromValue(Value &value) {
  return dyn_cast<IntegerAttr>(value.getDefiningOp()->getAttr("value")).getInt();
} 
}// namespace

template <typename OpTy>
class ForwardOperands : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().getTypes() == op->getOperands().getTypes())
      return rewriter.notifyMatchFailure(op, "operand types already match");
    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

class ReturnOpTypeConversion : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Instruction-level Ops: Transform Xop to X_IntrOp
//===----------------------------------------------------------------------===//

struct BuckyBallFlushLowering : public ConvertOpToLLVMPattern<FlushOp> {
  using ConvertOpToLLVMPattern<FlushOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(FlushOp flushOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = flushOp.getLoc();
    Value skip = flushOp.getSkip();
    IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(0);
    Value rs2 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rs2Attr);
    rewriter.replaceOpWithNewOp<Flush_IntrOp>(flushOp, skip, rs2);
    return success();
  }
};

struct BuckyBallMvinLowering : public ConvertOpToLLVMPattern<MvinOp> {
  using ConvertOpToLLVMPattern<MvinOp>::ConvertOpToLLVMPattern;
  explicit BuckyBallMvinLowering(LLVMTypeConverter &typeConverter,
                               int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(MvinOp mvinOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = mvinOp.getInput();
    Location loc = input.getLoc();

    // get aMemAddr pointer
    MemRefType memRefType = dyn_cast<MemRefType>(mvinOp.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    
    // get aSpAddr
    Value spadAddrValue = mvinOp.getAddr();
    // get matrix's row and col
    // Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value row = rewriter.create<memref::DimOp>(loc, input, 0);
    Value col = rewriter.create<memref::DimOp>(loc, input, 1);
    
    // cast index to i64
    row = rewriter.create<arith::IndexCastOp>(loc, i64Type, row);
    col = rewriter.create<arith::IndexCastOp>(loc, i64Type, col);
    
    Value shift1 = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(addrLen));
    Value shift2 = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(2 * addrLen));
    
    row = rewriter.create<arith::ShLIOp>(loc, row, shift2);
    col = rewriter.create<arith::ShLIOp>(loc, col, shift1);

    // rs1 = indexCastOp
    // rs2 = col << (2 * addrLen) | row << addrLen | spadAddrValue
    Value rs1 = indexCastOp;
    Value rs2 = rewriter.create<arith::OrIOp>(loc, col, 
        rewriter.create<arith::OrIOp>(loc, row, spadAddrValue));
    
    rewriter.replaceOpWithNewOp<Mvin_IntrOp>(mvinOp, rs1, rs2);
    return success();
  }

private:
  int64_t addrLen;
};

struct BuckyBallMvin2Lowering : public ConvertOpToLLVMPattern<Mvin2Op> {
  using ConvertOpToLLVMPattern<Mvin2Op>::ConvertOpToLLVMPattern;
  explicit BuckyBallMvin2Lowering(LLVMTypeConverter &typeConverter,
                                int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(Mvin2Op mvin2Op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = mvin2Op.getInput();
    Location loc = input.getLoc();
    MemRefType memRefType = dyn_cast<MemRefType>(mvin2Op.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);

    Value spadAddrValue = mvin2Op.getAddr();
    Value row = rewriter.create<memref::DimOp>(loc, input, 0);
    Value col = rewriter.create<memref::DimOp>(loc, input, 1);
    
    row = rewriter.create<arith::IndexCastOp>(loc, i64Type, row);
    col = rewriter.create<arith::IndexCastOp>(loc, i64Type, col);
    
    Value shift1 = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(addrLen));
    Value shift2 = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(2 * addrLen));
    
    row = rewriter.create<arith::ShLIOp>(loc, row, shift2);
    col = rewriter.create<arith::ShLIOp>(loc, col, shift1);
    
    Value rs1 = indexCastOp;
    Value rs2 = rewriter.create<arith::OrIOp>(loc, col, 
        rewriter.create<arith::OrIOp>(loc, row, spadAddrValue));
    
    rewriter.replaceOpWithNewOp<Mvin2_IntrOp>(mvin2Op, rs1, rs2);
    return success();
  }

private:
  int64_t addrLen;
};

struct BuckyBallMvin3Lowering : public ConvertOpToLLVMPattern<Mvin3Op> {
  using ConvertOpToLLVMPattern<Mvin3Op>::ConvertOpToLLVMPattern;
  explicit BuckyBallMvin3Lowering(LLVMTypeConverter &typeConverter,
                                int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(Mvin3Op mvin3Op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = mvin3Op.getInput();
    Location loc = input.getLoc();
    MemRefType memRefType = dyn_cast<MemRefType>(mvin3Op.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);

    Value spadAddrValue = mvin3Op.getAddr();
    Value row = rewriter.create<memref::DimOp>(loc, input, 0);
    Value col = rewriter.create<memref::DimOp>(loc, input, 1);
    
    row = rewriter.create<arith::IndexCastOp>(loc, i64Type, row);
    col = rewriter.create<arith::IndexCastOp>(loc, i64Type, col);
    
    Value shift1 = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(addrLen));
    Value shift2 = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(2 * addrLen));
    
    row = rewriter.create<arith::ShLIOp>(loc, row, shift2);
    col = rewriter.create<arith::ShLIOp>(loc, col, shift1);
    
    Value rs1 = indexCastOp;
    Value rs2 = rewriter.create<arith::OrIOp>(loc, col, 
        rewriter.create<arith::OrIOp>(loc, row, spadAddrValue));

    rewriter.replaceOpWithNewOp<Mvin3_IntrOp>(mvin3Op, rs1, rs2);
    return success();
  }

private:
  int64_t addrLen;
};

struct BuckyBallMvoutLowering : public ConvertOpToLLVMPattern<MvoutOp> {
  using ConvertOpToLLVMPattern<MvoutOp>::ConvertOpToLLVMPattern;
  explicit BuckyBallMvoutLowering(LLVMTypeConverter &typeConverter,
                                int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(MvoutOp mvoutOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value output = mvoutOp.getOutput();
    Location loc = output.getLoc();
    MemRefType memRefType = dyn_cast<MemRefType>(mvoutOp.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, output);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);

    Value spadAddrValue = mvoutOp.getAddr();
    Value row = rewriter.create<memref::DimOp>(loc, output, 0);
    Value col = rewriter.create<memref::DimOp>(loc, output, 1);
    row = rewriter.create<arith::IndexCastOp>(loc, i64Type, row);
    col = rewriter.create<arith::IndexCastOp>(loc, i64Type, col);
    
    Value shift1 = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(addrLen));
    Value shift2 = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(2 * addrLen));
    row = rewriter.create<arith::ShLIOp>(loc, row, shift2);
    col = rewriter.create<arith::ShLIOp>(loc, col, shift1);
    
    // rs1 = indexCastOp
    // rs2 = col << (2 * addrLen) | row << addrLen | spadAddrValue
    Value rs1 = indexCastOp;
    Value rs2 = rewriter.create<arith::OrIOp>(loc, col, 
        rewriter.create<arith::OrIOp>(loc, row, spadAddrValue));
    rewriter.replaceOpWithNewOp<Mvout_IntrOp>(mvoutOp, rs1, rs2);
    return success();
  }

private:
  int64_t addrLen;
};

//===----------------------------------------------------------------------===//
// Warp-level Ops
//===----------------------------------------------------------------------===//
struct BuckyBallVecMulWarp16Lowering : public ConvertOpToLLVMPattern<VecMulWarp16Op> {
  using ConvertOpToLLVMPattern<VecMulWarp16Op>::ConvertOpToLLVMPattern;
  explicit BuckyBallVecMulWarp16Lowering(LLVMTypeConverter &typeConverter,
                                  int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(VecMulWarp16Op vecMulWarp16Op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = vecMulWarp16Op.getLoc();
    Value aSpAddr = vecMulWarp16Op.getASpAddr();
    Value bSpAddr = vecMulWarp16Op.getBSpAddr();
    Value cSpAddr = vecMulWarp16Op.getCSpAddr();
    Value nLen = vecMulWarp16Op.getNLen();
    // aSpAddr << addrLen
    Value shift1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(addrLen));
    aSpAddr = rewriter.create<arith::ShLIOp>(loc, aSpAddr, shift1);
    // nLen << (2 * addrLen)
    Value shift2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(2 * addrLen));    
    nLen = rewriter.create<arith::ShLIOp>(loc, nLen, shift2);
    // rs1 = aSpAddr << addrLen | bSpAddr
    // rs2 = nLen << addrLen | cSpAddr
    Value rs1 = rewriter.create<arith::OrIOp>(loc, aSpAddr, bSpAddr);
    Value rs2 = rewriter.create<arith::OrIOp>(loc, nLen, cSpAddr);
    rewriter.replaceOpWithNewOp<VecMulWarp16_IntrOp>(vecMulWarp16Op, rs1, rs2);
    return success();
  }
private:
  int64_t addrLen;
};

//===----------------------------------------------------------------------===//
// Meta-Tile Ops (Tile-inside)
//===----------------------------------------------------------------------===//
class BuckyBallMetaTileMatMulLowering : public ConvertOpToLLVMPattern<MetaTileMatMulOp> {
public:
  using ConvertOpToLLVMPattern<MetaTileMatMulOp>::ConvertOpToLLVMPattern;
  explicit BuckyBallMetaTileMatMulLowering(LLVMTypeConverter &typeConverter, 
                                          int64_t lane)
      : ConvertOpToLLVMPattern(typeConverter), lane(lane) {}
  LogicalResult
  matchAndRewrite(MetaTileMatMulOp metaTileMatMulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = metaTileMatMulOp.getLoc();
    
    Value aMemArray = metaTileMatMulOp.getAMemArray();
    Value aMetaTileArray = metaTileMatMulOp.getAMetaTileArray();
    // A, B, C Matrix's base address
    Value aSpAddrStart = metaTileMatMulOp.getASpAddrStart();
    Value bSpAddrStart = metaTileMatMulOp.getBSpAddrStart();
    Value cSpAddrStart = metaTileMatMulOp.getCSpAddrStart();
    Value metaMLen = metaTileMatMulOp.getMetaMLen();
    Value metaNLen = metaTileMatMulOp.getMetaNLen();
    Value metaMNum = metaTileMatMulOp.getMetaMNum();
    Value metaNNum = metaTileMatMulOp.getMetaNNum();
    Value laneVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(lane));    
        
    // <mvinA>
    rewriter.create<MvinOp>(loc, aMetaTileArray, aSpAddrStart);
    
    // M loop over metaMNum
    // upperBound is dynamic value, so we need to cast it to index type
    Operation *loopOp = nullptr;
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), metaMNum);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto mLoop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
    loopOp = mLoop.getOperation();
    rewriter.setInsertionPointToStart(mLoop.getBody()); 
    Value mLoopVal = mLoop.getInductionVar();
    // aSpAddr = aSpAddrStart + loopIdx * lane
    Value mLoopIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), mLoopVal);
    Value aOffset = rewriter.create<arith::MulIOp>(loc, mLoopIdx, laneVal);
    Value aSpAddr = rewriter.create<arith::AddIOp>(loc, aSpAddrStart, aOffset);
    // bSpAddr = bSpAddrStart (not changed)
    Value bSpAddr = bSpAddrStart;
    // cSpAddr = cSpAddrStart + loopIdx * lane
    Value cOffset = rewriter.create<arith::MulIOp>(loc, mLoopIdx, laneVal);
    Value cSpAddr = rewriter.create<arith::AddIOp>(loc, cSpAddrStart, cOffset);
    // nLen = metaNLen * lane
    Value nLen = rewriter.create<arith::MulIOp>(loc, metaNLen, laneVal);
    rewriter.create<VecMulWarp16Op>(loc, aSpAddr, bSpAddr, cSpAddr, nLen);
    rewriter.setInsertionPointAfter(loopOp);

    rewriter.eraseOp(metaTileMatMulOp);
    return success();
  }
private:
  int64_t lane;
};

//===----------------------------------------------------------------------===//
// Merge-Tile Ops
//===----------------------------------------------------------------------===//
class BuckyBallMergeTileMatMulLowering : public ConvertOpToLLVMPattern<MergeTileMatMulOp> {
public:
  using ConvertOpToLLVMPattern<MergeTileMatMulOp>::ConvertOpToLLVMPattern;
  explicit BuckyBallMergeTileMatMulLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern(typeConverter) {}

  LogicalResult
  matchAndRewrite(MergeTileMatMulOp mergeTileMatMulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {    
    Location loc = mergeTileMatMulOp.getLoc();

    Value aMergeTileArray = mergeTileMatMulOp.getAMergeTileArray();
    Value aMemArray = mergeTileMatMulOp.getAMemArray();
    Value bMergeTileArray = mergeTileMatMulOp.getBMergeTileArray();
    Value bMemArray = mergeTileMatMulOp.getBMemArray();
    Value aSpAddrStart = mergeTileMatMulOp.getASpAddrStart();
    Value bSpAddrStart = mergeTileMatMulOp.getBSpAddrStart();
    Value cSpAddrStart = mergeTileMatMulOp.getCSpAddrStart();
    Value metaMNum = mergeTileMatMulOp.getMetaMNum();
    Value metaNNum = mergeTileMatMulOp.getMetaNNum();
    Value metaKNum = mergeTileMatMulOp.getMetaKNum();
    Value metaMLen = mergeTileMatMulOp.getMetaMLen();
    Value metaNLen = mergeTileMatMulOp.getMetaNLen();
    Value metaKLen = mergeTileMatMulOp.getMetaKLen();

    Value M = rewriter.create<memref::DimOp>(loc, aMemArray, 0);

    // MvinB 
    rewriter.create<MvinOp>(loc, bMergeTileArray, bSpAddrStart);

    // K loop
    Operation *loopOp = nullptr;
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), metaKNum);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);    
    auto kLoop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
    loopOp = kLoop.getOperation();
    rewriter.setInsertionPointToStart(kLoop.getBody());
    Value kLoopIdx = kLoop.getInductionVar();
    Value kLoopI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), kLoopIdx);
    Value M_i64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), M);
    // aSpAddr = aSpAddrStart + kIdx * M
    Value aSpAddrOffset = rewriter.create<arith::MulIOp>(loc, kLoopI64, M_i64);
    aSpAddrStart = rewriter.create<arith::AddIOp>(loc, aSpAddrStart, aSpAddrOffset);
    // bSpAddr stays the same 
    Value bSpAddr = bSpAddrStart;
    // cSpAddr stays the same 
    Value cSpAddr = cSpAddrStart;
    // Create subview for A matrix slice in K dimension
    Value metaKLenIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), metaKLen);
    Value metaTileKStart = rewriter.create<arith::MulIOp>(loc, kLoopIdx, metaKLenIdx); 
    // the input of subview must be index type, not i64 type
    Value aMetaTileArray = rewriter.create<memref::SubViewOp>(
      loc, aMemArray,
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(0), metaTileKStart}, // start
      SmallVector<OpFoldResult>{M, metaKLenIdx}, // size 
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)} // stride
    ); 
    rewriter.create<MetaTileMatMulOp>(loc, aMetaTileArray, aMemArray,
                                      aSpAddrStart, bSpAddrStart, cSpAddrStart,
                                      metaMNum, metaNNum, metaMLen, metaNLen);
    rewriter.setInsertionPointAfter(loopOp);
    
    rewriter.eraseOp(mergeTileMatMulOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tile division Ops
//===----------------------------------------------------------------------===//
class BuckyBallVecTileMatMulLowering : public ConvertOpToLLVMPattern<VecTileMatMulOp> {
  // compute the up limit of merge-tile's size
  // mergeTileSpadRows = 
  //    (M_mergeTileLen * M_metaTileLen) * (K_mergeTileLen * K_metaTileLen) + 
  //    (K_mergeTileLen * K_metaTileLen) * (N_mergeTileLen * N_metaTileLen)
  Value mergeTileSpadRows(ConversionPatternRewriter &rewriter, Location loc,
                         Value mMergeTileLen, Value nMergeTileLen, Value kMergeTileLen, 
                         Value mMetaTileLen, Value nMetaTileLen, Value kMetaTileLen) const {
    Value aMatrixRows = rewriter.create<arith::MulIOp>(loc,
        rewriter.create<arith::MulIOp>(loc, mMergeTileLen, mMetaTileLen),
        rewriter.create<arith::MulIOp>(loc, kMergeTileLen, kMetaTileLen));
    Value bMatrixRows = rewriter.create<arith::MulIOp>(loc,
        rewriter.create<arith::MulIOp>(loc, kMergeTileLen, kMetaTileLen),
        rewriter.create<arith::MulIOp>(loc, nMergeTileLen, nMetaTileLen));
    return rewriter.create<arith::AddIOp>(loc, aMatrixRows, bMatrixRows);
  }

  // mergeTileAccRows = M_mergeTileLen * M_metaTileLen * N_mergeTileLen * N_metaTileLen
  Value mergeTileAccRows(ConversionPatternRewriter &rewriter, Location loc,
                        Value mMergeTileLen, Value nMergeTileLen,
                        Value mMetaTileLen, Value nMetaTileLen) const {
    return rewriter.create<arith::MulIOp>(loc,
        rewriter.create<arith::MulIOp>(loc, mMergeTileLen, mMetaTileLen),
        rewriter.create<arith::MulIOp>(loc, nMergeTileLen, nMetaTileLen));
  }

public:
  using ConvertOpToLLVMPattern<VecTileMatMulOp>::ConvertOpToLLVMPattern;
  explicit BuckyBallVecTileMatMulLowering(LLVMTypeConverter &typeConverter, 
                                          int64_t dim, int64_t addrLen, 
                                          int64_t spadRows, int64_t accRows,
                                          size_t sizeOfElemT, size_t sizeOfAccT,
                                          int64_t lane, int64_t warp)
      : ConvertOpToLLVMPattern(typeConverter), dim(dim), addrLen(addrLen),
        spadRows(spadRows), accRows(accRows), sizeOfElemT(sizeOfElemT), sizeOfAccT(sizeOfAccT), lane(lane), warp(warp) {}

  LogicalResult
  matchAndRewrite(VecTileMatMulOp vecTileMatMulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
  // Convert aArray, bArray, cArray, dArray to ArrayindexCast 
  Value aMemArray = vecTileMatMulOp.getAMemArray();
  Value bMemArray = vecTileMatMulOp.getBMemArray();
  Value cMemArray = vecTileMatMulOp.getCMemArray();
  // Value dArray = vecTileMatMulOp.getDArray();
  MemRefType aMemArrayType = dyn_cast<MemRefType>(aMemArray.getType());
  MemRefType bMemArrayType = dyn_cast<MemRefType>(bMemArray.getType());
  MemRefType cMemArrayType = dyn_cast<MemRefType>(cMemArray.getType());
  // MemRefType dArrayType = dyn_cast<MemRefType>(dArray.getType());
  StridedLayoutAttr aMemArrayLayout = dyn_cast<StridedLayoutAttr>(aMemArrayType.getLayout());
  StridedLayoutAttr bMemArrayLayout = dyn_cast<StridedLayoutAttr>(bMemArrayType.getLayout());
  StridedLayoutAttr cMemArrayLayout = dyn_cast<StridedLayoutAttr>(cMemArrayType.getLayout());
  SmallVector<Type> resultType = {rewriter.getIndexType()};
  TypeRange typeRange(resultType);
  Location loc = vecTileMatMulOp.getLoc();
  IntegerType i64Type = rewriter.getI64Type();

  Value aMemArrayExtractOp =
      rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange, aMemArray);
  if (aMemArrayLayout) {
    Value offset = rewriter.create<arith::ConstantIndexOp>(
        loc, aMemArrayLayout.getOffset() * sizeOfElemT);
    aMemArrayExtractOp =
        rewriter.create<arith::AddIOp>(loc, aMemArrayExtractOp, offset);
  }
  Value aMemArrayindexCastOp =
      rewriter.create<arith::IndexCastOp>(loc, i64Type, aMemArrayExtractOp);
  
  Value bMemArrayExtractOp =
      rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange, bMemArray);
  if (bMemArrayLayout) {
    Value offset = rewriter.create<arith::ConstantIndexOp>(
        loc, bMemArrayLayout.getOffset() * sizeOfElemT);
    bMemArrayExtractOp =
        rewriter.create<arith::AddIOp>(loc, bMemArrayExtractOp, offset);
  }
  Value bMemArrayindexCastOp =
      rewriter.create<arith::IndexCastOp>(loc, i64Type, bMemArrayExtractOp);

  Value cMemArrayExtractOp =
      rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange, cMemArray);
  if (cMemArrayLayout) {
    Value offset = rewriter.create<arith::ConstantIndexOp>(
        loc, cMemArrayLayout.getOffset() * sizeOfElemT);
    cMemArrayExtractOp =
        rewriter.create<arith::AddIOp>(loc, cMemArrayExtractOp, offset);
  }
  Value cMemArrayindexCastOp =
      rewriter.create<arith::IndexCastOp>(loc, i64Type, cMemArrayExtractOp);
  
  // Value dArrayExtractOp =
  //     rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange, dArray);
  // Value dArrayindexCastOp =
  //     rewriter.create<arith::IndexCastOp>(loc, i64Type, dArrayExtractOp);

  // Get A, B, C Matrix's shape, A[M][K], B[K][N], C[M][N]
  Value M = rewriter.create<memref::DimOp>(loc, aMemArray, 0);
  Value K = rewriter.create<memref::DimOp>(loc, aMemArray, 1);  
  Value N = rewriter.create<memref::DimOp>(loc, bMemArray, 1);  

  // assert(K < 1024 && "K must be less than 1024");
  
  // M,N,K dimension's meta-tile's length
  Value mMetaTileLen = rewriter.create<arith::ConstantIndexOp>(loc, lane);
  Value nMetaTileLen = rewriter.create<arith::ConstantIndexOp>(loc, lane);
  Value kMetaTileLen = rewriter.create<arith::ConstantIndexOp>(loc, warp);

  // M_Padded = ((M + mMetaTileLen - 1) / mMetaTileLen) * mMetaTileLen
  Value mPadded = rewriter.create<arith::MulIOp>(loc,
      rewriter.create<arith::DivUIOp>(loc,
          rewriter.create<arith::AddIOp>(loc, M,
              rewriter.create<arith::SubIOp>(loc, mMetaTileLen,
                  rewriter.create<arith::ConstantIndexOp>(loc, 1))),
          mMetaTileLen),
      mMetaTileLen);
  
  // N_Padded = ((N + nMetaTileLen - 1) / nMetaTileLen) * nMetaTileLen
  Value nPadded = rewriter.create<arith::MulIOp>(loc,
      rewriter.create<arith::DivUIOp>(loc,
          rewriter.create<arith::AddIOp>(loc, N,
              rewriter.create<arith::SubIOp>(loc, nMetaTileLen, 
                  rewriter.create<arith::ConstantIndexOp>(loc, 1))),
          nMetaTileLen),
      nMetaTileLen);
      
  // K_Padded = ((K + kMetaTileLen - 1) / kMetaTileLen) * kMetaTileLen
  Value kPadded = rewriter.create<arith::MulIOp>(loc,
      rewriter.create<arith::DivUIOp>(loc,
          rewriter.create<arith::AddIOp>(loc, K,
              rewriter.create<arith::SubIOp>(loc, kMetaTileLen, 
                  rewriter.create<arith::ConstantIndexOp>(loc, 1))),
          kMetaTileLen),
      kMetaTileLen);

  // compute how many meta-tile
  // mTileNum = (mPadded + mMetaTileLen - 1) / mMetaTileLen
  Value mTileNum = rewriter.create<arith::DivUIOp>(loc, 
      rewriter.create<arith::AddIOp>(loc, mPadded, 
          rewriter.create<arith::SubIOp>(loc, mMetaTileLen, 
          rewriter.create<arith::ConstantIndexOp>(loc, 1))),
      mMetaTileLen);
  // nTileNum = (nPadded + nMetaTileLen - 1) / nMetaTileLen
  Value nTileNum = rewriter.create<arith::DivUIOp>(loc,
      rewriter.create<arith::AddIOp>(loc, nPadded,
          rewriter.create<arith::SubIOp>(loc, nMetaTileLen, 
              rewriter.create<arith::ConstantIndexOp>(loc, 1))),
      nMetaTileLen);
  // kTileNum = (kPadded + kMetaTileLen - 1) / kMetaTileLen
  Value kTileNum = rewriter.create<arith::DivUIOp>(loc,
      rewriter.create<arith::AddIOp>(loc, kPadded,
          rewriter.create<arith::SubIOp>(loc, kMetaTileLen, 
              rewriter.create<arith::ConstantIndexOp>(loc, 1))),
      kMetaTileLen);

  // Last tile lengths
  // mLastTileLen = (mPadded % mMetaTileLen == 0) ? mMetaTileLen : mPadded % mMetaTileLen
  /*
  Value mMod = rewriter.create<arith::RemUIOp>(loc, mPadded, mMetaTileLen);
  Value mModIsZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, 
      mMod, rewriter.create<arith::ConstantIndexOp>(loc, 0));
  Value mLastTileLen = rewriter.create<arith::SelectOp>(loc, mModIsZero, 
      mMetaTileLen, mMod);
  // nLastTileLen = (nPadded % nMetaTileLen == 0) ? nMetaTileLen : nPadded % nMetaTileLen
  Value nMod = rewriter.create<arith::RemUIOp>(loc, nPadded, nMetaTileLen);
  Value nModIsZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
      nMod, rewriter.create<arith::ConstantIndexOp>(loc, 0));
  Value nLastTileLen = rewriter.create<arith::SelectOp>(loc, nModIsZero,
      nMetaTileLen, nMod);
  // kLastTileLen = (kPadded % kMetaTileLen == 0) ? kMetaTileLen : kPadded % kMetaTileLen
  Value kMod = rewriter.create<arith::RemUIOp>(loc, kPadded, kMetaTileLen);
  Value kModIsZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
      kMod, rewriter.create<arith::ConstantIndexOp>(loc, 0));
  Value kLastTileLen = rewriter.create<arith::SelectOp>(loc, kModIsZero,
      kMetaTileLen, kMod);
  */

  // merge meta-tile to Merge-tile 
  // according to accumulator's size, fill it as much as possible
  // MergeTileLen: how many meta-tile in each dimension in Merge-tile
  // spadRows / 2: remain half of spadRows for double buffer (we do double buffer in default)
  Value maxSpadRows = rewriter.create<arith::DivUIOp>(loc, 
      rewriter.create<arith::ConstantIndexOp>(loc, spadRows),
      rewriter.create<arith::ConstantIndexOp>(loc, 2));
  Value maxAccRows = rewriter.create<arith::ConstantIndexOp>(loc, accRows);
  
  // Initialize merge tile lengths
  Value mMergeTileLen = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value nMergeTileLen = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value kMergeTileLen = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // extend merge tile length in N dimension
  auto Loop1 = rewriter.create<scf::WhileOp>(
      loc, TypeRange{nMergeTileLen.getType()}, ValueRange{nMergeTileLen},
      [&](OpBuilder &builder, Location loc, ValueRange args) {
          // n++
          Value currentN = args[0];
          Value nextN = builder.create<arith::AddIOp>(loc, currentN, 
              builder.create<arith::ConstantIndexOp>(loc, 1));

          Value spadRows = mergeTileSpadRows(rewriter, loc, 
              mMergeTileLen, nextN, kMergeTileLen,
              mMetaTileLen, nMetaTileLen, kMetaTileLen);
          Value accRows = mergeTileAccRows(rewriter, loc,
              mMergeTileLen, nextN, mMetaTileLen, nMetaTileLen);
          // mergeTileSpadRows < upperBound
          Value spadUpperBound = builder.create<arith::CmpIOp>(loc, 
              arith::CmpIPredicate::ule, spadRows, maxSpadRows);
          // mergeTileAccRows < upperBound
          Value accUpperBound = builder.create<arith::CmpIOp>(loc, 
              arith::CmpIPredicate::ule, accRows, maxAccRows);
          // (nextN + 1) * nMetaTileLen <= nPadded
          Value sizeUpperBound = builder.create<arith::CmpIOp>(loc, 
              arith::CmpIPredicate::ule,
              builder.create<arith::MulIOp>(loc, nextN, nMetaTileLen),
              nPadded);
          
          Value Increase = builder.create<arith::AndIOp>(loc,
              builder.create<arith::AndIOp>(loc, spadUpperBound, accUpperBound),
              sizeUpperBound);
          
          builder.create<scf::ConditionOp>(loc, Increase, ValueRange{nextN});
      },
      [&](OpBuilder &afterBuilder, Location loc, ValueRange args) {
          Value newN = args[0];
          afterBuilder.create<scf::YieldOp>(loc, ValueRange{newN});
      });
  nMergeTileLen = Loop1.getResult(0);
  
  // extend merge tile length in M dimension
  auto Loop2 = rewriter.create<scf::WhileOp>(
      loc, TypeRange{mMergeTileLen.getType()}, ValueRange{mMergeTileLen},
      [&](OpBuilder &builder, Location loc, ValueRange args) {
          // m++
          Value currentM = args[0];
          Value nextM = builder.create<arith::AddIOp>(loc, currentM, 
              builder.create<arith::ConstantIndexOp>(loc, 1));
          
          Value spadRows = mergeTileSpadRows(rewriter, loc, 
              nextM, nMergeTileLen, kMergeTileLen,
              mMetaTileLen, nMetaTileLen, kMetaTileLen);
          Value accRows = mergeTileAccRows(rewriter, loc,
              nextM, nMergeTileLen, mMetaTileLen, nMetaTileLen);
          // mergeTileSpadRows < upperBound
          Value spadUpperBound = builder.create<arith::CmpIOp>(loc, 
              arith::CmpIPredicate::ule, spadRows, maxSpadRows);
          // mergeTileAccRows < upperBound
          Value accUpperBound = builder.create<arith::CmpIOp>(loc, 
              arith::CmpIPredicate::ule, accRows, maxAccRows);
          // (nextM + 1) * mMetaTileLen <= mPadded
          Value sizeUpperBound = builder.create<arith::CmpIOp>(loc, 
              arith::CmpIPredicate::ule,
              builder.create<arith::MulIOp>(loc, nextM, mMetaTileLen),
              mPadded);
          
          Value Increase = builder.create<arith::AndIOp>(loc,
              builder.create<arith::AndIOp>(loc, spadUpperBound, accUpperBound),
              sizeUpperBound);
          
          builder.create<scf::ConditionOp>(loc, Increase, ValueRange{nextM});
      },
      [&](OpBuilder &afterBuilder, Location loc, ValueRange args) {
          Value newM = args[0];
          afterBuilder.create<scf::YieldOp>(loc, ValueRange{newM});
      });
  mMergeTileLen = Loop2.getResult(0);

  // compute how many merge-tile 
  Value mMergeTileSize = rewriter.create<arith::MulIOp>(loc, mMergeTileLen, mMetaTileLen);
  Value nMergeTileSize = rewriter.create<arith::MulIOp>(loc, nMergeTileLen, nMetaTileLen);
  Value kMergeTileSize = rewriter.create<arith::MulIOp>(loc, kMergeTileLen, kMetaTileLen);
  // mMergeTileNum = (mPadded + mMergeTileSize - 1) / mMergeTileSize
  Value mMergeTileNum = rewriter.create<arith::DivUIOp>(loc,
      rewriter.create<arith::AddIOp>(loc, mPadded,
          rewriter.create<arith::SubIOp>(loc, mMergeTileSize, 
              rewriter.create<arith::ConstantIndexOp>(loc, 1))),
      mMergeTileSize);
  // nMergeTileNum = (nPadded + nMergeTileSize - 1) / nMergeTileSize
  Value nMergeTileNum = rewriter.create<arith::DivUIOp>(loc,
      rewriter.create<arith::AddIOp>(loc, nPadded,
          rewriter.create<arith::SubIOp>(loc, nMergeTileSize, 
              rewriter.create<arith::ConstantIndexOp>(loc, 1))),
      nMergeTileSize);
  // kMergeTileNum = (kPadded + kMergeTileSize - 1) / kMergeTileSize
  Value kMergeTileNum = rewriter.create<arith::DivUIOp>(loc,
      rewriter.create<arith::AddIOp>(loc, kPadded,
          rewriter.create<arith::SubIOp>(loc, kMergeTileSize, 
              rewriter.create<arith::ConstantIndexOp>(loc, 1))),
      kMergeTileSize);

  // Calculate last merge tile length
  // mLastMergeTileLen = (mPadded % mMergeTileSize == 0) ? mMergeTileSize : mPadded % mMergeTileSize
  Value mlastRemaining = rewriter.create<arith::RemUIOp>(loc, mPadded, mMergeTileSize);
  Value mlastIsZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
      mlastRemaining, rewriter.create<arith::ConstantIndexOp>(loc, 0));
  Value mLastMergeTileLen = rewriter.create<arith::SelectOp>(loc, mlastIsZero,
      mMergeTileLen,
      rewriter.create<arith::DivUIOp>(loc,
          rewriter.create<arith::AddIOp>(loc, mlastRemaining,
              rewriter.create<arith::SubIOp>(loc, mMetaTileLen, 
                  rewriter.create<arith::ConstantIndexOp>(loc, 1))),
          mMetaTileLen));
  
  // nLastMergeTileLen = (nPadded % nMergeTileSize == 0) ? nMergeTileSize : nPadded % nMergeTileSize
  Value nlastRemaining = rewriter.create<arith::RemUIOp>(loc, nPadded, nMergeTileSize);
  Value nlastIsZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
      nlastRemaining, rewriter.create<arith::ConstantIndexOp>(loc, 0));
  Value nLastMergeTileLen = rewriter.create<arith::SelectOp>(loc, nlastIsZero,
      nMergeTileLen,
      rewriter.create<arith::DivUIOp>(loc,
          rewriter.create<arith::AddIOp>(loc, nlastRemaining,
              rewriter.create<arith::SubIOp>(loc, nMetaTileLen, 
                  rewriter.create<arith::ConstantIndexOp>(loc, 1))),
          nMetaTileLen));
  
  // kLastMergeTileLen = (kPadded % kMergeTileSize == 0) ? kMergeTileSize : kPadded % kMergeTileSize
  Value klastRemaining = rewriter.create<arith::RemUIOp>(loc, kPadded, kMergeTileSize);
  Value klastIsZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
      klastRemaining, rewriter.create<arith::ConstantIndexOp>(loc, 0));
  Value kLastMergeTileLen = rewriter.create<arith::SelectOp>(loc, klastIsZero,
      kMergeTileLen,
      rewriter.create<arith::DivUIOp>(loc,
          rewriter.create<arith::AddIOp>(loc, klastRemaining,
              rewriter.create<arith::SubIOp>(loc, kMetaTileLen, 
                  rewriter.create<arith::ConstantIndexOp>(loc, 1))),
          kMetaTileLen));
  
  // target scratchpad/accumulator addr
  // dim just means element numbers in each scratchpad line 
  // 00=A, 01=B, 10=D, 11=C
  // this is a kind of logic mapping, not a physical mapping
  // scratchpad addr and accumulator addr may not continue in logical mapping,
  // this is just keep them easy to be indexed
  // aSpAddrStart = 0
  Value aSpAddrStart = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  aSpAddrStart = rewriter.create<arith::IndexCastOp>(loc, i64Type, aSpAddrStart);

  // bSpAddrStart = spadRows - (dimN_MergeTileLen * dimN_MetaTileLen) * (dimN_MergeTileLen * dimN_MetaTileLen) / dim;
  Value bSpAddrStart = rewriter.create<arith::SubIOp>(loc, 
      rewriter.create<arith::ConstantIndexOp>(loc, spadRows), 
      rewriter.create<arith::DivUIOp>(loc, 
          rewriter.create<arith::MulIOp>(loc, 
              rewriter.create<arith::MulIOp>(loc, nMergeTileLen, nMetaTileLen),
              rewriter.create<arith::MulIOp>(loc, nMergeTileLen, nMetaTileLen)),
          rewriter.create<arith::ConstantIndexOp>(loc, dim)));
  bSpAddrStart = rewriter.create<arith::IndexCastOp>(loc, i64Type, bSpAddrStart);
  
  // cSpAddrStart = 1 << (addrLen - 1);
  Value cSpAddrStart = rewriter.create<arith::ShLIOp>(loc, 
      rewriter.create<arith::ConstantIndexOp>(loc, 1), 
      rewriter.create<arith::SubIOp>(loc, 
          rewriter.create<arith::ConstantIndexOp>(loc, addrLen), 
          rewriter.create<arith::ConstantIndexOp>(loc, 1)));
  cSpAddrStart = rewriter.create<arith::IndexCastOp>(loc, i64Type, cSpAddrStart);
  
  // divide into meta-tile to caculate
  Operation *kloopOp = nullptr;
  Operation *mloopOp = nullptr;
  SmallVector<Value, 3> loopIvs;
  Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  scf::ForOp kLoop, mLoop, nLoop;  
  for (size_t i = 0; i < 3; i++) {
    Value upperBound;
    switch (i) {
      // OutSide MergeTile Loop
      // for (int i = 0; i < K; i++)
      //   for (int j = 0; j < M; j++)
      //     for (int k = 0; k < N; k++)
      //      <MergeTileMatMulOp>
      // <mvoutC>
      case 2: 
        upperBound = kMergeTileNum;
        // upperBound = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), kMergeTileNum);
        kLoop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
        kloopOp = kLoop.getOperation();
        rewriter.setInsertionPointToStart(kLoop.getBody());
        loopIvs.push_back(kLoop.getInductionVar());
        break;
      case 1: 
        upperBound = mMergeTileNum;
        // upperBound = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), mMergeTileNum);
        mLoop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
        mloopOp = mLoop.getOperation();
        rewriter.setInsertionPointToStart(mLoop.getBody());
        loopIvs.push_back(mLoop.getInductionVar());
        break;
      case 0: 
        upperBound = nMergeTileNum;
        // upperBound = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), nMergeTileNum);
        nLoop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
        rewriter.setInsertionPointToStart(nLoop.getBody());
        loopIvs.push_back(nLoop.getInductionVar());
        break;
    }
  }
  Value mergeNIdx = loopIvs[0];
  Value mergeMIdx = loopIvs[1];
  Value mergeKIdx = loopIvs[2];

  // Check if current tile is the last tile in each dimension
  Value isLastM = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, 
      mergeMIdx, rewriter.create<arith::SubIOp>(loc, mMergeTileNum, 
      rewriter.create<arith::ConstantIndexOp>(loc, 1)));
  Value isLastK = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, 
      mergeKIdx, rewriter.create<arith::SubIOp>(loc, kMergeTileNum, 
      rewriter.create<arith::ConstantIndexOp>(loc, 1)));
  Value isLastN = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, 
      mergeNIdx, rewriter.create<arith::SubIOp>(loc, nMergeTileNum, 
      rewriter.create<arith::ConstantIndexOp>(loc, 1)));

  // // Use select operation to choose correct length
  Value currentMLen = rewriter.create<arith::SelectOp>(loc, isLastM,
      rewriter.create<arith::MulIOp>(loc, mLastMergeTileLen, mMetaTileLen),
      rewriter.create<arith::MulIOp>(loc, mMergeTileLen, mMetaTileLen));
  Value currentKLen = rewriter.create<arith::SelectOp>(loc, isLastK,
      rewriter.create<arith::MulIOp>(loc, kLastMergeTileLen, kMetaTileLen),
      rewriter.create<arith::MulIOp>(loc, kMergeTileLen, kMetaTileLen));
  Value currentNLen = rewriter.create<arith::SelectOp>(loc, isLastN,
      rewriter.create<arith::MulIOp>(loc, nLastMergeTileLen, nMetaTileLen),
      rewriter.create<arith::MulIOp>(loc, nMergeTileLen, nMetaTileLen));

  // Calculate starting positions
  Value mergeTileMStart = rewriter.create<arith::MulIOp>(loc,
      rewriter.create<arith::MulIOp>(loc, mergeMIdx, mMergeTileLen),
      mMetaTileLen);
  Value mergeTileKStart = rewriter.create<arith::MulIOp>(loc,
      rewriter.create<arith::MulIOp>(loc, mergeKIdx, kMergeTileLen),
      kMetaTileLen);
  Value mergeTileNStart = rewriter.create<arith::MulIOp>(loc,
      rewriter.create<arith::MulIOp>(loc, mergeNIdx, nMergeTileLen),
      nMetaTileLen);

  // Create subview
  Value aMergeTile = rewriter.create<memref::SubViewOp>(
      loc, aMemArray,
      SmallVector<OpFoldResult>{mergeTileMStart, mergeTileKStart},
      SmallVector<OpFoldResult>{currentMLen, currentKLen},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});
  Value bMergeTile = rewriter.create<memref::SubViewOp>(
      loc, bMemArray,
      SmallVector<OpFoldResult>{mergeTileKStart, mergeTileNStart},
      SmallVector<OpFoldResult>{currentKLen, currentNLen},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});

  mMergeTileLen = rewriter.create<arith::IndexCastOp>(loc, i64Type, mMergeTileLen);
  nMergeTileLen = rewriter.create<arith::IndexCastOp>(loc, i64Type, nMergeTileLen);
  kMergeTileLen = rewriter.create<arith::IndexCastOp>(loc, i64Type, kMergeTileLen);
  mMetaTileLen = rewriter.create<arith::IndexCastOp>(loc, i64Type, mMetaTileLen);
  nMetaTileLen = rewriter.create<arith::IndexCastOp>(loc, i64Type, nMetaTileLen);
  kMetaTileLen = rewriter.create<arith::IndexCastOp>(loc, i64Type, kMetaTileLen);

  rewriter.create<MergeTileMatMulOp>(loc,
      aMergeTile, aMemArray, bMergeTile, bMemArray, 
      aSpAddrStart, bSpAddrStart, cSpAddrStart,
      mMergeTileLen, nMergeTileLen, kMergeTileLen,
      mMetaTileLen, nMetaTileLen, kMetaTileLen);

  // Execute mvout at the end of m loop
  rewriter.setInsertionPointAfter(mloopOp);
  
  // TODO:fix this
  Value cMergeTile = rewriter.create<memref::SubViewOp>(
    loc, cMemArray,
    SmallVector<OpFoldResult>{rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)},
    SmallVector<OpFoldResult>{rewriter.getIndexAttr(16), rewriter.getIndexAttr(16)},
    SmallVector<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});
  rewriter.create<MvoutOp>(loc, cMergeTile, cSpAddrStart);

  rewriter.setInsertionPointAfter(kloopOp);

  rewriter.eraseOp(vecTileMatMulOp);
  return success();
}
private:
  int64_t dim;
  int64_t addrLen;
  int64_t spadRows;
  int64_t accRows;
  size_t sizeOfElemT;
  size_t sizeOfAccT;
  int64_t lane;
  int64_t warp;
};

//===----------------------------------------------------------------------===//
// Sparse Ops
//===----------------------------------------------------------------------===//
// struct BuckyBallCSRtoResidueLowering : public ConvertOpToLLVMPattern<CSRtoResidueOp> {
//   using ConvertOpToLLVMPattern<CSRtoResidueOp>::ConvertOpToLLVMPattern;
//   explicit BuckyBallCSRtoResidueLowering(LLVMTypeConverter &typeConverter, int64_t addrLen)
//     	: ConvertOpToLLVMPattern<CSRtoResidueOp>(typeConverter), addrLen(addrLen) {}
//   LogicalResult
//   matchAndRewrite(CSRtoResidueOp csrtoResidueOp, typename CSRtoResidueOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//   	Location loc = csrtoResidueOp.getLoc();
//   	Value csrRowPtrSpAddr = csrtoResidueOp.getCsrRowPtrSpAddr();
//   	Value csrColIdxSpAddr = csrtoResidueOp.getCsrColIdxSpAddr();
//   	Value residueArraySpAddr = csrtoResidueOp.getResidueArraySpAddr();
//   	Value iterNum = csrtoResidueOp.getIterNum();

// 		Value shift1 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(addrLen));
// 		csrRowPtrSpAddr = rewriter.create<arith::ShLIOp>(loc, csrRowPtrSpAddr, shift1);
// 		iterNum = rewriter.create<arith::ShLIOp>(loc, iterNum, shift1);
// 		// rs1 = csrRowPtrSpAddr << addrLen | csrColIdxSpAddr
// 		// rs2 = iterNum << addrLen | residueArraySpAddr
//   	Value rs1 = rewriter.create<arith::OrIOp>(loc, csrRowPtrSpAddr, csrColIdxSpAddr);
//   	Value rs2 = rewriter.create<arith::OrIOp>(loc, iterNum, residueArraySpAddr);
//   	rewriter.replaceOpWithNewOp<CSRtoResidue_IntrOp>(csrtoResidueOp, rs1, rs2);
//   	return success();
//   }
// private:
//   int64_t addrLen;
// };

// class BuckyBallSparseMergeTileMatMulLowering : public ConvertOpToLLVMPattern<SparseMergeTileMatMulOp> {
// public:
//   using ConvertOpToLLVMPattern<SparseMergeTileMatMulOp>::ConvertOpToLLVMPattern;
//   explicit BuckyBallSparseMergeTileMatMulLowering(LLVMTypeConverter &typeConverter)
//       : ConvertOpToLLVMPattern(typeConverter) {}

//   LogicalResult
//   matchAndRewrite(SparseMergeTileMatMulOp sparseMergeTileMatMulOp, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     return success();
//   }
// private:
// };


// class BuckyBallVecSparseTileMatMulLowering : public ConvertOpToLLVMPattern<VecSparseTileMatMulOp> {
// public:
//   using ConvertOpToLLVMPattern<VecSparseTileMatMulOp>::ConvertOpToLLVMPattern;
//   explicit BuckyBallVecSparseTileMatMulLowering(LLVMTypeConverter &typeConverter, size_t sizeOfElemT)
//       : ConvertOpToLLVMPattern(typeConverter), sizeOfElemT(sizeOfElemT) {}

//   LogicalResult
//   matchAndRewrite(VecSparseTileMatMulOp vecSparseTileMatMulOp, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//   Location loc = vecSparseTileMatMulOp.getLoc();
//   Value csrValue  = vecSparseTileMatMulOp.getCsrValue();
//   Value csrRowPtr = vecSparseTileMatMulOp.getCsrRowPtr();
//   Value csrColIdx = vecSparseTileMatMulOp.getCsrColIdx();
//   Value bMemArray = vecSparseTileMatMulOp.getBMemArray();
//   Value cMemArray = vecSparseTileMatMulOp.getCMemArray();

//   MemRefType csrValueType = dyn_cast<MemRefType>(csrValue.getType());
//   MemRefType csrRowPtrType = dyn_cast<MemRefType>(csrRowPtr.getType());
//   MemRefType csrColIdxType = dyn_cast<MemRefType>(csrColIdx.getType());
//   MemRefType bMemArrayType = dyn_cast<MemRefType>(bMemArray.getType());
//   MemRefType cMemArrayType = dyn_cast<MemRefType>(cMemArray.getType());

//   StridedLayoutAttr csrValueTypeLayout = dyn_cast<StridedLayoutAttr>(csrValueType.getLayout());
//   StridedLayoutAttr csrRowPtrTypeLayout = dyn_cast<StridedLayoutAttr>(csrRowPtrType.getLayout());
//   StridedLayoutAttr csrColIdxTypeLayout = dyn_cast<StridedLayoutAttr>(csrColIdxType.getLayout());
//   StridedLayoutAttr bMemArrayTypeLayout = dyn_cast<StridedLayoutAttr>(bMemArrayType.getLayout());
//   StridedLayoutAttr cMemArrayTypeLayout = dyn_cast<StridedLayoutAttr>(cMemArrayType.getLayout());

//   SmallVector<Type> resultType = {rewriter.getIndexType()};
//   TypeRange typeRange(resultType);

//   IntegerType i64Type = rewriter.getI64Type();

//   // Convert csrValue, csrRowPtr, csrColIdx, bMemArray, cMemArray to ArrayindexCast 

//   Value csrValueExtractOp =
//       rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange, csrValue);
//   if (csrValueTypeLayout) {
//     Value offset = rewriter.create<arith::ConstantIndexOp>(
//         loc, csrValueTypeLayout.getOffset() * sizeOfElemT);
//     csrValueExtractOp =
//         rewriter.create<arith::AddIOp>(loc, csrValueExtractOp, offset);
//   }
//   Value csrValueindexCastOp =
//       rewriter.create<arith::IndexCastOp>(loc, i64Type, csrValueExtractOp);
  
//   Value csrRowPtrExtractOp =
//       rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange, csrRowPtr);
//   if (csrRowPtrTypeLayout) {
//     Value offset = rewriter.create<arith::ConstantIndexOp>(
//         loc, csrRowPtrTypeLayout.getOffset() * sizeOfElemT);
//     csrRowPtrExtractOp =
//         rewriter.create<arith::AddIOp>(loc, csrRowPtrExtractOp, offset);
//   }
//   Value csrRowPtrindexCastOp =
//       rewriter.create<arith::IndexCastOp>(loc, i64Type, csrRowPtrExtractOp);

//   Value csrColIdxExtractOp =
//       rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange, csrColIdx);
//   if (csrColIdxTypeLayout) {
//     Value offset = rewriter.create<arith::ConstantIndexOp>(
//         loc, csrColIdxTypeLayout.getOffset() * sizeOfElemT);
//     csrColIdxExtractOp =
//         rewriter.create<arith::AddIOp>(loc, csrColIdxExtractOp, offset);
//   }
//   Value csrColIdxindexCastOp =
//       rewriter.create<arith::IndexCastOp>(loc, i64Type, csrColIdxExtractOp);

//   Value bMemArrayExtractOp =
//       rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange, bMemArray);
//   if (bMemArrayTypeLayout) {
//     Value offset = rewriter.create<arith::ConstantIndexOp>(
//         loc, bMemArrayTypeLayout.getOffset() * sizeOfElemT);
//     bMemArrayExtractOp =
//         rewriter.create<arith::AddIOp>(loc, bMemArrayExtractOp, offset);
//   }
//   Value bMemArrayindexCastOp =
//       rewriter.create<arith::IndexCastOp>(loc, i64Type, bMemArrayExtractOp);

//   Value cMemArrayExtractOp =
//       rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange, cMemArray);
//   if (cMemArrayTypeLayout) {
//     Value offset = rewriter.create<arith::ConstantIndexOp>(
//         loc, cMemArrayTypeLayout.getOffset() * sizeOfElemT);
//     cMemArrayExtractOp =
//         rewriter.create<arith::AddIOp>(loc, cMemArrayExtractOp, offset);
//   }
//   Value cMemArrayindexCastOp =
//       rewriter.create<arith::IndexCastOp>(loc, i64Type, cMemArrayExtractOp);

//   // Get A, B, C Matrix's shape, A[M][K], B[K][N], C[M][N]
//   Value NNZ = rewriter.create<memref::DimOp>(loc, csrValue, 0);
// 	Value M = rewriter.create<memref::DimOp>(loc, bMemArray, 0);
// 	Value K = rewriter.create<memref::DimOp>(loc, bMemArray, 1);
// 	Value N = rewriter.create<memref::DimOp>(loc, cMemArray, 1);

// // Loop K
// // step 1: merge metatiles into merge tiles. cscColptr
// // csr -> rowptr[i+16] - rowptr[i] = NNZ[i] 每16列的非零元素个数
// // csr -> rowptr[i+1] - rowptr[i] = NNZ[i] 每16列的非零元素个数

//   Operation *loopOp = nullptr;
//   Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
//   Value upperBound = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), metaKNum);
//   Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);    
//   auto kLoop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
//   loopOp = kLoop.getOperation();
//   rewriter.setInsertionPointToStart(kLoop.getBody());

//   rewriter.setInsertionPointAfter(loopOp);

// // step 2: longest-first

// // step 3: mvinA(cscRowIdx), mvinA(cscValue)

// // step 4: mvinB

// // step 5: vecMatMulWarp16

// // step 6: mvout

//   return success();
// }
// private:
//   size_t sizeOfElemT;
// };




void mlir::populateBuckyBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, int64_t dim,
    int64_t addrLen, int64_t accRows, int64_t spadRows, size_t sizeOfElemT,
    size_t sizeOfAccT, int64_t warp , int64_t lane) {
  patterns
      .add<ForwardOperands<func::CallOp>, ForwardOperands<func::CallIndirectOp>,
           ForwardOperands<func::ReturnOp>>(converter, &converter.getContext());
  patterns.add<BuckyBallFlushLowering>(converter);
  patterns.add<BuckyBallMvinLowering>(converter, addrLen);
  patterns.add<BuckyBallMvin2Lowering>(converter, addrLen);
  patterns.add<BuckyBallMvin3Lowering>(converter, addrLen);
  patterns.add<BuckyBallMvoutLowering>(converter, addrLen);
  patterns.add<BuckyBallVecMulWarp16Lowering>(converter, addrLen);
  patterns.add<BuckyBallMetaTileMatMulLowering>(converter, lane);
  patterns.add<BuckyBallMergeTileMatMulLowering>(converter);
  patterns.add<BuckyBallVecTileMatMulLowering>(converter, dim, 
      addrLen, spadRows, accRows, sizeOfElemT, sizeOfAccT, warp, lane);
  // patterns.add<BuckyBallVecSparseTileMatMulLowering>(converter, sizeOfElemT);
  // patterns.add<BuckyBallCSRtoResidueLowering>(converter, addrLen);
  // patterns.add<BuckyBallSparseMergeTileMatMulLowering>(converter, sizeOfElemT);	
}

void mlir::configureBuckyBallLegalizeForExportTarget(
    LLVMConversionTarget &target) {
//   target.addLegalDialect<arith::ArithDialect,
//                          scf::SCFDialect, 
//                          memref::MemRefDialect,
//                          BuckyBallDialect,
//                          LLVM::LLVMDialect>();
  target.addLegalOp<Flush_IntrOp, Mvin_IntrOp, Mvin2_IntrOp, Mvin3_IntrOp, 
                    Mvout_IntrOp, VecMulWarp16_IntrOp>();
  target.addIllegalOp<FlushOp, MvinOp, Mvin2Op, Mvin3Op, MvoutOp, 
                      VecTileMatMulOp, MergeTileMatMulOp, 
                      MetaTileMatMulOp, VecMulWarp16Op, PrintOp, PrintScalarOp>();
}
