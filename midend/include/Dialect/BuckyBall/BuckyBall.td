//===-- BuckyBall.td - BuckyBall dialect operation definitions --- tablegen --===//
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
// This file defines the basic operations for the BuckyBall dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_OPS
#define BUCKYBALL_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// BuckyBall dialect definition
//===----------------------------------------------------------------------===//

def BuckyBall_Dialect : Dialect {
  let name = "buckyball";
  let cppNamespace = "::buddy::buckyball";
  let summary = "Basic dialect to target RISC-V BuckyBall extension";
  let description = [{
    BuckyBall is an accelerator based on Vector and software-defined 
    pipeline architecture. For more details about BuckyBall,
    please see the [BuckyBall repo](TBC).
  }];
}

//===----------------------------------------------------------------------===//
// BuckyBall operation definitions
//===----------------------------------------------------------------------===//
class BuckyBall_Op<string mnemonic, list<Trait> traits = []> :
  Op<BuckyBall_Dialect, mnemonic, traits> {}

/* Multi-Level Tiling Design
 *  -> VecTileMatMulOp
 *  aMemArray, bMemArray, cMemArray - (Value) abc matrix's main memory address
 *  warpNum - use how many warps (choose the exec vector unit)  
 *
 *  -> MergeTileMatMulOp (divide <mergeM x mergeN x mergeK> into <mergeM x mergeN x 1>) Loop over mergeKLen
 *  aMemArray, bMemArray - (MemRef) MergeTile's main memory memref
 *  aMemAddrStart, bMemAddrStart, cMemAddrStart - abc matrix's relative main memory address ptr
 *  aSpAddrStart, bSpAddrStart, cSpAddrStart - abc matrix's scrachpad/accumulator ptr
 *  m, n, k - abc matrix's shape/size (used to calculate abc matrix's address stride) 
 *  mergeMLen, mergeNLen, mergeKLen - mergeTile's shape/size
 *
 *  -> MetaTileMatMulOp (divide <mergeM x mergeN x 1> into <1 x mergeN x 1>) Loop over mergeMLen
 *  aMemArray - (MemRef) MetaTile's main memory memref
 *  aSpAddrStart, bSpAddrStart, cSpAddrStart - abc matrix's relative start address of scrachpad/accumulator ptr
 *  mergeMLen, mergeNLen - metaTile's shape/size
 *  
 *  -> VecMulWarp16Op
 *  aSpAddrStart, bSpAddrStart, cSpAddrStart - abc matrix's relative start address of scrachpad/accumulator ptr
 *  nLen - VecMul's lane size (sum of mergeNLen)
 *
 *  -> mvin
 *  input - MemRef
 *  addr - I64
 *
 *  -> mvout
 *  output - MemRef
 *  addr - I64
 *
*/

// Tile level operation
def VecTileMatMulOp : BuckyBall_Op<"bb_tile_matmul"> {
  let summary = "Vector-matrix multiplication";
  let description = [{
    Perform matrix multiplication operation in vector unit.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       MemRefRankOf<[AnyType], [2]>:$bMemArray,
                       MemRefRankOf<[AnyType], [2]>:$cMemArray,
                       DefaultValuedAttr<I64Attr, "16">:$warpNum);
  let assemblyFormat = [{$aMemArray $bMemArray $cMemArray attr-dict `:` 
    type($aMemArray) type($bMemArray) type($cMemArray)}];
}

def MergeTileMatMulOp : BuckyBall_Op<"bb_mergetile_matmul"> {
  let summary = "Merge tile matrix multiplication";
  let description = [{
    Perform matrix multiplication operation in vector unit.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMergeTileArray,
                       MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       MemRefRankOf<[AnyType], [2]>:$bMergeTileArray,
                       MemRefRankOf<[AnyType], [2]>:$bMemArray,
                       I64:$aSpAddrStart, I64:$bSpAddrStart, I64:$cSpAddrStart,
                       I64:$metaMNum, I64:$metaNNum, I64:$metaKNum,
                       I64:$metaMLen, I64:$metaNLen, I64:$metaKLen);
  let assemblyFormat = [{$aMergeTileArray $aMemArray $bMergeTileArray $bMemArray $aSpAddrStart $bSpAddrStart $cSpAddrStart $metaMNum $metaNNum $metaKNum $metaMLen $metaNLen $metaKLen attr-dict `:` 
    type($aMergeTileArray) type($aMemArray) type($bMergeTileArray) type($bMemArray) type($aSpAddrStart) type($bSpAddrStart) type($cSpAddrStart) type($metaMNum) type($metaNNum) type($metaKNum) type($metaMLen) type($metaNLen) type($metaKLen)}];
}

def MetaTileMatMulOp : BuckyBall_Op<"bb_metatile_matmul"> {
  let summary = "Meta tile matrix multiplication";
  let description = [{
    Perform matrix multiplication operation in vector unit.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMetaTileArray,
                       MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       I64:$aSpAddrStart, I64:$bSpAddrStart, I64:$cSpAddrStart,
                       I64:$metaMNum, I64:$metaNNum,
                       I64:$metaMLen, I64:$metaNLen);
  let assemblyFormat = [{$aMetaTileArray $aMemArray $aSpAddrStart $bSpAddrStart $cSpAddrStart $metaMNum $metaNNum $metaMLen $metaNLen attr-dict `:` 
     type($aMetaTileArray) type($aMemArray) type($aSpAddrStart) type($bSpAddrStart) type($cSpAddrStart) type($metaMNum) type($metaNNum) type($metaMLen) type($metaNLen)}];
}

// instruction level operation
def VecMulWarp16Op : BuckyBall_Op<"bb_mul_warp16"> {
  let summary = "Vector multiply with 16 warp parallelism";
  let description = [{
    Perform vector multiply operation in vector unit 
    with warp number is 16.
  }];
  let arguments = (ins I64:$aSpAddr, 
                       I64:$bSpAddr, 
                       I64:$cSpAddr, 
                       I64:$nLen);
                  //  DefaultValuedAttr<BoolAttr, "false">:$toVec,
                  //  DefaultValuedAttr<I8Attr, "0">:$srcVecIdx,
                  //  DefaultValuedAttr<I8Attr, "2">:$dstVecIdx);
  let assemblyFormat = [{$aSpAddr $bSpAddr $cSpAddr $nLen attr-dict `:` 
    type($aSpAddr) type($bSpAddr) type($cSpAddr) type($nLen)}];
}

// def VecTransOp : BuckyBall_Op<"vec_trans"> {
//   let summary = "Vector transpose";
//   let description = [{
//     Perform transpose operation on vector.
//   }];
//   let arguments = (ins MemRefRankOf<[AnyType], [2]>:$inputArray,
//                        MemRefRankOf<[AnyType], [2]>:$outputArray,
//                        DefaultValuedAttr<I8Attr, "0">:$toVec,
//                        DefaultValuedAttr<I8Attr, "16">:$vecIdx);
//   let assemblyFormat = [{$inputArray $outputArray attr-dict `:` 
//     type($inputArray) type($outputArray) }];
// }

// def VecSparseTileMatMulOp : BuckyBall_Op<"bb_sparse_tile_matmul"> {
//   let summary = "Vector sparse matrix multiplication";
//   let description = [{
//     Perform sparse matrix multiplication operation in vector unit.
//   }];
//   let arguments = (ins MemRefRankOf<[AnyType], [1]>:$csrValue,
//                        MemRefRankOf<[AnyType], [1]>:$csrRowPtr,
//                        MemRefRankOf<[AnyType], [1]>:$csrColIdx,
//                        MemRefRankOf<[AnyType], [2]>:$bMemArray,
//                        MemRefRankOf<[AnyType], [2]>:$cMemArray);
//   let assemblyFormat = [{$csrValue $csrRowPtr $csrColIdx $bMemArray $cMemArray attr-dict `:` 
//     type($csrValue) type($csrRowPtr) type($csrColIdx) type($bMemArray) type($cMemArray)}];
// }

// def CSRtoResidueOp : BuckyBall_Op<"bb_csr_to_residue"> {
//   let summary = "Convert CSR to residue";
//   let description = [{
//     Convert CSR to residue.
//   }];
//   let arguments = (ins I64:$csrRowPtrSpAddr, 
//                        I64:$csrColIdxSpAddr,
//                        I64:$residueArraySpAddr,
//                        I64:$iterNum);
//   let assemblyFormat = [{$csrRowPtrSpAddr $csrColIdxSpAddr $residueArraySpAddr $iterNum attr-dict `:` 
//     type($csrRowPtrSpAddr) type($csrColIdxSpAddr) type($residueArraySpAddr) type($iterNum)}];
// }

def FlushOp : BuckyBall_Op<"flush"> {
  let summary = "Flush op";
  let description = [{
    Flush operation flushes the TLB.
    If skip = 1, the current TLB request is skipped.
    Otherwise, the current TLB request is repeated.
  }];
  let arguments = (ins I64:$skip);
  let assemblyFormat = "$skip attr-dict `:` type($skip)";
}

def MvinOp : BuckyBall_Op<"bb_mvin"> {
  let summary = "Load operation";
  let description = [{
    Move data from main memory to scratchpad
    - MemRef to load in.
      (including DRAM address, number of columns, number of rows)
    - Local scratchpad or accumulator address.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input, I64:$addr);
  let assemblyFormat = "$input $addr attr-dict `:` type($input) type($addr)";
}

// def Mvin2Op : BuckyBall_Op<"mvin2"> {
//   let summary = "Load operation";
//   let description = [{
//       Similar to Mvin
//           Move data from main memory to scratchpad
//       - MemRef to load in.
//         (including DRAM address, number of columns, number of rows)
//       - Local scratchpad or accumulator address.
//   }];
//   let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input, I64:$addr);
//   let assemblyFormat = "$input $addr attr-dict `:` type($input) type($addr)";
// }

// def Mvin3Op : BuckyBall_Op<"mvin3"> {
//   let summary = "Load operation";
//   let description = [{
//       Similar to Mvin and Mvin2
//           Move data from main memory to scratchpad
//       - MemRef to load in.
//         (including DRAM address, number of columns, number of rows)
//       - Local scratchpad or accumulator address.
//   }];
//   let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input, I64:$addr);
//   let assemblyFormat = "$input $addr attr-dict `:` type($input) type($addr)";
// }

def MvoutOp : BuckyBall_Op<"bb_mvout"> {
  let summary = "Store operation";
  let description = [{
    Move data from scratchpad to L2/DRAM
    - MemRef to store from scratchpad.
      (including DRAM address, number of columns, number of rows)
    - Local scratchpad address.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$output, I64:$addr);
  let assemblyFormat = "$output $addr attr-dict `:` type($output) type($addr)";
}

def PrintOp : BuckyBall_Op<"print"> {
  let summary = "Print memref value.";
  let arguments = (ins AnyTypeOf<[I8MemRef, I32MemRef, F32MemRef, F64MemRef]>:$input); 
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def PrintScalarOp : BuckyBall_Op<"print_scalar"> {
  let summary = "Print a scalar value.";
  let arguments = (ins AnyTypeOf<[I64, I32, F32, F64]>:$input); 
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// BuckyBall intrinsic operation definitions
//===----------------------------------------------------------------------===//

class BuckyBall_IntrOpBase<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/BuckyBall_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[],
                  /*list<int> overloadedOperands=*/[],
                  /*list<Trait> traits=*/traits,
                  /*int numResults=*/0>;

// will be detected as Flush_IntrOp (truncated the first letter)
def BuckyBall_Flush_IntrOp : BuckyBall_IntrOpBase<"flush">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def BuckyBall_Mvin_IntrOp : BuckyBall_IntrOpBase<"bb_mvin">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

// def BuckyBall_Mvin2_IntrOp : BuckyBall_IntrOpBase<"mvin2">,
//   Arguments<(ins LLVM_Type, LLVM_Type)>;

// def BuckyBall_Mvin3_IntrOp : BuckyBall_IntrOpBase<"mvin3">,
//   Arguments<(ins LLVM_Type, LLVM_Type)>;

def BuckyBall_Mvout_IntrOp : BuckyBall_IntrOpBase<"bb_mvout">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def BuckyBall_VecMulWarp16_IntrOp : BuckyBall_IntrOpBase<"bb_mul_warp16">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

// def BuckyBall_CSRtoResidue_IntrOp : BuckyBall_IntrOpBase<"bb_csr_to_residue">,
//   Arguments<(ins LLVM_Type, LLVM_Type)>;

// def BuckyBall_VecTrans_IntrOp : BuckyBall_IntrOpBase<"vec_trans">,
//   Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // BUCKYBALL_OPS
