module {
  func.func private @subgraph0(memref<16x16xf32, strided<[16, 1], offset: ?>>, memref<16x16xf32, strided<[16, 1], offset: ?>>) -> memref<16x16xf32>
  func.func @forward(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) -> memref<16x16xf32> {
    %cast = memref.cast %arg1 : memref<16x16xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
    %cast_0 = memref.cast %arg0 : memref<16x16xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
    %0 = call @subgraph0(%cast, %cast_0) : (memref<16x16xf32, strided<[16, 1], offset: ?>>, memref<16x16xf32, strided<[16, 1], offset: ?>>) -> memref<16x16xf32>
    return %0 : memref<16x16xf32>
  }
}

