module {
  func.func private @subgraph0() -> memref<16x8xf32>
  func.func @forward(%arg0: memref<31xi32>, %arg1: memref<16x8xi32>, %arg2: memref<31xi32>) -> memref<16x8xf32> {
    %0 = call @subgraph0() : () -> memref<16x8xf32>
    return %0 : memref<16x8xf32>
  }
}

