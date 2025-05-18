module {
  func.func @subgraph0() -> tensor<16x8xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x8xf32>
    return %cst : tensor<16x8xf32>
  }
}

