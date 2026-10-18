module {
  func.func @axpy(%arg0: f64, %arg1: f64, %arg2: f64) -> f64 {
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    return %1 : f64
  }
  func.func @twice(%arg0: f64) -> f64 {
    %0 = arith.constant 2.0 : f64
    %1 = arith.constant 0.0 : f64
    %2 = call @axpy(%0, %arg0, %1) : (f64, f64, f64) -> f64
    return %2 : f64
  }
}
