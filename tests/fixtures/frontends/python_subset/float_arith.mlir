module {
  func.func @blend(%arg0: f64, %arg1: f64, %arg2: i64) -> f64 {
    %0 = arith.negf %arg0 : f64
    %1 = arith.constant 1.5 : f64
    %2 = arith.subf %arg1, %1 : f64
    %3 = arith.divf %0, %2 : f64
    %4 = arith.sitofp %arg2 : i64 to f64
    %5 = arith.mulf %3, %4 : f64
    return %5 : f64
  }
}
