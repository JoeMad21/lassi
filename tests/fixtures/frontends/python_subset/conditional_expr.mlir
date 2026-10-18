module {
  func.func @inside(%arg0: f64, %arg1: f64) -> i64 {
    %0 = arith.mulf %arg0, %arg0 : f64
    %1 = arith.mulf %arg1, %arg1 : f64
    %2 = arith.addf %0, %1 : f64
    %3 = arith.constant 4.0 : f64
    %4 = arith.cmpf ole, %2, %3 : f64
    %5 = scf.if %4 -> (i64) {
      %6 = arith.constant 1 : i64
      scf.yield %6 : i64
    } else {
      %7 = arith.constant 0 : i64
      scf.yield %7 : i64
    }
    return %5 : i64
  }
}
