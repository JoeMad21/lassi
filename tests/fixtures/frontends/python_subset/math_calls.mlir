module {
  func.func @norm(%arg0: f64, %arg1: f64) -> f64 {
    %0 = arith.mulf %arg0, %arg0 : f64
    %1 = arith.mulf %arg1, %arg1 : f64
    %2 = arith.addf %0, %1 : f64
    %3 = math.sqrt %2 : f64
    return %3 : f64
  }
  func.func @wave(%arg0: f64, %arg1: i64) -> f64 {
    %0 = arith.negf %arg0 : f64
    %1 = math.exp %0 : f64
    %2 = math.cos %arg0 : f64
    %3 = arith.mulf %1, %2 : f64
    %4 = math.sin %arg0 : f64
    %5 = math.absf %4 : f64
    %6 = arith.addf %3, %5 : f64
    %7 = arith.sitofp %arg1 : i64 to f64
    %8 = math.log %7 : f64
    %9 = arith.subf %6, %8 : f64
    return %9 : f64
  }
  func.func @whole(%arg0: f64) -> i64 {
    %0 = arith.fptosi %arg0 : f64 to i64
    %1 = arith.constant -3 : i64
    %2 = math.absi %1 : i64
    %3 = arith.addi %0, %2 : i64
    return %3 : i64
  }
}
