module {
  func.func @mix(%arg0: i64, %arg1: i64) -> i64 {
    %0 = arith.addi %arg0, %arg1 : i64
    %1 = arith.subi %arg0, %arg1 : i64
    %2 = arith.muli %0, %1 : i64
    %3 = arith.constant 3 : i64
    %4 = arith.floordivsi %2, %3 : i64
    %5 = arith.floordivsi %4, %arg1 : i64
    %6 = arith.muli %5, %arg1 : i64
    %7 = arith.subi %4, %6 : i64
    return %7 : i64
  }
  func.func @bits(%arg0: i64, %arg1: i64) -> i64 {
    %0 = arith.andi %arg0, %arg1 : i64
    %1 = arith.xori %arg0, %arg1 : i64
    %2 = arith.constant 2 : i64
    %3 = arith.shli %1, %2 : i64
    %4 = arith.constant 1 : i64
    %5 = arith.shrsi %3, %4 : i64
    %6 = arith.ori %0, %5 : i64
    return %6 : i64
  }
  func.func @ratio(%arg0: i64, %arg1: i64) -> f64 {
    %0 = arith.sitofp %arg0 : i64 to f64
    %1 = arith.sitofp %arg1 : i64 to f64
    %2 = arith.divf %0, %1 : f64
    return %2 : f64
  }
}
