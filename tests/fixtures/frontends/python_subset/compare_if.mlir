module {
  func.func @classify(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: i1) -> i64 {
    %0 = memref.alloca() : memref<i64>
    %1 = arith.constant 0 : i64
    memref.store %1, %0[] : memref<i64>
    %2 = arith.cmpf olt, %arg0, %arg1 : f64
    %3 = scf.if %2 -> (i1) {
      %4 = arith.constant true
      scf.yield %4 : i1
    } else {
      %5 = arith.cmpf ogt, %arg0, %arg2 : f64
      scf.yield %5 : i1
    }
    scf.if %3 {
      %6 = arith.constant 2 : i64
      memref.store %6, %0[] : memref<i64>
    } else {
      %7 = scf.if %arg3 -> (i1) {
        %8 = arith.cmpf oeq, %arg0, %arg1 : f64
        %9 = arith.constant true
        %10 = arith.xori %8, %9 : i1
        scf.yield %10 : i1
      } else {
        %11 = arith.constant false
        scf.yield %11 : i1
      }
      scf.if %7 {
        %12 = arith.constant 1 : i64
        memref.store %12, %0[] : memref<i64>
      } else {
        %13 = arith.constant -1 : i64
        memref.store %13, %0[] : memref<i64>
      }
    }
    %14 = memref.load %0[] : memref<i64>
    return %14 : i64
  }
  func.func @is_small(%arg0: i64) -> i1 {
    %0 = arith.constant 9 : i64
    %1 = arith.cmpi sle, %arg0, %0 : i64
    return %1 : i1
  }
}
