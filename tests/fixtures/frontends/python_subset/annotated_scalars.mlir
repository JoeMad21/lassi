module {
  func.func @clamp_count(%arg0: i64, %arg1: f64) -> i64 {
    %0 = memref.alloca() : memref<i64>
    %1 = memref.alloca() : memref<f64>
    %2 = memref.alloca() : memref<i1>
    %3 = memref.alloca() : memref<i64>
    %4 = arith.constant 10 : i64
    memref.store %4, %0[] : memref<i64>
    %5 = arith.constant 0.5 : f64
    memref.store %5, %1[] : memref<f64>
    %6 = memref.load %1[] : memref<f64>
    %7 = arith.mulf %arg1, %6 : f64
    %8 = arith.constant 1.0 : f64
    %9 = arith.cmpf ogt, %7, %8 : f64
    memref.store %9, %2[] : memref<i1>
    memref.store %arg0, %3[] : memref<i64>
    %10 = memref.load %2[] : memref<i1>
    scf.if %10 {
      %11 = memref.load %0[] : memref<i64>
      memref.store %11, %3[] : memref<i64>
    }
    %12 = memref.load %3[] : memref<i64>
    return %12 : i64
  }
}
