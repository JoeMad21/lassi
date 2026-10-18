module {
  func.func @fill(%arg0: i64, %arg1: f64, %arg2: memref<?xf64>) {
    %0 = memref.alloca() : memref<i64>
    %1 = arith.index_cast %arg0 : i64 to index
    %2 = arith.constant 0 : index
    %3 = arith.constant 1 : index
    scf.for %4 = %2 to %1 step %3 {
      %5 = arith.index_cast %4 : index to i64
      memref.store %5, %0[] : memref<i64>
      %6 = memref.load %0[] : memref<i64>
      %7 = arith.index_cast %6 : i64 to index
      memref.store %arg1, %arg2[%7] : memref<?xf64>
    }
    return
  }
  func.func @first(%arg0: memref<?xf64>) -> f64 {
    %0 = arith.constant 0 : i64
    %1 = arith.index_cast %0 : i64 to index
    %2 = memref.load %arg0[%1] : memref<?xf64>
    return %2 : f64
  }
  func.func @run(%arg0: i64, %arg1: memref<?xf64>) -> f64 {
    %0 = memref.alloca() : memref<f64>
    %1 = memref.alloca() : memref<i64>
    %2 = arith.constant 2 : i64
    %3 = arith.sitofp %2 : i64 to f64
    call @fill(%arg0, %3, %arg1) : (i64, f64, memref<?xf64>) -> ()
    %4 = arith.constant 0.0 : f64
    memref.store %4, %0[] : memref<f64>
    %5 = arith.index_cast %arg0 : i64 to index
    %6 = arith.constant 0 : index
    %7 = arith.constant 1 : index
    scf.for %8 = %6 to %5 step %7 {
      %9 = arith.index_cast %8 : index to i64
      memref.store %9, %1[] : memref<i64>
      %10 = memref.load %0[] : memref<f64>
      %11 = func.call @first(%arg1) : (memref<?xf64>) -> f64
      %12 = memref.load %1[] : memref<i64>
      %13 = arith.sitofp %12 : i64 to f64
      %14 = arith.mulf %11, %13 : f64
      %15 = arith.addf %10, %14 : f64
      memref.store %15, %0[] : memref<f64>
    }
    %16 = memref.load %0[] : memref<f64>
    return %16 : f64
  }
}
