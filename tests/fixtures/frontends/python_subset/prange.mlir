module {
  func.func @row_sums(%arg0: i64, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.constant 0 : index
    %2 = arith.constant 1 : index
    scf.parallel (%3) = (%1) to (%0) step (%2) {
      memref.alloca_scope {
        %4 = memref.alloca() : memref<i64>
        %5 = memref.alloca() : memref<f64>
        %6 = memref.alloca() : memref<i64>
        %7 = arith.index_cast %3 : index to i64
        memref.store %7, %4[] : memref<i64>
        %8 = arith.constant 0.0 : f64
        memref.store %8, %5[] : memref<f64>
        %9 = arith.index_cast %arg0 : i64 to index
        %10 = arith.constant 0 : index
        %11 = arith.constant 1 : index
        scf.for %12 = %10 to %9 step %11 {
          %13 = arith.index_cast %12 : index to i64
          memref.store %13, %6[] : memref<i64>
          %14 = memref.load %5[] : memref<f64>
          %15 = memref.load %4[] : memref<i64>
          %16 = arith.muli %15, %arg0 : i64
          %17 = memref.load %6[] : memref<i64>
          %18 = arith.addi %16, %17 : i64
          %19 = arith.index_cast %18 : i64 to index
          %20 = memref.load %arg1[%19] : memref<?xf64>
          %21 = arith.addf %14, %20 : f64
          memref.store %21, %5[] : memref<f64>
        }
        %22 = memref.load %5[] : memref<f64>
        %23 = memref.load %4[] : memref<i64>
        %24 = arith.index_cast %23 : i64 to index
        memref.store %22, %arg2[%24] : memref<?xf64>
      }
      scf.yield
    }
    return
  }
}
