module {
  func.func @stripes(%arg0: i64, %arg1: memref<?xf64>) {
    %0 = memref.alloca() : memref<i64>
    %1 = arith.index_cast %arg0 : i64 to index
    %2 = arith.constant 0 : index
    %3 = arith.constant 1 : index
    scf.for %4 = %2 to %1 step %3 {
      %5 = arith.index_cast %4 : index to i64
      memref.store %5, %0[] : memref<i64>
      %6 = arith.constant 0.0 : f64
      %7 = memref.load %0[] : memref<i64>
      %8 = arith.index_cast %7 : i64 to index
      memref.store %6, %arg1[%8] : memref<?xf64>
    }
    %9 = arith.constant 1 : i64
    %10 = arith.index_cast %9 : i64 to index
    %11 = arith.index_cast %arg0 : i64 to index
    %12 = arith.constant 2 : index
    scf.parallel (%13) = (%10) to (%11) step (%12) {
      memref.alloca_scope {
        %14 = memref.alloca() : memref<i64>
        %15 = arith.index_cast %13 : index to i64
        memref.store %15, %14[] : memref<i64>
        %16 = arith.constant 1.0 : f64
        %17 = memref.load %14[] : memref<i64>
        %18 = arith.index_cast %17 : i64 to index
        memref.store %16, %arg1[%18] : memref<?xf64>
      }
      scf.yield
    }
    return
  }
}
