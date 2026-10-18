module {
  func.func @ramp(%arg0: i64) {
    %0 = memref.alloca() : memref<i64>
    %1 = arith.index_cast %arg0 : i64 to index
    %2 = memref.alloc(%1) : memref<?xf64>
    %3 = arith.constant 0 : index
    %4 = arith.constant 1 : index
    %5 = arith.constant 0.0 : f64
    scf.for %6 = %3 to %1 step %4 {
      memref.store %5, %2[%6] : memref<?xf64>
    }
    %7 = arith.constant 16 : i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = memref.alloc(%8) : memref<?xi32>
    %10 = arith.constant 0 : index
    %11 = arith.constant 1 : index
    %12 = arith.constant 0 : i32
    scf.for %13 = %10 to %8 step %11 {
      memref.store %12, %9[%13] : memref<?xi32>
    }
    %14 = arith.index_cast %arg0 : i64 to index
    %15 = arith.constant 0 : index
    %16 = arith.constant 1 : index
    scf.for %17 = %15 to %14 step %16 {
      %18 = arith.index_cast %17 : index to i64
      memref.store %18, %0[] : memref<i64>
      %19 = arith.constant 0.5 : f64
      %20 = memref.load %0[] : memref<i64>
      %21 = arith.sitofp %20 : i64 to f64
      %22 = arith.mulf %19, %21 : f64
      %23 = memref.load %0[] : memref<i64>
      %24 = arith.index_cast %23 : i64 to index
      memref.store %22, %2[%24] : memref<?xf64>
    }
    %25 = arith.constant 0 : index
    %26 = memref.dim %2, %25 : memref<?xf64>
    %27 = arith.index_cast %26 : index to i64
    %28 = arith.constant 0 : i64
    %29 = arith.index_cast %28 : i64 to index
    %30 = arith.trunci %27 : i64 to i32
    memref.store %30, %9[%29] : memref<?xi32>
    memref.dealloc %2 : memref<?xf64>
    memref.dealloc %9 : memref<?xi32>
    return
  }
}
