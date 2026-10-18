module {
  func.func @scale(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: memref<?xi8>) {
    %0 = memref.alloca() : memref<i64>
    %1 = arith.constant 0 : index
    %2 = memref.dim %arg0, %1 : memref<?xf32>
    %3 = arith.index_cast %2 : index to i64
    %4 = arith.index_cast %3 : i64 to index
    %5 = arith.constant 0 : index
    %6 = arith.constant 1 : index
    scf.for %7 = %5 to %4 step %6 {
      %8 = arith.index_cast %7 : index to i64
      memref.store %8, %0[] : memref<i64>
      %9 = memref.load %0[] : memref<i64>
      %10 = arith.index_cast %9 : i64 to index
      %11 = memref.load %arg0[%10] : memref<?xf32>
      %12 = arith.extf %11 : f32 to f64
      %13 = memref.load %0[] : memref<i64>
      %14 = arith.index_cast %13 : i64 to index
      %15 = memref.load %arg1[%14] : memref<?xi32>
      %16 = arith.extsi %15 : i32 to i64
      %17 = arith.sitofp %16 : i64 to f64
      %18 = arith.mulf %12, %17 : f64
      %19 = memref.load %0[] : memref<i64>
      %20 = arith.index_cast %19 : i64 to index
      %21 = arith.truncf %18 : f64 to f32
      memref.store %21, %arg0[%20] : memref<?xf32>
      %22 = memref.load %0[] : memref<i64>
      %23 = arith.index_cast %22 : i64 to index
      %24 = memref.load %arg2[%23] : memref<?xi8>
      %25 = arith.extui %24 : i8 to i64
      %26 = arith.constant 1 : i64
      %27 = arith.addi %25, %26 : i64
      %28 = arith.trunci %27 : i64 to i8
      memref.store %28, %arg2[%23] : memref<?xi8>
    }
    return
  }
}
