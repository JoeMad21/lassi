module {
  func.func @total(%arg0: i64) -> i64 {
    %0 = memref.alloca() : memref<i64>
    %1 = memref.alloca() : memref<i64>
    %2 = memref.alloca() : memref<i64>
    %3 = memref.alloca() : memref<i64>
    %4 = arith.constant 0 : i64
    memref.store %4, %0[] : memref<i64>
    %5 = arith.index_cast %arg0 : i64 to index
    %6 = arith.constant 0 : index
    %7 = arith.constant 1 : index
    scf.for %8 = %6 to %5 step %7 {
      %9 = arith.index_cast %8 : index to i64
      memref.store %9, %1[] : memref<i64>
      %10 = memref.load %0[] : memref<i64>
      %11 = memref.load %1[] : memref<i64>
      %12 = arith.addi %10, %11 : i64
      memref.store %12, %0[] : memref<i64>
    }
    %13 = arith.constant 1 : i64
    %14 = arith.index_cast %13 : i64 to index
    %15 = arith.index_cast %arg0 : i64 to index
    %16 = arith.constant 1 : index
    scf.for %17 = %14 to %15 step %16 {
      %18 = arith.index_cast %17 : index to i64
      memref.store %18, %2[] : memref<i64>
      %19 = memref.load %0[] : memref<i64>
      %20 = memref.load %2[] : memref<i64>
      %21 = arith.subi %19, %20 : i64
      memref.store %21, %0[] : memref<i64>
    }
    %22 = arith.constant 0 : i64
    %23 = arith.index_cast %22 : i64 to index
    %24 = arith.index_cast %arg0 : i64 to index
    %25 = arith.constant 3 : index
    scf.for %26 = %23 to %24 step %25 {
      %27 = arith.index_cast %26 : index to i64
      memref.store %27, %3[] : memref<i64>
      %28 = memref.load %0[] : memref<i64>
      %29 = memref.load %3[] : memref<i64>
      %30 = arith.constant 2 : i64
      %31 = arith.muli %29, %30 : i64
      %32 = arith.addi %28, %31 : i64
      memref.store %32, %0[] : memref<i64>
    }
    %33 = memref.load %0[] : memref<i64>
    return %33 : i64
  }
}
