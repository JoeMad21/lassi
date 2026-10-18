module {
  func.func @collatz_steps(%arg0: i64) -> i64 {
    %0 = memref.alloca() : memref<i64>
    memref.store %arg0, %0[] : memref<i64>
    %1 = memref.alloca() : memref<i64>
    %2 = arith.constant 0 : i64
    memref.store %2, %1[] : memref<i64>
    scf.while : () -> () {
      %3 = memref.load %0[] : memref<i64>
      %4 = arith.constant 1 : i64
      %5 = arith.cmpi ne, %3, %4 : i64
      scf.condition(%5)
    } do {
      %6 = memref.load %0[] : memref<i64>
      %7 = arith.constant 2 : i64
      %8 = arith.floordivsi %6, %7 : i64
      %9 = arith.muli %8, %7 : i64
      %10 = arith.subi %6, %9 : i64
      %11 = arith.constant 0 : i64
      %12 = arith.cmpi eq, %10, %11 : i64
      scf.if %12 {
        %13 = memref.load %0[] : memref<i64>
        %14 = arith.constant 2 : i64
        %15 = arith.floordivsi %13, %14 : i64
        memref.store %15, %0[] : memref<i64>
      } else {
        %16 = arith.constant 3 : i64
        %17 = memref.load %0[] : memref<i64>
        %18 = arith.muli %16, %17 : i64
        %19 = arith.constant 1 : i64
        %20 = arith.addi %18, %19 : i64
        memref.store %20, %0[] : memref<i64>
      }
      %21 = memref.load %1[] : memref<i64>
      %22 = arith.constant 1 : i64
      %23 = arith.addi %21, %22 : i64
      memref.store %23, %1[] : memref<i64>
      scf.yield
    }
    %24 = memref.load %1[] : memref<i64>
    return %24 : i64
  }
}
