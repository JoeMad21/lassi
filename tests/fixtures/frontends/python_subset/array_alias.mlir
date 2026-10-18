module {
  func.func @mirror(%arg0: memref<?xi64>, %arg1: i64) -> i64 {
    %0 = arith.constant 0 : i64
    %1 = arith.index_cast %0 : i64 to index
    memref.store %arg1, %arg0[%1] : memref<?xi64>
    %2 = arith.constant 0 : i64
    %3 = arith.index_cast %2 : i64 to index
    %4 = memref.load %arg0[%3] : memref<?xi64>
    %5 = arith.constant 0 : index
    %6 = memref.dim %arg0, %5 : memref<?xi64>
    %7 = arith.index_cast %6 : index to i64
    %8 = arith.addi %4, %7 : i64
    return %8 : i64
  }
}
