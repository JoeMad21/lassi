module {
  memref.global "private" constant @lassi_io_name0 : memref<2xi8> = dense<[121, 0]>
  func.func private @lassi_io_read_f64(i64) -> memref<?xf64>
  func.func private @lassi_io_write_f64(i64, memref<?xi8>, memref<?xi64>, memref<?xf64>)
  func.func @double(%arg0: i64, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %0 = memref.alloca() : memref<i64>
    %1 = arith.index_cast %arg0 : i64 to index
    %2 = arith.constant 0 : index
    %3 = arith.constant 1 : index
    scf.for %4 = %2 to %1 step %3 {
      %5 = arith.index_cast %4 : index to i64
      memref.store %5, %0[] : memref<i64>
      %6 = arith.constant 2.0 : f64
      %7 = memref.load %0[] : memref<i64>
      %8 = arith.index_cast %7 : i64 to index
      %9 = memref.load %arg1[%8] : memref<?xf64>
      %10 = arith.mulf %6, %9 : f64
      %11 = memref.load %0[] : memref<i64>
      %12 = arith.index_cast %11 : i64 to index
      memref.store %10, %arg2[%12] : memref<?xf64>
    }
    return
  }
  func.func @main() {
    %0 = memref.alloca() : memref<i64>
    %1 = arith.constant 1 : i64
    %2 = call @lassi_io_read_f64(%1) : (i64) -> memref<?xf64>
    %3 = arith.constant 0 : index
    %4 = memref.dim %2, %3 : memref<?xf64>
    %5 = arith.index_cast %4 : index to i64
    memref.store %5, %0[] : memref<i64>
    %6 = memref.load %0[] : memref<i64>
    %7 = arith.index_cast %6 : i64 to index
    %8 = memref.alloc(%7) : memref<?xf64>
    %9 = arith.constant 0 : index
    %10 = arith.constant 1 : index
    %11 = arith.constant 0.0 : f64
    scf.for %12 = %9 to %7 step %10 {
      memref.store %11, %8[%12] : memref<?xf64>
    }
    %13 = memref.load %0[] : memref<i64>
    call @double(%13, %2, %8) : (i64, memref<?xf64>, memref<?xf64>) -> ()
    %14 = arith.constant 2 : i64
    %15 = memref.get_global @lassi_io_name0 : memref<2xi8>
    %16 = memref.cast %15 : memref<2xi8> to memref<?xi8>
    %17 = memref.load %0[] : memref<i64>
    %18 = memref.alloca() : memref<1xi64>
    %19 = arith.constant 0 : index
    memref.store %17, %18[%19] : memref<1xi64>
    %20 = memref.cast %18 : memref<1xi64> to memref<?xi64>
    call @lassi_io_write_f64(%14, %16, %20, %8) : (i64, memref<?xi8>, memref<?xi64>, memref<?xf64>) -> ()
    memref.dealloc %8 : memref<?xf64>
    return
  }
}
