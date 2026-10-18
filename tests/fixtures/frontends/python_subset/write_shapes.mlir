module {
  memref.global "private" constant @lassi_io_name0 : memref<5xi8> = dense<[110, 111, 114, 109, 0]>
  memref.global "private" constant @lassi_io_name1 : memref<5xi8> = dense<[103, 114, 105, 100, 0]>
  memref.global "private" constant @lassi_io_name2 : memref<5xi8> = dense<[108, 97, 115, 116, 0]>
  func.func private @lassi_io_write_f64(i64, memref<?xi8>, memref<?xi64>, memref<?xf64>)
  func.func private @lassi_io_write_i32(i64, memref<?xi8>, memref<?xi64>, memref<?xi32>)
  func.func @main() {
    %0 = memref.alloca() : memref<i64>
    %1 = arith.constant 1 : i64
    %2 = arith.index_cast %1 : i64 to index
    %3 = memref.alloc(%2) : memref<?xf64>
    %4 = arith.constant 0 : index
    %5 = arith.constant 1 : index
    %6 = arith.constant 0.0 : f64
    scf.for %7 = %4 to %2 step %5 {
      memref.store %6, %3[%7] : memref<?xf64>
    }
    %8 = arith.constant 6 : i64
    %9 = arith.index_cast %8 : i64 to index
    %10 = memref.alloc(%9) : memref<?xi32>
    %11 = arith.constant 0 : index
    %12 = arith.constant 1 : index
    %13 = arith.constant 0 : i32
    scf.for %14 = %11 to %9 step %12 {
      memref.store %13, %10[%14] : memref<?xi32>
    }
    %15 = arith.constant 1 : i64
    %16 = memref.get_global @lassi_io_name0 : memref<5xi8>
    %17 = memref.cast %16 : memref<5xi8> to memref<?xi8>
    %18 = memref.alloca() : memref<0xi64>
    %19 = memref.cast %18 : memref<0xi64> to memref<?xi64>
    call @lassi_io_write_f64(%15, %17, %19, %3) : (i64, memref<?xi8>, memref<?xi64>, memref<?xf64>) -> ()
    %20 = arith.constant 2 : i64
    %21 = memref.get_global @lassi_io_name1 : memref<5xi8>
    %22 = memref.cast %21 : memref<5xi8> to memref<?xi8>
    %23 = arith.constant 2 : i64
    %24 = arith.constant 3 : i64
    %25 = memref.alloca() : memref<2xi64>
    %26 = arith.constant 0 : index
    memref.store %23, %25[%26] : memref<2xi64>
    %27 = arith.constant 1 : index
    memref.store %24, %25[%27] : memref<2xi64>
    %28 = memref.cast %25 : memref<2xi64> to memref<?xi64>
    call @lassi_io_write_i32(%20, %22, %28, %10) : (i64, memref<?xi8>, memref<?xi64>, memref<?xi32>) -> ()
    %29 = arith.constant 2 : i64
    %30 = arith.index_cast %29 : i64 to index
    %31 = arith.constant 0 : index
    %32 = arith.constant 1 : index
    scf.for %33 = %31 to %30 step %32 {
      %34 = arith.index_cast %33 : index to i64
      memref.store %34, %0[] : memref<i64>
      %35 = memref.load %0[] : memref<i64>
      %36 = memref.load %0[] : memref<i64>
      %37 = arith.index_cast %36 : i64 to index
      %38 = arith.trunci %35 : i64 to i32
      memref.store %38, %10[%37] : memref<?xi32>
      memref.alloca_scope {
        %39 = arith.constant 3 : i64
        %40 = memref.get_global @lassi_io_name2 : memref<5xi8>
        %41 = memref.cast %40 : memref<5xi8> to memref<?xi8>
        %42 = arith.constant 6 : i64
        %43 = memref.alloca() : memref<1xi64>
        %44 = arith.constant 0 : index
        memref.store %42, %43[%44] : memref<1xi64>
        %45 = memref.cast %43 : memref<1xi64> to memref<?xi64>
        func.call @lassi_io_write_i32(%39, %41, %45, %10) : (i64, memref<?xi8>, memref<?xi64>, memref<?xi32>) -> ()
      }
    }
    memref.dealloc %3 : memref<?xf64>
    memref.dealloc %10 : memref<?xi32>
    return
  }
}
