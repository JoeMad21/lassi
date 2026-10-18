module {
  func.func @saxpy_sqrt(%arg0: f64, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) {
    %0 = arith.constant 0 : index
    %1 = memref.dim %arg3, %0 : memref<?xf64>
    %2 = arith.constant 0 : index
    %3 = arith.constant 1 : index
    scf.parallel (%4) = (%2) to (%1) step (%3) {
      %5 = memref.load %arg1[%4] : memref<?xf64>
      %6 = arith.mulf %arg0, %5 : f64
      %7 = memref.load %arg2[%4] : memref<?xf64>
      %8 = math.sqrt %7 : f64
      %9 = arith.addf %6, %8 : f64
      memref.store %9, %arg3[%4] : memref<?xf64>
      scf.yield
    }
    return
  }
}
