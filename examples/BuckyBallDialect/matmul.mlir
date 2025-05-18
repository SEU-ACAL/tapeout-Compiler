func.func @buckyball_matmul(%aMemArray: memref<256x256xi8>, 
                            %bMemArray: memref<256x256xi8>, 
                            %cMemArray: memref<256x256xi8>) {
  buckyball.bb_tile_matmul %aMemArray %bMemArray %cMemArray {warpNum = 16 : i8} : 
    memref<256x256xi8> memref<256x256xi8> memref<256x256xi8>
  return
}
