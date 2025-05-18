  const uint32_t aSpAddrStart = 0;
  const uint32_t bSpAddrStart = spadRows - 
      (dimN_MergeTileLen * dimN_MetaTileLen) * (dimN_MergeTileLen * dimN_MetaTileLen) / dim;
  const uint32_t cSpAddrStart = 1 << (addrLen - 1);
