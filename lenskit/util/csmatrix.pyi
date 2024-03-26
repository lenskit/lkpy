import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array

class CSMatrix:
    nrows: int
    ncols: int
    nnz: int

    rowptr: npt.NDArray[np.int32]
    colind: npt.NDArray[np.int32]
    values: npt.NDArray[np.float64]

    def __init__(
        self,
        nr: int,
        nc: int,
        rps: npt.NDArray[np.int32],
        cis: npt.NDArray[np.int32],
        vs: npt.NDArray[np.float64],
    ): ...
    def row_ep(self, row: int) -> tuple[int, int]: ...
    @staticmethod
    def from_scipy(matrix: csr_array) -> CSMatrix: ...
