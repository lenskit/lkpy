from lenskit.data.matrix import SparseRowArray
from lenskit.data.types import NPMatrix
from lenskit.parallel import AccelTask

def train_explicit_matrix(
    matrix: SparseRowArray,
    this: NPMatrix,
    other: NPMatrix,
    reg: float,
) -> AccelTask[float]: ...
def train_implicit_matrix(
    matrix: SparseRowArray,
    this: NPMatrix,
    other: NPMatrix,
    otor: NPMatrix,
) -> AccelTask[float]: ...
