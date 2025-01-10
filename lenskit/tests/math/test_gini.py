import numpy as np

from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nph
from pytest import approx, mark

from lenskit.stats import gini


def test_gini_uniform():
    ones = np.ones(100)
    assert gini(ones) == approx(0)


def test_completely_unequal():
    arr = np.zeros(10000)
    arr[0] = 1000
    assert gini(arr) == approx(1, abs=0.001)


@mark.skipif(np.version.version < "2.0", reason="NumPy too old")
@given(
    nph.arrays(
        st.one_of(nph.floating_dtypes(sizes=[32, 64]), nph.integer_dtypes()),
        nph.array_shapes(max_dims=1, min_side=2),
        elements={"allow_nan": False, "allow_infinity": False, "min_value": 0},
    )
)
def test_random_values(xs):
    # max value here, large max_values don't work on elements
    assume(np.all(xs < 1e12))
    assume(np.any(xs > 0))

    g = gini(xs)
    assert g >= 0
    assert g <= 1

    # The gini should be equal to the difference between ideal area and actual
    # area under the curve.  We can approximate this with NumPy trapezoid.
    n = len(xs)
    ideal = np.sum(xs) * len(xs) * 0.5
    xss = np.zeros(n + 1, dtype="f8")
    xss[1:] = np.cumsum(np.sort(xs))
    actual = np.trapezoid(xss)

    print(g, actual, ideal)
    # we use max just to deal with extremely small values
    assert g == approx(max((ideal - actual) / ideal, 0), abs=0.001)
