from numpy.random import SeedSequence

from hypothesis import given
from hypothesis import strategies as st

from lenskit.util.random import make_seed


@given(
    st.one_of(
        st.integers(min_value=0),
        st.binary(),
        st.uuids(),
        st.text(),
        st.lists(st.integers(min_value=0)),
    )
)
def test_make_seed(seed):
    ss = make_seed(seed)
    assert isinstance(ss, SeedSequence)
    state = ss.generate_state(1)
    assert len(state) == 1
    assert state[0]
