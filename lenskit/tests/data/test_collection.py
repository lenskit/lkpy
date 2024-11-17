import pickle

from pytest import raises

from lenskit.data.collection import UserIDKey, _create_key, project_key


def test_generic_key():
    usk = _create_key(("user_id", "seq_no"), "alphabet", 42)

    assert isinstance(usk, tuple)
    assert usk == ("alphabet", 42)
    assert usk.user_id == "alphabet"  # type: ignore
    assert usk.seq_no == 42  # type: ignore


def test_pickle_generic():
    usk = _create_key(("user_id", "seq_no"), "alphabet", 42)

    bs = pickle.dumps(usk)
    usk2 = pickle.loads(bs)

    assert isinstance(usk2, tuple)
    assert usk2 == usk


def test_project_key():
    usk = _create_key(("user_id", "seq_no"), "alphabet", 42)

    uk = project_key(usk, UserIDKey)
    assert isinstance(uk, UserIDKey)
    assert uk.user_id == "alphabet"
    assert uk == ("alphabet",)


def test_project_missing_fails():
    usk = _create_key(("seq_no",), 42)

    with raises(TypeError, match="missing field"):
        project_key(usk, UserIDKey)
