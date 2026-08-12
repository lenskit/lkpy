from lenskit import _accel


def test_parse_json():
    json = """
    {"name": "FOOBIE BLETCH", "tags": ["foo", "bar"]}
    """
    obj = _accel.data.pyon_loads(json)
    assert obj == {"name": "FOOBIE BLETCH", "tags": ["foo", "bar"]}


def test_parse_single_quotes():
    json = """
    {'name': 'FOOBIE BLETCH', 'tags': ['foo', 'bar'], 'count': 7, 'active': false, 'value': 1.0}
    """
    obj = _accel.data.pyon_loads(json)
    assert obj == {
        "name": "FOOBIE BLETCH",
        "tags": ["foo", "bar"],
        "count": 7,
        "active": False,
        "value": 1.0,
    }


def test_parse_escape_dquote():
    json = r"""{"name": "foo\"bob"}"""
    obj = _accel.data.pyon_loads(json)
    assert obj == {"name": 'foo"bob'}


def test_parse_escape_squote():
    json = r"""{"name": 'pe\'taq'}"""
    obj = _accel.data.pyon_loads(json)
    assert obj == {"name": "pe'taq"}


def test_parse_escape_tab():
    json = r"""{"name": '\t'}"""
    obj = _accel.data.pyon_loads(json)
    assert obj == {"name": "\t"}


def test_parse_escape_lf():
    json = r"""{"name": '\n'}"""
    obj = _accel.data.pyon_loads(json)
    assert obj == {"name": "\n"}


def test_parse_escape_cr():
    json = r"""{"name": '\r'}"""
    obj = _accel.data.pyon_loads(json)
    assert obj == {"name": "\r"}


def test_parse_escape_unicode():
    json = r"""{"name": '\u2230'}"""
    obj = _accel.data.pyon_loads(json)
    assert obj == {"name": "\u2230"}
