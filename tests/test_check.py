import pytest

from lenskit.check import check_value


def test_check_value_passes():
    check_value(True, "value should be true")
    # it should complete successfully!


def test_check_value_fails():
    with pytest.raises(ValueError):
        check_value(False, "value should be true")


def test_check_meaningful_value_fails():
    with pytest.raises(ValueError):
        check_value(5 < 4, "five should be less than four")


def test_check_meaningful_value_succeeds():
    check_value(3 < 4, "three should be less than four")
    # it should complete successfully
