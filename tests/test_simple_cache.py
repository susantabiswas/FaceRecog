from face_recog.exceptions import InvalidCacheInitializationData
import pytest
from face_recog.simple_cache import SimpleCache


def test_correct_initializer(correct_cache_init_data):
    """Check if a simple cache can be initialized
    with expected data structure i.e., list"""
    for test_case in correct_cache_init_data:
        try:
            ob = SimpleCache(test_case)
        except Exception:
            pytest.fail()


def test_incorrect_initializer(incorrect_cache_init_data):
    """Check if a simple cache can be initialized
    with an incorect data structure"""
    for test_case in incorrect_cache_init_data:
        with pytest.raises(InvalidCacheInitializationData):
            ob = SimpleCache(test_case)


def test_unique_add_data(simple_cache_init_data, simple_cache_data2):
    """ Check if duplicates get added or not"""
    ob = SimpleCache(simple_cache_init_data)
    ob.add_data(simple_cache_data2)
    assert sorted(ob.get_all_data(), key=lambda x: x["name"]) == sorted(
        [
            {"encoding": (-3.4, 0.3, -0.823, 1), "name": "test2"},
            {"encoding": (-3.4, 0.3, -0.823, 1), "name": "test1"},
        ],
        key=lambda x: x["name"],
    )


def test_add_data(simple_cache_init_data, simple_cache_data3):
    """ Check if a data item gets added"""
    ob = SimpleCache(simple_cache_init_data)
    ob.add_data(simple_cache_data3)
    assert sorted(ob.get_all_data(), key=lambda x: x["name"]) == sorted(
        [
            {"encoding": (-3.4, 0.3, -0.823, 1), "name": "test1"},
            {"encoding": (-3.4, 0.3, -0.823, 1), "name": "test3"},
            {"encoding": (-3.4, 0.3, -0.823, 1), "name": "test2"},
        ],
        key=lambda x: x["name"],
    )


def test_get_all_data(simple_cache_init_data):
    """ Check the data from cache"""
    ob = SimpleCache(simple_cache_init_data)
    assert sorted(ob.get_all_data(), key=lambda x: x["name"]) == sorted(
        [
            {"encoding": (-3.4, 0.3, -0.823, 1), "name": "test2"},
            {"encoding": (-3.4, 0.3, -0.823, 1), "name": "test1"},
        ],
        key=lambda x: x["name"],
    )


def test_delete_data(simple_cache_init_data, simple_cache_data3):
    """ Check if delete works"""
    ob = SimpleCache(simple_cache_init_data)
    ob.add_data(simple_cache_data3)
    ob.delete_data(face_id="test1")

    assert sorted(ob.get_all_data(), key=lambda x: x["name"]) == sorted(
        [
            {"encoding": (-3.4, 0.3, -0.823, 1), "name": "test3"},
            {"encoding": (-3.4, 0.3, -0.823, 1), "name": "test2"},
        ],
        key=lambda x: x["name"],
    )
