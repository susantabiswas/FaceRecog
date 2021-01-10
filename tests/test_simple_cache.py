from face_recog.exceptions import InvalidCacheInitializationData
import pytest
from face_recog.simple_cache import SimpleCache

def test_correct_initializer():
    """ Check if a simple cache can be initialized
    with expected data structure i.e., list"""
    ob = SimpleCache([1, 2])
    assert ob.get_all_data == [1, 2]

def test_correct_initializer():
    """ Check if a simple cache can be initialized
    with an incoorect data structure"""
    with pytest.raises(InvalidCacheInitializationData):
        ob = SimpleCache("1, 2")
    
def test_unique_add_data():
    """ Check if duplicates get added or not"""
    ob = SimpleCache([1, 2])
    ob.add_data(1)
    assert ob.get_all_data() == [1, 2]
 
def test_add_data():
    """ Check if a data item gets added"""
    ob = SimpleCache([1, 2])
    ob.add_data(11)
    assert ob.get_all_data() == [1, 2, 11]

def test_get_all_data():
    """ Check the data from cache"""
    ob = SimpleCache([1, 2])
    assert ob.get_all_data() == [1, 2]