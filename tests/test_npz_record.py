import numpy as np
import pytest

from npz_record import NPZRecord, ImmutableKeysDict


def test_npz_record_create():
    rec = NPZRecord('test', keys=['a', 'b', 'c'])
    assert len(rec) == 0



def test_save_load():
    r = NPZRecord('test', keys=['ecg', 'qrs'])
    for i in range(10):
        ecg = np.array(list(range(i + 1)))
        # print(ecg)

        r.add(ecg=ecg, qrs=np.array([0, 1]))
        # r.add(ecg=np.array(list(range(i))), qrs=np.array([i]))

    print("R -----")
    print(r)
    print()
    r.save()

    r2 = NPZRecord('test')
    # r.add(x=21)
    r2.load()

    assert r == r2


def test_get_attr():

    r = NPZRecord('test2', keys=['x', 'ys'])
    r.add(x=1, ys=[1,2])
    r.save()
    # r = NPZRecord('test2', 'x', 'ys')
    r = NPZRecord('test2', keys=['x', 'ys'])
    r.load()
    # print(r)
    assert r.x == 1
    assert np.equal(r.ys, [1, 2])


def try_irregular_shape():
    r = NPZRecord('test2', keys=['x', 'ys'], defaults={'x': -1, 'ys': 0})
    # r = NPZRecord('test2', keys=['x', 'ys'], defaults={'x': -1})
    for i in range(5):

        if i % 3 == 0:
            r.add(ys=-13.2)
        elif i % 2 == 0:
            r.add(x=2, ys=42.2)
        else:
            r.add(x=1)
    r.save()
    # r = NPZRecord('test2', 'x', 'ys')
    r2 = NPZRecord('test2', keys=['x', 'ys'])
    r2.load()
    print(r2)
    print(r2.x)
    print(r2.ys)
    print(r == r2)

def test_iterate():
    r = NPZRecord('test2', keys=['a', 'b'])

    for i in range(3):
        r.add(a=i, b=[i,i+1,i+2])

    for i, x in enumerate(r):
        assert len(x) == 2
        assert(x[0] == i)
        assert len(x[1]) == 3
        assert x[1][0] == i
        assert x[1][1] == i+1
        assert x[1][2] == i+2


def test_immutable_dict():
    a_vals = [1,2,3,4]
    d = {'a': a_vals, 'b': 5}
    print(d)
    d = ImmutableKeysDict(d)

    with pytest.raises(ValueError):
        d['c'] = 12


def test_non_strict():
    # The simplest way to use it, does not check for keys beforehand
    r = NPZRecord('test_non_strict')

    r.add(ecg=[123,4,5], x=42, y=12.120)
    assert len(r) == 1
    r.add(ecg=[100])
    assert len(r) == 2

    r2 = NPZRecord('test_non_strict')
    r2.load()
    print(r2)
    print(len(r2))
    print(r == r2)

test_iterate()