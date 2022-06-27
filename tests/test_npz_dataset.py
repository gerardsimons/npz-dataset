import os.path
import shutil

import pytest

from npz_record import NPZDataset, NPZRecord



@pytest.fixture()
def r1():
    path = '/tmp/test_npz_dataset/r1'
    shutil.rmtree(path, ignore_errors=True)
    r = NPZRecord(path)
    r.add(x=1)
    r.add(x=1)
    r.add(x=1)
    return r

@pytest.fixture()
def r2():
    path = '/tmp/test_npz_dataset/r2'
    shutil.rmtree(path, ignore_errors=True)
    r = NPZRecord(path)
    r.add(x=2)
    return r


@pytest.fixture()
def dataset(r1, r2):
    return NPZDataset([r1, r2])


def test_create_dataset(r1, r2):

    ds = NPZDataset([r1, r2])
    assert ds.records == [r1, r2]


def test_len_dataset(dataset):
    assert len(dataset) == 4


def test_save(ds):
    ds.save()
    ds2 = NPZDataset(ds.path)
    ds2.load()
    assert ds2 == ds1
    # ds2 = ds

# def test_iterate_dataset():


