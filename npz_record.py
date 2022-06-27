import glob
import json
import os
import traceback
from collections import UserDict, defaultdict
from functools import lru_cache
from typing import List

import numpy as np


class ImmutableKeysDict(UserDict):

    def __init__(self, _dict):
        super().__init__()
        self._dict = _dict

    def __setitem__(self, key, value):
        if key not in self._dict:
            raise ValueError("Cannot add new keys to ImmutableKeysDict")
        self._dict[key] = value

    # TODO: There must be some smarter way to reroute all of these methods to the _dict
    def __getitem__(self, item):
        return self._dict[item]

    def __contains__(self, item):
        return item in self._dict

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()


class NPZRecordError(Exception):
    pass


class NPZRecord:
    """
    An NPZ Record acts as a wrapper around the numpy NPZ file format. It is able to persist **kwarg like data
    compatible with NPZ to disk, and read it back into memory.
    """

    def __init__(self, path, keys: List[str]=None, defaults: dict=None):
        """

        @param path:
        @param keys: If passed we use this to verify
        @param defaults:
        """
        if keys is None:
            keys = []
        self.path = path
        # TODO: Check path is writable directory
        # We do not use np.array because it can be non rectangular
        self.keys = keys

        if defaults is None:
            defaults = {}
        self.strict = len(self.keys)
        if self.strict:
            self._kwarg = ImmutableKeysDict({k: [] for k in keys})
        else:
            self._kwarg = defaultdict(list)
        # for k in keys:
        #     # TODO: Raise error when no default value set, but no
        #     if k not in defaults:
        #         defaults[k] = []
        #     self._defaults = ImmutableKeysDict({k: [] for k in keys})
        # else:
        #     self._defaults = defaults
        self._defaults = ImmutableKeysDict(defaults)
        self._len = 0

    def add(self, **kwarg):

        if self.strict:
            for k in self._kwarg.keys():
                try:
                    new_val = kwarg.pop(k) # Cannot use default as default since it may not exist
                    if new_val is None:
                        new_val = self._defaults[k]
                    self._kwarg[k].append(new_val)

                except KeyError as e:
                    raise NPZRecordError(f'No value passed for {k}, and no default value set')
            if len(kwarg):
                raise KeyError(f"NPZRecord add received unknown keys {kwarg.keys()}")
            self._len += 1
        else:
            for k, v in kwarg.items():
                self._kwarg[k].append(v)
            self._len += 1

    def save(self):
        # todo: compress only when flag is set
        # Convert all to np.arrays
        for k, v in self._kwarg.items():
            self._kwarg[k] = np.array(v, dtype=object)
        print(f"Saving to {self.path} : ", len(self))
        np.savez_compressed(self.path, **self._kwarg, dtype=object)

    def load(self):
        npz_kwargs = np.load(self.path + ".npz", allow_pickle=True)

        self._kwarg = ImmutableKeysDict({k: np.array(v) for k, v in npz_kwargs.items() if k != 'dtype'})
        self._len = max((len(x) for x in self._kwarg.values()))
        # for k, v in npz_kwargs.items():

        # print(dict(self._kwarg.items()))

    def __getitem__(self, x):
        return self._kwarg[x]

    def __eq__(self, other):
        print(self._kwarg.keys(), other._kwarg.keys())
        if self._kwarg.keys() != other._kwarg.keys():
            return False
        for k, v in self._kwarg.items():
            try:
                for vv, ovv in zip(self[k], other[k]):
                    if not np.array_equal(vv, ovv):
                        return False
            except KeyError as e:
                return False

        return True

    def __len__(self):
        return self._len

    def __str__(self):
        repl = "\n"
        vals = [f'{k}={str(v).replace(repl, " ")}' for k, v in self._kwarg.items()]
        return f"NPZRecord size={len(self)} {'; '.join(vals)}"

    def __getattr__(self, name):
        if name in self._kwarg:
            return self._kwarg[name]
        else:
            raise NotImplementedError(f"NPZRecord has no key or attribute called {name}")

    def __iter__(self):
        self._iter_count = 0
        return self

    def __next__(self):
        if self._iter_count < self._len:
            # ret = {k : v[self._iter_count] for k, v in self._kwarg.items()} # Return as dict
            ret = [v[self._iter_count] for v in self._kwarg.values()] # Return as list
            self._iter_count += 1
            return ret
        raise StopIteration


META_FILE_NAME = 'npz_manifest.json'

class NPZDataset:
    """
    NPZDataset wraps around a collection of NPZRecords and some metadata.
    """
    def __init__(self, path):
        self.path = path
        self._records = []
        self._find_records()

    @property
    def records(self):
        return self._records

    def __len__(self):
        # TODO: Cache len result
        return sum((len(x) for x in self._records))

    def _find_records(self):
        subdirs = glob.glob(os.path.join(self.path, "*/"))
        for subdir in subdirs:
            try:
                r = NPZRecord(subdir)
                self._records.append(r)
            except NotImplementedError as e:
                traceback.print_exc()

    def _rel_path(self, p):
        return os.path.join(self.path, p)


    def _load_meta(self):
        # Find meta file
        try:
            with open(self._meta_file_path(), 'r') as fp:
                self._meta = json.load(fp)
        except FileNotFoundError:
            print("Woops")
            # raise ValueError(f"No {MANIFEST_FILE_NAME} file found in {self.path}")

    def _meta_file_path(self):
        return self._rel_path(META_FILE_NAME)

    def save(self):
        # Save meta data
        for r in self.records:
            r.save()

    def add_meta(self, k, v):
        if k in self._meta:
            raise ValueError("Meta key already exists")
        self._meta[k] = v

    def load(self):
        # Load meta data
        with open(self._meta_file_path()) as fp:
            self.meta = json.load(fp)
        for r in self.records:
            r.load()

    def __eq__(self, other):
        if len(other.records) != len(self.records):
            return False
        else:
            
