import os
from .base import Base
from ._z5py import create_group
from .dataset import Dataset


class Group(Base):

    def __init__(self, path, is_zarr, mode):
        super(Group, self).__init__(path, is_zarr, mode)

    @classmethod
    def make_group(cls, path, is_zarr, mode):
        create_group(path, is_zarr, cls.file_modes[mode])
        return cls(path, is_zarr, mode)

    @classmethod
    def open_group(cls, path, is_zarr, mode):
        return cls(path, is_zarr, mode)

    def create_group(self, key):
        if key in self.keys():
            raise RuntimeError("Group %s is already existing" % key)
        path = os.path.join(self.path, key)
        return Group.make_group(path, self.is_zarr, self.mode)

    def __getitem__(self, key):
        if key not in self:
            raise RuntimeError("Key %s does not exist" % key)
        path = os.path.join(self.path, key)
        if self.is_group(key):
            return Group.open_group(path, self.is_zarr, self.mode)
        else:
            return Dataset.open_dataset(path, self._internal_mode)
