import os
import shutil
from contextlib import contextmanager

from .base import BaseImageStorage


class LocalImageStorage(BaseImageStorage):
    def __init__(self, storage_root: str):
        self.storage_root = storage_root
        os.makedirs(self.storage_root, exist_ok=True)

    def _save_file(self, src_filepath: str, path_in_storage: str):
        dst_filepath = os.path.join(self.storage_root, path_in_storage)
        if os.path.dirname(dst_filepath):
            os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
        shutil.copyfile(src_filepath, dst_filepath)

    @contextmanager
    def _load_file(self, path_in_storage: str):
        dst_filepath = os.path.join(self.storage_root, path_in_storage)
        yield dst_filepath
