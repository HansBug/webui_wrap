import os
from functools import lru_cache

from .base import BaseImageStorage
from .local import LocalImageStorage


@lru_cache()
def load_storage_from_env() -> BaseImageStorage:
    if os.environ.get('LOCAL_IMG_STORAGE_DIR'):
        return LocalImageStorage(os.environ.get('LOCAL_IMG_STORAGE_DIR'))
    else:
        return LocalImageStorage(os.path.abspath('images'))
