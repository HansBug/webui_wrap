import os
from functools import lru_cache

from .base import BaseImageStorage
from .local import LocalImageStorage
from .record import ImageRecorder


@lru_cache()
def load_storage_from_env() -> BaseImageStorage:
    if os.environ.get('LOCAL_IMG_STORAGE_DIR'):
        return LocalImageStorage(os.environ.get('LOCAL_IMG_STORAGE_DIR'))
    else:
        return LocalImageStorage(os.path.abspath('images'))


@lru_cache()
def load_recorder_from_env() -> ImageRecorder:
    if os.environ.get('LOCAL_IMG_STORAGE_DIR'):
        return ImageRecorder(
            storage=load_storage_from_env(),
            root_dir=os.environ.get('LOCAL_IMG_STORAGE_DIR'),
        )
    else:
        return ImageRecorder(
            storage=load_storage_from_env(),
            root_dir=os.path.abspath('images'),
        )
