import os.path
from contextlib import contextmanager
from typing import Optional

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from hbutils.random import random_md5_with_timestamp
from hbutils.system import TemporaryDirectory


class BaseImageStorage:
    def _save_file(self, src_filepath: str, path_in_storage: str):
        raise NotImplementedError

    def put_image(self, image: Image.Image, meta_text: Optional[str] = None):
        image_filename = f'{random_md5_with_timestamp()}.png'
        prefix = image_filename[:8]
        image_dst_path = os.path.join(prefix, image_filename)
        with TemporaryDirectory() as td:
            img_file = os.path.join(td, image_filename)
            if meta_text:
                info = PngInfo()
                info.add_text('parameters', meta_text)
                image.save(img_file, pnginfo=info)
            else:
                image.save(img_file)

            self._save_file(img_file, image_dst_path)

        return image_filename

    @contextmanager
    def _load_file(self, path_in_storage: str):
        raise NotImplementedError

    def get_image(self, image_file: str) -> Image.Image:
        prefix = os.path.splitext(image_file)[0][:8]
        image_dst_path = os.path.join(prefix, image_file)
        with self._load_file(image_dst_path) as imgfile:
            image = Image.open(imgfile)
            image.load()
            return image
