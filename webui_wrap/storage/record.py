import json
import os.path
import time
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from imgutils.sd import parse_sdmeta_from_text
from imgutils.tagging import get_wd14_tags

from .base import BaseImageStorage


def _value_safe(x):
    if isinstance(x, (type(None), int, float, str)):
        return x
    else:
        return json.dumps(x)


class ImageRecorder:
    def __init__(self, storage: BaseImageStorage, root_dir: str):
        self.image_storage = storage
        self._root_dir = root_dir
        os.makedirs(self._root_dir, exist_ok=True)

        self._records_file = os.path.join(self._root_dir, 'records.parquet')
        self._records = []
        self._df_records = pd.DataFrame(self._records)

        self._tags_file = os.path.join(self._root_dir, 'tags.parquet')
        self._d_tags = {}
        self._df_tags = pd.DataFrame(list(self._d_tags.values()))

        self._has_untransed_data = False
        self._lock = Lock()
        self._sync_from_local()

    def _sync_from_local(self):
        if os.path.exists(self._records_file):
            self._df_records = pd.read_parquet(self._records_file)
            self._df_records = self._df_records.replace(np.NaN, None)
        else:
            self._df_records = pd.DataFrame([])
        self._records = self._df_records.to_dict('records')

        if os.path.exists(self._tags_file):
            self._df_tags = pd.read_parquet(self._tags_file)
            self._df_tags = self._df_tags.replace(np.NaN, None)
        else:
            self._df_tags = pd.DataFrame([])
        self._d_tags = {item['tag']: item for item in self._df_tags.to_dict('records')}
        self._has_untransed_data = False

    def _sync_dataframes(self):
        if self._has_untransed_data:
            self._df_records = pd.DataFrame(self._records)
            self._df_records = self._df_records.sort_values(by=['created_at'], ascending=[False])
            self._df_tags = pd.DataFrame(list(self._d_tags.values()))
            self._df_tags = self._df_tags.sort_values(by=['count', 'tag', 'type'], ascending=[False, True, True])
            self._has_untransed_data = False

    def _save_to_local(self):
        self._sync_dataframes()
        self._df_records.to_parquet(self._records_file, engine='pyarrow', index=False)
        self._df_tags.to_parquet(self._tags_file, engine='pyarrow', index=False)

    def put_image(self, image: Image.Image, meta_text: Optional[str] = None):
        with self._lock:
            ratings, general, character, embedding = get_wd14_tags(
                image, fmt=('rating', 'general', 'character', 'embedding'))
            rs = np.array(list(ratings.keys()))
            vs = np.array([ratings.get(r, 0.0) for r in rs])
            rating = str(rs[np.argmax(vs)].item())

            metainfo = parse_sdmeta_from_text(meta_text or image.info.get('parameters'))
            filename = self.image_storage.put_image(image, meta_text)

            self._records.append({
                'filename': filename,
                'rating': rating,
                'tags': ' '.join(['', *general.keys(), *character.keys(), '']),
                'width': image.width,
                'height': image.height,
                'prompt': metainfo.prompt,
                'neg_prompt': metainfo.neg_prompt,
                'created_at': time.time(),
                **{key: _value_safe(value) for key, value in metainfo.parameters.items()},
            })
            tags_pairs = [
                *[(tag, 'general') for tag in general.keys()],
                *[(tag, 'character') for tag in character.keys()],
            ]
            for tag, tag_type in tags_pairs:
                if tag not in self._d_tags:
                    self._d_tags[tag] = {'tag': tag, 'type': tag_type, 'count': 0}
                self._d_tags[tag]['count'] += 1
            self._has_untransed_data = True
            return filename

    def save(self):
        with self._lock:
            self._save_to_local()
