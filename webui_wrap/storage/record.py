import io
import json
import logging
import os.path
import time
from functools import lru_cache
from threading import Lock
from typing import Optional, List
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from PIL import Image
from hbutils.string import plural_word
from huggingface_hub import hf_hub_download
from imgutils.sd import parse_sdmeta_from_text
from imgutils.tagging import get_wd14_tags
from imgutils.tagging.wd14 import MODEL_NAMES

from .base import BaseImageStorage

_TAGGER_MODEL = 'SwinV2_v3'


def _value_safe(x):
    if isinstance(x, (type(None), int, float, str)):
        return x
    else:
        return json.dumps(x)


@lru_cache()
def _load_tags_database():
    df_tags = pd.read_csv(hf_hub_download(
        repo_id='deepghs/wd14_tagger_with_embeddings',
        repo_type='model',
        filename=f'{MODEL_NAMES[_TAGGER_MODEL]}/tags_info.csv',
    ))
    df_tags = df_tags.replace(np.NaN, None)
    df_tags = df_tags[df_tags['category'].isin({0, 4})]

    retval = {}
    for item in df_tags.to_dict('records'):
        for name in json.loads(item['aliases']):
            retval[name] = item

    return retval


@lru_cache()
def _load_general_tags_info():
    df_general_tags = pd.read_csv(hf_hub_download(
        repo_id='deepghs/tags_meta',
        repo_type='dataset',
        filename='general_tags.csv',
    ))
    df_general_tags = df_general_tags.replace(np.NaN, None)
    return {item['name']: item for item in df_general_tags.to_dict('records')}


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
                image,
                fmt=('rating', 'general', 'character', 'embedding'),
                model_name=_TAGGER_MODEL,
            )
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

    def query_with_tags(self, tags: List[str], neg_tags: List[str]) -> List[Image.Image]:
        _db_tags = _load_tags_database()
        query_tags, query_neg_tags = [], []
        for tag in tags:
            if tag not in _db_tags:
                logging.warning(f'Tag {tag!r} unrecognizable, it will be ignored.')
            else:
                query_tags.append(_db_tags[tag]['name'])
        for tag in neg_tags:
            if tag not in _db_tags:
                logging.warning(f'Negative tag {tag!r} recognizable, it will be ignored.')
            else:
                query_neg_tags.append(_db_tags[tag]['name'])

        logging.info(f'Querying with tags: {query_tags!r} and negative tags: {query_neg_tags!r} ...')
        df_query = self._df_records
        for tag in query_tags:
            df_query = df_query[df_query['tags'].str.contains(f' {tag} ', regex=False)]
        for tag in query_neg_tags:
            df_query = df_query[~df_query['tags'].str.contains(f' {tag} ', regex=False)]

        return [self.image_storage.get_image(filename) for filename in df_query['filename']]

    def list_tags(self):
        return self._df_tags.to_dict('records')

    def get_tag_info(self, tag: str):
        if tag in self._d_tags:
            count = self._d_tags[tag]['count']
        else:
            count = 0
        tag_category = _load_tags_database()[tag]['category']

        with io.StringIO() as sf:
            print(f'# Tag: {tag}', file=sf)
            print(f'', file=sf)

            danbooru_wiki_url = f'https://safebooru.donmai.us/wiki_pages/{quote_plus(tag)}'
            print(f'Tag category: {"Character" if tag_category == 4 else "General"}', file=sf)
            print(f'', file=sf)
            print(f'Current count: {plural_word(count, "image")}', file=sf)
            print(f'', file=sf)
            print(f'Danbooru wiki: [{tag} - wiki]({danbooru_wiki_url})', file=sf)
            print(f'', file=sf)

            if tag_category == 0:
                general_tag_info = _load_general_tags_info()[tag]

                alias_names = json.loads(general_tag_info['aliases'])
                other_names = json.loads(general_tag_info['other_names'])
                if alias_names or other_names:
                    print(f'## Aliases', file=sf)
                    print(f'', file=sf)
                    if alias_names:
                        print(f'Alias names: {", ".join([f"`{t}`" for t in alias_names])}', file=sf)
                        print(f'', file=sf)
                    if other_names:
                        print(f'Other names: {", ".join([f"`{t}`" for t in other_names])}', file=sf)
                        print(f'', file=sf)

                print('## Translation', file=sf)
                print(f'', file=sf)
                print(f'### English - {general_tag_info["en_tag"]}', file=sf)
                print(f'', file=sf)
                print(f'{general_tag_info["en_desc"]}', file=sf)
                print(f'', file=sf)
                print(f'### Chinese - {general_tag_info["zh_tag"]}', file=sf)
                print(f'', file=sf)
                print(f'{general_tag_info["zh_desc"]}', file=sf)
                print(f'', file=sf)
                print(f'### Japanese - {general_tag_info["jp_tag"]}', file=sf)
                print(f'', file=sf)
                print(f'{general_tag_info["jp_desc"]}', file=sf)
                print(f'', file=sf)

                if general_tag_info['wiki_desc']:
                    print('## Raw Wiki Text', file=sf)
                    print(f'', file=sf)
                    print(f'{general_tag_info["wiki_desc"]}', file=sf)
                    print(f'', file=sf)

            return sf.getvalue()
