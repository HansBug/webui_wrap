from functools import lru_cache
from typing import List

from .webui import _get_client_scripts, get_webui_client


@lru_cache()
def has_adetailer() -> bool:
    return 'adetailer' in _get_client_scripts()


@lru_cache()
def get_adetailer_version():
    client = get_webui_client()
    return client.custom_get('/adetailer/v1/version')['version']


@lru_cache()
def get_adetailer_models() -> List[str]:
    client = get_webui_client()
    return client.custom_get('/adetailer/v1/ad_model')['ad_model']
