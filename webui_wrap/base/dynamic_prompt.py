import logging
from functools import lru_cache
from typing import Optional

from .webui import _get_client_scripts


@lru_cache()
def _get_dynamic_prompts_name() -> Optional[str]:
    for name in _get_client_scripts():
        if 'dynamic' in name and 'prompts' in name:
            logging.info(f'Dynamic prompts found, name: {name!r}')
            return name
    logging.error('Dynamic prompts not found.')
    return None


def has_dynamic_prompts() -> bool:
    return bool(_get_dynamic_prompts_name())


def dynamic_prompt_params(
        is_enabled: bool = True,  # "Dynamic Prompts enabled"
        is_combinatorial: bool = False,  # "Combinatorial generation"
        combinatorial_batches: int = 1,  # [1, 10, 1], "Combinatorial batches"
        is_magic_prompt: bool = False,  # "Magic prompt"
        is_feeling_lucky: bool = False,  # "I'm feeling lucky"
        is_attention_grabber: bool = False,  # "Attention grabber"
        min_attention: float = 1.1,  # [-1, 2, 0.1], "Minimum attention"
        max_attention: float = 1.5,  # [-1, 2, 0.1], "Maximum attention"
        magic_prompt_length: int = 100,  # [30, 300, 10], "Max magic prompt length"
        magic_temp_value: float = 0.7,  # [0.1, 3.0, 0.1], "Magic prompt creativity"
        use_fixed_seed: bool = False,  # "Fixed seed"
        unlink_seed_from_prompt: bool = False,  # "Unlink seed from prompt"
        disable_negative_prompt: bool = True,  # "Don't apply to negative prompts"
        enable_jinja_templates: bool = False,  # "Enable Jinja2 templates"
        no_image_generation: bool = False,  # "Don't generate images"
        max_generations: int = 0,
        # [0, 1000, 1], "Max generations (0 = all combinations - the batch count value is ignored)"
        magic_model: Optional[str] = None,
        magic_blocklist_regex: Optional[str] = "",  # "Magic prompt blocklist regex"
):
    _name = _get_dynamic_prompts_name()
    if not _name or not is_enabled:
        return {}
    else:
        return {
            _name: {
                "args": [
                    is_enabled,
                    is_combinatorial,
                    combinatorial_batches,
                    is_magic_prompt,
                    is_feeling_lucky,
                    is_attention_grabber,
                    min_attention,
                    max_attention,
                    magic_prompt_length,
                    magic_temp_value,
                    use_fixed_seed,
                    unlink_seed_from_prompt,
                    disable_negative_prompt,
                    enable_jinja_templates,
                    no_image_generation,
                    max_generations,
                    magic_model,
                    magic_blocklist_regex,
                ]
            }
        }
