from enum import Enum
from functools import lru_cache
from typing import Dict, List, Tuple

from .webui import auto_init_webui, get_webui_client


@lru_cache()
def get_cn_modules():
    auto_init_webui()
    client = get_webui_client()
    return client.controlnet_module_list()


@lru_cache()
def get_cn_models():
    auto_init_webui()
    client = get_webui_client()
    return client.controlnet_model_list()


preprocessor_filters = {
    "All": "none",
    "Canny": "canny",
    "Depth": "depth_midas",
    "NormalMap": "normal_bae",
    "OpenPose": "openpose_full",
    "MLSD": "mlsd",
    "Lineart": "lineart_standard (from white bg & black line)",
    "SoftEdge": "softedge_pidinet",
    "Scribble/Sketch": "scribble_pidinet",
    "Segmentation": "seg_ofade20k",
    "Shuffle": "shuffle",
    "Tile/Blur": "tile_resample",
    "Inpaint": "inpaint_only",
    "InstructP2P": "none",
    "Reference": "reference_only",
    "Recolor": "recolor_luminance",
    "Revision": "revision_clipvision",
    "T2I-Adapter": "none",
    "IP-Adapter": "ip-adapter_clip_sd15",
    "Instant_ID": "instant_id",
    "SparseCtrl": "none",
}

preprocessor_filters_aliases = {
    'instructp2p': ['ip2p'],
    'segmentation': ['seg'],
    'normalmap': ['normal'],
    't2i-adapter': ['t2i_adapter', 't2iadapter', 't2ia'],
    'ip-adapter': ['ip_adapter', 'ipadapter'],
    'scribble/sketch': ['scribble', 'sketch'],
    'tile/blur': ['tile', 'blur'],
    'openpose': ['openpose', 'densepose'],
}

preprocessor_aliases = {
    "invert": "invert (from white bg & black line)",
    "lineart_standard": "lineart_standard (from white bg & black line)",
    "lineart": "lineart_realistic",
    "color": "t2ia_color_grid",
    "clip_vision": "t2ia_style_clipvision",
    "pidinet_sketch": "t2ia_sketch_pidi",
    "depth": "depth_midas",
    "normal_map": "normal_midas",
    "hed": "softedge_hed",
    "hed_safe": "softedge_hedsafe",
    "pidinet": "softedge_pidinet",
    "pidinet_safe": "softedge_pidisafe",
    "segmentation": "seg_ufade20k",
    "oneformer_coco": "seg_ofcoco",
    "oneformer_ade20k": "seg_ofade20k",
    "pidinet_scribble": "scribble_pidinet",
    "inpaint": "inpaint_global_harmonious",
    "anime_face_segment": "seg_anime_face",
    "densepose": "densepose (pruple bg & purple torso)",
    "densepose_parula": "densepose_parula (black bg & blue torso)",
    "te_hed": "softedge_teed",
}

ui_preprocessor_keys = ['none', preprocessor_aliases['invert']]
ui_preprocessor_keys += sorted([preprocessor_aliases.get(k, k)
                                for k in get_cn_modules()
                                if preprocessor_aliases.get(k, k) not in ui_preprocessor_keys])


class StableDiffusionVersion(Enum):
    """The version family of stable diffusion model."""

    UNKNOWN = 0
    SD1x = 1
    SD2x = 2
    SDXL = 3

    @staticmethod
    def detect_from_model_name(model_name: str) -> "StableDiffusionVersion":
        """Based on the model name provided, guess what stable diffusion version it is.
        This might not be accurate without actually inspect the file content.
        """
        if any(f"sd{v}" in model_name.lower() for v in ("14", "15", "16")):
            return StableDiffusionVersion.SD1x

        if "sd21" in model_name or "2.1" in model_name:
            return StableDiffusionVersion.SD2x

        if "xl" in model_name.lower():
            return StableDiffusionVersion.SDXL

        return StableDiffusionVersion.UNKNOWN

    def encoder_block_num(self) -> int:
        if self in (StableDiffusionVersion.SD1x, StableDiffusionVersion.SD2x, StableDiffusionVersion.UNKNOWN):
            return 12
        else:
            return 9  # SDXL

    def controlnet_layer_num(self) -> int:
        return self.encoder_block_num() + 1

    def is_compatible_with(self, other: "StableDiffusionVersion") -> bool:
        """ Incompatible only when one of version is SDXL and other is not. """
        return (
                any(v == StableDiffusionVersion.UNKNOWN for v in [self, other]) or
                sum(v == StableDiffusionVersion.SDXL for v in [self, other]) != 1
        )


def select_control_type(
        control_type: str,
        sd_version: StableDiffusionVersion = StableDiffusionVersion.UNKNOWN,
        cn_models: Dict = None,  # Override or testing
) -> Tuple[List[str], List[str], str, str]:
    default_option = preprocessor_filters[control_type]
    pattern = control_type.lower()
    preprocessor_list = ui_preprocessor_keys
    if cn_models is None:
        all_models = get_cn_models()
    else:
        all_models = list(cn_models.keys())

    if not any(x.lower() == 'none' for x in all_models):
        all_models = ['None', *all_models]

    if pattern == "all":
        return (
            preprocessor_list,
            all_models,
            'none',  # default option
            "None"  # default model
        )
    filtered_preprocessor_list = [
        x
        for x in preprocessor_list
        if ((
                    pattern in x.lower() or
                    any(a in x.lower() for a in preprocessor_filters_aliases.get(pattern, [])) or
                    x.lower() == "none"
            ) and (
                sd_version.is_compatible_with(StableDiffusionVersion.detect_from_model_name(x))
            ))
    ]
    if pattern in ["canny", "lineart", "scribble/sketch", "mlsd"]:
        filtered_preprocessor_list += [
            x for x in preprocessor_list if "invert" in x.lower()
        ]
    if pattern in ["sparsectrl"]:
        filtered_preprocessor_list += [
            x for x in preprocessor_list if "scribble" in x.lower()
        ]
    filtered_model_list = [
        model for model in all_models
        if model.lower() == "none" or
           ((
                    pattern in model.lower() or
                    any(a in model.lower() for a in preprocessor_filters_aliases.get(pattern, []))
            ) and (
                sd_version.is_compatible_with(StableDiffusionVersion.detect_from_model_name(model))
            ))
    ]
    assert len(filtered_model_list) > 0, "'None' model should always be available."
    if default_option not in filtered_preprocessor_list:
        default_option = filtered_preprocessor_list[0]
    if len(filtered_model_list) == 1:
        default_model = "None"
    else:
        default_model = filtered_model_list[1]
        for x in filtered_model_list:
            if "11" in x.split("[")[0]:
                default_model = x
                break

    return (
        filtered_preprocessor_list,
        filtered_model_list,
        default_option,
        default_model
    )
