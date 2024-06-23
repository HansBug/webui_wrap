import json
import logging
from functools import lru_cache

import gradio as gr
from hbutils.string import plural_word
from webuiapi import ControlNetUnit, ADetailer

from .adetailer import create_adetailer_ui
from .controlnet import create_controlnet_ui
from ..base import auto_init_webui, get_webui_client, WEBUI_SAMPLERS, has_dynamic_prompts, dynamic_prompt_params, \
    has_controlnet, has_adetailer
from ..storage import load_recorder_from_env


def t2i_infer(
        prompt, neg_prompt: str, seed: int = -1,
        sampler_name='DPM++ 2M Karras', cfg_scale=7, steps=30,
        firstphase_width=512, firstphase_height=768,
        batch_size=1,
        enable_hr: bool = False, hr_resize_x=832, hr_resize_y=1216,
        denoising_strength=0.6, hr_second_pass_steps=20, hr_upscaler='R-ESRGAN 4x+ Anime6B',
        clip_skip: int = 2, base_model: str = 'meinamix_v11',

        dynamic_prompts_enabled: bool = False, dp_fixed_seed: bool = False,

        cn_enabled: bool = True, cn_input_image=None,
        cn_preprocessor: str = 'None', cn_model: str = 'None',
        cn_control_weight: float = 1.0, cn_start_control_step=0.0,
        cn_end_control_step=1.0, cn_control_mode=0, cn_resize_mode="Crop and Resize",

        ad_enabled: bool = False, ad_model: str = 'None',
        ad_prompt: str = '', ad_neg_prompt: str = '',
):
    auto_init_webui()
    client = get_webui_client()
    client.util_set_model(base_model)

    controlnet_units = []
    if cn_enabled:
        controlnet_units.append(ControlNetUnit(
            input_image=cn_input_image,
            module=cn_preprocessor,
            model=cn_model,
            weight=cn_control_weight,
            guidance_start=cn_start_control_step,
            guidance_end=cn_end_control_step,
            control_mode=cn_control_mode,
            resize_mode=cn_resize_mode,
        ))

    adetailer_units = []
    if ad_enabled:
        adetailer_units.append(ADetailer(
            ad_model=ad_model,
            ad_prompt=ad_prompt,
            ad_negative_prompt=ad_neg_prompt,
            ad_clip_skip=clip_skip,
        ))

    logging.info('Inferring ...')
    result = client.txt2img(
        prompt=prompt,
        negative_prompt=neg_prompt,
        batch_size=batch_size,
        sampler_name=sampler_name,
        cfg_scale=cfg_scale,
        steps=steps,
        firstphase_width=firstphase_width,
        firstphase_height=firstphase_height,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        denoising_strength=denoising_strength,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_upscaler=hr_upscaler,
        seed=seed,
        enable_hr=enable_hr,
        override_settings={
            'CLIP_stop_at_last_layers': clip_skip,
        },
        controlnet_units=controlnet_units,
        adetailer=adetailer_units,
        alwayson_scripts={
            **dynamic_prompt_params(
                is_enabled=dynamic_prompts_enabled,
                is_combinatorial=dynamic_prompts_enabled,
                use_fixed_seed=dp_fixed_seed,
            )
        },
    )

    logging.info(f'T2I complete, {plural_word(len(result.images), "image")} get.')
    meta_infos = [image.info.get('parameters') for image in result.images]
    recorder = load_recorder_from_env()
    logging.info(f'Recording {plural_word(len(result.images), "image")} to system.')
    for image, meta_info in zip(result.images, meta_infos):
        recorder.put_image(image, meta_info)
    recorder.save()

    return result.images, json.dumps(meta_infos)


@lru_cache()
def _get_hires_upscalers():
    client = get_webui_client()
    return [item['name'] for item in client.get_upscalers()]


_DEFAULT_PROMPT = """
(safe:1.10), best quality, masterpiece, highres, solo, (saber_fatestaynightufotable:1.10), 11 <lora:saber_fatestaynightufotable:0.80>
"""

_DEFAULT_NEG_PROMPT = """
(worst quality, low quality:1.40), (zombie, sketch, interlocked fingers, comic:1.10), (full body:1.10), lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, white border, (english text, chinese text:1.05), (censored, mosaic censoring, bar censor:1.20)
"""


def create_t2i_ui(gr_base_model: gr.Dropdown, gr_clip_skip: gr.Slider):
    auto_init_webui()

    with gr.Row():
        with gr.Column():
            with gr.Row():
                gr_prompt = gr.TextArea(label='Prompt', value=_DEFAULT_PROMPT.lstrip(), show_copy_button=True)
            with gr.Row():
                gr_neg_prompt = gr.TextArea(label='negative prompt', value=_DEFAULT_NEG_PROMPT.lstrip(),
                                            show_copy_button=True)

            with gr.Tabs():
                with gr.Tab('General'):
                    with gr.Row():
                        gr_sampler = gr.Dropdown(label='Sampler', value='Euler a', choices=WEBUI_SAMPLERS)
                        gr_steps = gr.Slider(value=25, minimum=1, maximum=50, step=1, label='Steps')
                        gr_cfg_scale = gr.Slider(value=7.0, minimum=0.1, maximum=15.0, step=0.1, label='CFG Scale')

                    with gr.Row():
                        gr_width = gr.Slider(value=512, minimum=128, maximum=2048, step=16, label='Width')
                        gr_height = gr.Slider(value=768, minimum=128, maximum=2048, step=16, label='Height')

                    with gr.Row():
                        gr_seed = gr.Textbox(value='-1', label='Seed')
                        gr_batch_size = gr.Slider(value=1, minimum=1, maximum=16, step=1, label='Batch Size')

                with gr.Tab('Hires Fix'):
                    with gr.Row():
                        gr_enable_hr = gr.Checkbox(value=False, label='Enable Hires Fix')

                    with gr.Row():
                        gr_hires_width = gr.Slider(value=832, minimum=128, maximum=2048, step=16, label='Hires Width')
                        gr_hires_height = gr.Slider(value=1216, minimum=128, maximum=2048, step=16,
                                                    label='Hires Height')

                    with gr.Row():
                        gr_denoising_strength = gr.Slider(value=0.6, minimum=0.000, maximum=1.000, step=0.05,
                                                          label='Denoising Strength')
                        gr_hires_steps = gr.Slider(value=20, minimum=1, maximum=50, step=1, label='Hires Steps')

                    with gr.Row():
                        gr_hires_upscaler = gr.Dropdown(value='R-ESRGAN 4x+ Anime6B', label='Hires Upscaler',
                                                        choices=_get_hires_upscalers())

                with gr.Tab('Dynamic Prompts', visible=has_dynamic_prompts()):
                    gr_dynamic_prompts_enabled = gr.Checkbox(
                        value=False,
                        label='Enable Dynamic Prompts' if has_dynamic_prompts() else
                        'Enable Dynamic Prompts (No Plugin, Please install it before using)',
                        interactive=has_dynamic_prompts(),
                    )
                    gr_dp_fixed_seed = gr.Checkbox(
                        value=False, label='Use Fixed Seed',
                        interactive=has_dynamic_prompts()
                    )
                with gr.Tab('ControlNet', visible=has_controlnet()):
                    gr_controlnet_components = create_controlnet_ui()
                with gr.Tab('Adetailer', visible=has_adetailer()):
                    gr_adetailer_components = create_adetailer_ui()

        with gr.Column():
            gr_generate = gr.Button(value='Generate', variant='primary')
            gr_gallery = gr.Gallery(label='Gallery')
            gr_hidden_metas = gr.TextArea(visible=False, interactive=False)
            gr_meta_info = gr.Text(label='Meta Information', value='', lines=10, show_copy_button=True,
                                   interactive=False)

            def _gallery_select(hidden_meta: str, evt: gr.SelectData):
                if evt.selected:
                    return json.loads(hidden_meta)[evt.index] or '<empty>'
                else:
                    return 'N/A'

            gr_gallery.select(
                _gallery_select,
                inputs=[gr_hidden_metas],
                outputs=[gr_meta_info],
            )

        gr_generate.click(
            t2i_infer,
            inputs=[
                gr_prompt, gr_neg_prompt, gr_seed,
                gr_sampler, gr_cfg_scale, gr_steps, gr_width, gr_height,
                gr_batch_size,
                gr_enable_hr, gr_hires_width, gr_hires_height,
                gr_denoising_strength, gr_hires_steps, gr_hires_upscaler,
                gr_clip_skip, gr_base_model,
                gr_dynamic_prompts_enabled, gr_dp_fixed_seed,
                *gr_controlnet_components,
                *gr_adetailer_components,
            ],
            outputs=[gr_gallery, gr_hidden_metas],
        )
