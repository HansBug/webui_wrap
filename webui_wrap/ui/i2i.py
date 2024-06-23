import json
import logging

import gradio as gr
import numpy as np
from hbutils.string import plural_word

from ..base import auto_init_webui, get_webui_client, WEBUI_SAMPLERS
from ..storage import load_recorder_from_env


def i2i_infer(init_image, inpaint_blur, prompt, neg_prompt: str, seed: int = -1,
              sampler_name='DPM++ 2M Karras', cfg_scale=7, img_cfg_scale=1.5, steps=30,
              firstphase_width=512, firstphase_height=768, denoising_strength=0.75,
              batch_size=1,
              clip_skip: int = 2, base_model: str = 'meinamix_v11'):
    auto_init_webui()
    client = get_webui_client()
    client.util_set_model(base_model)

    origin_image = init_image['background']
    mask_image = init_image['layers'][-1]
    mask_alpha = np.isclose(np.array(mask_image)[..., 3].astype(np.float32) / 255.0, 1.0)
    mask_used = np.any(mask_alpha)

    result = client.img2img(
        images=[origin_image],
        mask_image=mask_image if mask_used else None,
        mask_blur=inpaint_blur,
        prompt=prompt,
        negative_prompt=neg_prompt,
        sampler_name=sampler_name,
        cfg_scale=cfg_scale,
        image_cfg_scale=img_cfg_scale,
        seed=seed,
        steps=steps,
        width=firstphase_width,
        height=firstphase_height,
        denoising_strength=denoising_strength,
        batch_size=batch_size,
        override_settings={
            'CLIP_stop_at_last_layers': clip_skip,
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


_DEFAULT_PROMPT = """
(safe:1.10), best quality, masterpiece, highres, solo, (saber_fatestaynightufotable:1.10), 11 <lora:saber_fatestaynightufotable:0.80>
"""

_DEFAULT_NEG_PROMPT = """
(worst quality, low quality:1.40), (zombie, sketch, interlocked fingers, comic:1.10), (full body:1.10), lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, white border, (english text, chinese text:1.05), (censored, mosaic censoring, bar censor:1.20)
"""


def create_i2i_ui(gr_base_model: gr.Dropdown, gr_clip_skip: gr.Slider):
    auto_init_webui()

    with gr.Row():
        with gr.Column():
            with gr.Row():
                gr_prompt = gr.TextArea(label='Prompt', value=_DEFAULT_PROMPT.lstrip(), show_copy_button=True)
            with gr.Row():
                gr_neg_prompt = gr.TextArea(label='negative prompt', value=_DEFAULT_NEG_PROMPT.lstrip(),
                                            show_copy_button=True)
            with gr.Row():
                gr_init_image = gr.ImageEditor(label='Initial Image', type='pil', image_mode='RGBA')
            with gr.Row():
                gr.Markdown("""
                **If you want to inpaint some area, just brush the areas you need.**

                If no areas brushed, it will be simply image-to-image.
                """)
                gr_inpaint_blur = gr.Slider(label='Inpaint Mask Blur', value=4, minimum=1, maximum=64, step=1)

            with gr.Row():
                gr_sampler = gr.Dropdown(label='Sampler', value='Euler a', choices=WEBUI_SAMPLERS)
                gr_steps = gr.Slider(value=25, minimum=1, maximum=50, step=1, label='Steps')
                gr_denoising_strength = gr.Slider(value=0.6, minimum=0.000, maximum=1.000, step=0.05,
                                                  label='Denoising Strength')

            with gr.Row():
                gr_cfg_scale = gr.Slider(value=7.0, minimum=0.1, maximum=15.0, step=0.1, label='CFG Scale')
                gr_img_cfg_scale = gr.Slider(value=1.5, minimum=0.1, maximum=15.0, step=0.1,
                                             label='Image CFG Scale')

            with gr.Row():
                gr_width = gr.Slider(value=512, minimum=128, maximum=2048, step=16, label='Width')
                gr_height = gr.Slider(value=768, minimum=128, maximum=2048, step=16, label='Height')

            with gr.Row():
                gr_seed = gr.Textbox(value='-1', label='Seed')
                gr_batch_size = gr.Slider(value=1, minimum=1, maximum=16, step=1, label='Batch Size')

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
            i2i_infer,
            inputs=[
                gr_init_image, gr_inpaint_blur, gr_prompt, gr_neg_prompt, gr_seed,
                gr_sampler, gr_cfg_scale, gr_img_cfg_scale, gr_steps,
                gr_width, gr_height, gr_denoising_strength,
                gr_batch_size,
                gr_clip_skip, gr_base_model,
            ],
            outputs=[gr_gallery, gr_hidden_metas],
        )
