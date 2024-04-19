import logging
from functools import lru_cache

import gradio as gr
from hbutils.string import plural_word

from ..base import auto_init_webui, get_webui_client, WEBUI_SAMPLERS


def t2i_infer(prompt, neg_prompt: str, seed: int = -1,
              sampler_name='DPM++ 2M Karras', cfg_scale=7, steps=30,
              firstphase_width=512, firstphase_height=768,
              batch_size=1,
              enable_hr: bool = False, hr_resize_x=832, hr_resize_y=1216,
              denoising_strength=0.6, hr_second_pass_steps=20, hr_upscaler='R-ESRGAN 4x+ Anime6B',
              clip_skip: int = 2, base_model: str = 'meinamix_v11'):
    auto_init_webui()
    client = get_webui_client()
    client.util_set_model(base_model)

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
    )

    logging.info(f'Inference complete, {plural_word(len(result.images), "image")} get.')
    return result.images


@lru_cache()
def _get_hires_upscalers():
    client = get_webui_client()
    return [item['name'] for item in client.get_upscalers()]


_DEFAULT_PROMPT = """
(safe:1.10), best quality, masterpiece, highres, solo, (leto_arknights:1.10), 11 <lora:leto_arknights-000069:0.80>
"""

_DEFAULT_NEG_PROMPT = """
(worst quality, low quality:1.40), (zombie, sketch, interlocked fingers, comic:1.10), (full body:1.10), lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, white border, (english text, chinese text:1.05), (censored, mosaic censoring, bar censor:1.20)
"""


def create_t2i_ui(gr_base_model: gr.Dropdown, gr_clip_skip: gr.Slider):
    auto_init_webui()

    with gr.Row():
        with gr.Column():
            with gr.Row():
                gr_prompt = gr.TextArea(label='Prompt', value=_DEFAULT_PROMPT.lstrip())
            with gr.Row():
                gr_neg_prompt = gr.TextArea(label='negative prompt', value=_DEFAULT_NEG_PROMPT.lstrip())

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

        with gr.Column():
            gr_generate = gr.Button(value='Generate', variant='primary')
            gr_gallery = gr.Gallery(label='Gallery')
            gr_meta_info = gr.Code(label='Meta Information', value='', lines=15, language=None)

            def _gallery_select(evt: gr.SelectData):
                print(evt.selected)

            gr_gallery.select(
                _gallery_select,
                inputs=None,
                outputs=None,
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
            ],
            outputs=[gr_gallery, ],
        )
