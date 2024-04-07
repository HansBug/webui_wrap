import gradio as gr

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
    return result.images


def create_t2i_ui():
    auto_init_webui()

    with gr.Row():
        with gr.Column():
            gr_prompt = gr.TextArea(label='Prompt')
            gr_neg_prompt = gr.TextArea(label='negative prompt')
            gr_sampler = gr.Dropdown(label='Sampler', value='Euler a', choices=WEBUI_SAMPLERS)
            gr_seed = gr.Textbox(value='-1', label='Seed')
            gr_steps = gr.Slider(value=25, minimum=1, maximum=50, step=1, label='Steps')
            gr_cfg_scale = gr.Slider(value=7.0, minimum=0.1, maximum=15.0, step=0.1, label='CFG Scale')
            gr_width = gr.Slider(value=512, minimum=128, maximum=2048, step=16, label='Width')
            gr_height = gr.Slider(value=512, minimum=128, maximum=2048, step=16, label='Height')
            gr_batch_size = gr.Slider(value=1, minimum=1, maximum=16, step=1, label='Batch Size')

        with gr.Column():
            gr_generate = gr.Button(value='Generate', variant='primary')
            gr_gallery = gr.Gallery(label='Gallery')

        gr_generate.click(
            t2i_infer,
            inputs=[
                gr_prompt, gr_neg_prompt, gr_seed,
                gr_sampler, gr_cfg_scale, gr_steps, gr_width, gr_height,
                gr_batch_size
            ],
            outputs=[gr_gallery, ],
        )
