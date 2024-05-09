import gradio as gr

from ..base import get_adetailer_models, has_adetailer, get_adetailer_version


def create_adetailer_ui():
    gr_enable = gr.Checkbox(
        value=False,
        label=f'Enable Adetailer {get_adetailer_version()}' if has_adetailer() else 'Adetailer Not Available',
        interactive=has_adetailer(),
    )

    gr_model = gr.Dropdown(
        choices=['None', *get_adetailer_models()] if has_adetailer() else ['None'],
        value='None',
        label='Adetailer Model',
    )

    gr_prompt = gr.TextArea(
        placeholder='Prompt for Adetailer parts',
        value='',
        label='Adetailer Prompts',
        max_lines=3
    )
    gr_neg_prompt = gr.TextArea(
        placeholder='Negative prompt for Adetailer parts',
        value='',
        label='Adetailer Negative Prompts',
        max_lines=3
    )

    return gr_enable, gr_model, gr_prompt, gr_neg_prompt
