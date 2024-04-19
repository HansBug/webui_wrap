import gradio as gr

from webui_wrap.base import get_webui_client


def create_base_model_ui():
    client = get_webui_client()

    def base_model_refresh():
        return gr.Dropdown(
            value=client.util_get_current_model(),
            choices=client.util_get_model_names(),
            label='Base Model',
        )

    def _base_model_select(model_name):
        client.util_set_model(model_name)
        return base_model_refresh()

    with gr.Row():
        gr_base_model = base_model_refresh()
        gr_base_model_refresh = gr.Button(value='Refresh')
        gr_clip_skip = gr.Slider(value=2, minimum=1, maximum=3, label='Clip Skip')

        gr_base_model_refresh.click(
            fn=base_model_refresh,
            outputs=[gr_base_model],
        )
        gr_base_model.select(
            fn=_base_model_select,
            inputs=[gr_base_model],
            outputs=[gr_base_model],
        )

    return gr_base_model, gr_clip_skip
