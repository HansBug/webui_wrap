import click
import gradio as gr
from ditk import logging

from webui_wrap.base import auto_init_webui, get_webui_client
from webui_wrap.ui import create_t2i_ui, create_base_model_ui

logging.try_init_root(logging.INFO)
CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


@click.command(context_settings=CONTEXT_SETTINGS, help='Start UI')
@click.option('--bind_all', 'bind_all', is_flag=True, type=bool, default=False,
              help='Bind to all the server name.', show_default=True)
@click.option('--share', 'share', is_flag=True, type=bool, default=False,
              help='Create gradio share links.', show_default=True)
@click.option('--port', 'port', type=int, default=10187,
              help='Server port.', show_default=True)
def app(bind_all: bool, share: bool, port: int):
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

    with gr.Blocks() as demo:
        with gr.Row():
            gr_base_model, gr_clip_skip = create_base_model_ui()

        with gr.Row():
            with gr.Tabs():
                with gr.Tab('T2I'):
                    create_t2i_ui(gr_base_model, gr_clip_skip)

    demo.launch(
        share=bool(share),
        server_name='0.0.0.0'
        if bind_all else None,
        server_port=port
    )


if __name__ == '__main__':
    auto_init_webui()
    app()
