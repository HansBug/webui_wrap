import gradio as gr

from ..base import has_controlnet
from ..base.cn import preprocessor_filters, select_control_type


def create_controlnet_ui():
    with gr.Row():
        gr_enable_controlnet = gr.Checkbox(
            value=False,
            label='Enable ControlNet' if has_controlnet() else 'Enable ControlNet (No ControlNet Installed)',
            interactive=has_controlnet(),
        )

    with gr.Row():
        gr_input_image = gr.Image(
            label='Input Image',
            type='pil',
            interactive=has_controlnet(),
        )

    with gr.Row():
        filters = list(preprocessor_filters.keys())
        default_filter = filters[0]
        gr_control_type = gr.Dropdown(
            value=default_filter,
            label='Control Type',
            choices=filters,
            interactive=has_controlnet(),
        )

    with gr.Row():
        def _sync_model_and_preprocessors(control_type):
            filtered_preprocessor_list, filtered_model_list, default_preprocessor, default_model \
                = select_control_type(control_type)

            _gr_preprocessor = gr.Dropdown(
                value=default_preprocessor,
                label='Preprocessor',
                choices=filtered_preprocessor_list,
                interactive=has_controlnet(),
            )
            _gr_model = gr.Dropdown(
                value=default_model,
                label='Model',
                choices=filtered_model_list,
                interactive=has_controlnet(),
            )
            return _gr_preprocessor, _gr_model

        gr_preprocessor, gr_model = _sync_model_and_preprocessors(default_filter)
        gr_control_type.select(
            _sync_model_and_preprocessors,
            inputs=[gr_control_type],
            outputs=[gr_preprocessor, gr_model],
        )

    with gr.Row():
        gr_control_weight = gr.Slider(
            value=1.0,
            minimum=0.0,
            maximum=2.0,
            step=0.05,
            label='Control Weight',
            interactive=has_controlnet(),
        )
        gr_start_control_step = gr.Slider(
            value=0.0,
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            label='Starting Control Step',
            interactive=has_controlnet(),
        )
        gr_end_control_step = gr.Slider(
            value=1.0,
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            label='Ending Control Step',
            interactive=has_controlnet(),
        )

    with gr.Row():
        gr_control_mode = gr.Radio(
            value=0,
            choices=[
                ('Balanced', 0),
                ('My prompt is more important', 1),
                ('ControlNet is more important', 2),
            ],
            label='Control Mode',
            interactive=has_controlnet(),
        )

    with gr.Row():
        gr_resize_mode = gr.Radio(
            value="Crop and Resize",
            choices=[
                "Just Resize",
                "Crop and Resize",
                "Resize and Fill",
            ],
            label='Resize Mode',
            interactive=has_controlnet(),
        )

    return (gr_enable_controlnet, gr_input_image, gr_preprocessor, gr_model, gr_control_weight, gr_start_control_step,
            gr_end_control_step, gr_control_mode, gr_resize_mode)
