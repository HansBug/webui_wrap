import gradio as gr

from webui_wrap.t2i.ui import create_t2i_ui

if __name__ == '__main__':
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.Tab('T2I'):
                create_t2i_ui()

    demo.launch(share=True)
