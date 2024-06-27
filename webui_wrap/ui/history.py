import json
import re

import gradio as gr

from ..base import auto_init_webui
from ..storage import load_recorder_from_env


def create_history_ui():
    auto_init_webui()
    recorder = load_recorder_from_env()

    with gr.Tabs():
        with gr.Tab('Query By Tags'):
            def _query_from_recorder(query_text: str):
                segs = list(filter(bool, re.split(r'\s+', query_text)))
                tags, neg_tags = [], []
                for tag in segs:
                    if tag.startswith('-'):
                        neg_tags.append(tag[1:])
                    else:
                        tags.append(tag)

                images = recorder.query_with_tags(tags, neg_tags)
                meta_infos = [image.info.get('parameters') for image in images]
                return images, json.dumps(meta_infos)

            with gr.Row():
                with gr.Column():
                    gr_tags_query = gr.Textbox(value='', placeholder='Enter Tags Here', label='Query Tags')
                    gr_submit = gr.Button(value='Query', variant='primary')
                    gr_gallery = gr.Gallery(label='Gallery')

                with gr.Column():
                    gr_hidden_metas = gr.TextArea(visible=False, interactive=False)
                    gr_meta_info = gr.Text(label='Meta Information', value='', lines=20, show_copy_button=True,
                                           interactive=False)

                gr_submit.click(
                    fn=_query_from_recorder,
                    inputs=[gr_tags_query],
                    outputs=[gr_gallery, gr_hidden_metas],
                )

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

        with gr.Tab('About Tags'):
            pass
