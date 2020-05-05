from lucid.optvis import objectives

from lucid.modelzoo.vision_models import Model
import lucid.optvis.render as render
import argparse
from lucid.misc.io.showing import _image_url, _display_html
import tensorflow as tf


class LucidModel(Model):
    image_value_range = [-1, 1]
    input_name = 'input'


def show_image(image):
    html = ""
    data_url = _image_url(image)
    html += '<img width=\"100\" style=\"margin: 10px\" src=\"' + data_url + '\">'
    with open("img.html", "w") as f:
        f.write(html)
    _display_html(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--model_file', type=str, default='lucid_model.pb')
    args = parser.parse_args()

    model = LucidModel()
    model.model_path = args.model_file
    model.image_shape = [args.crop_size, args.crop_size, 3]

    print("Nodes in graph:")
    for node in model.graph_def.node:
        print(node.name)
    print("=" * 30)

    obj = objectives.channel("prediction/Conv2D", 0) - objectives.channel("prediction/Conv2D", 0)
    res = render.render_vis(model, obj, transforms=[])
    show_image(res)
