import os
import sys
import traceback

import numpy as np
import tensorflow as tf
from flask import Flask
from flask import jsonify
from flask import request

import utils

app = Flask(__name__)
model = None


def load_model():
    tf.config.set_soft_device_placement(True)

    # limit the gpu memory usage as much as it need.
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Detected {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

    global model
    model = tf.keras.models.load_model(os.environ['MODEL_PATH'])
    model.summary()
    return model


@app.route('/')
def hello_world():
    return jsonify({'status': {'code': 0, 'msg': 'success'}, 'data': 'hello world'})


@app.route('/image', methods=['POST'])
def digits_classification():
    global model
    if model is None:
        model = load_model()

    # accept files as URL, base64 encoded or file stream
    if request.content_type and request.content_type.startswith('application/json'):
        data = request.get_json()
        image_files = None
        image_base64s = data.get('base64s')
        image_urls = data.get('urls')
    else:
        # content-type: "application/x-www-form-unlencoded" or "multipart/form-data"
        data = request.form
        image_files = request.files.getlist('files')
        image_base64s = data.getlist('base64s')
        image_urls = data.getlist('urls')

    # image bytes
    images, request_image_type = utils.get_request_images_as_file(
        image_files=image_files,
        image_base64s=image_base64s,
        image_urls=image_urls
    )

    # preprocessing: resize, convert to gray, normalize
    images = utils.mnist_preprocess(images)

    logits = model.predict(images, verbose=0)
    probability = tf.nn.softmax(logits).numpy()
    prediction = np.argmax(probability, 1)

    return jsonify(
        {
            'status': {'code': 0, 'msg': 'success'},
            'data': {
                'probabilities': probability.tolist(),
                'predictions': prediction.tolist()
            }
        }
    )


@app.errorhandler(Exception)
def handle_unknown_error(e):
    return jsonify({
        'status': {'code': 500, 'msg': repr(traceback.format_exception(*sys.exc_info()))},
        'data': None
    })


if __name__ == '__main__':
    # 如果服务运行于 Nginx 之类的代理下，由于代理设置的某些 HTTP 报文头，可能存在请求不能转发的情况
    # 使用下面的方式能保证正常转发，但是仅在信任代理、信任请求方的情况下使用
    # https://flask.palletsprojects.com/en/1.1.x/deploying/wsgi-standalone/#proxy-setups
    # from werkzeug.contrib.fixers import ProxyFix
    # app.wsgi_app = ProxyFix(app.wsgi_app)
    model = load_model()
    app.run(host='0.0.0.0', port=5000)
