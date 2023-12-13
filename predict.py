import argparse
import gzip
import json
import os

import numpy as np
import tensorflow as tf

num_workers = 1


def load_data(data_dir):
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    paths = []
    for f in files:
        paths.append(os.path.join(data_dir, f))
    with gzip.open(paths[0], 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    with gzip.open(paths[2], 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    return (x_train, y_train), (x_test, y_test)


def mnist_test_images(batch_size=64, data_dir=None):
    # load dataset
    if data_dir:
        print(f'Loading mnist data from {data_dir}')
        (x_train, y_train), (x_test, y_test) = load_data(data_dir)
    else:
        print('Loading mnist data from tf.keras.datasets.mnist')
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    test_images = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size)

    return test_images


def get_strategy(strategy='off'):
    strategy = strategy.lower()
    # multiple nodes, every nodes have multiple GPUs
    if strategy == "multi_worker_mirrored":
        return tf.distribute.experimental.MultiWorkerMirroredStrategy()
    # single node with multiple GPUs
    if strategy == "mirrored":
        return tf.distribute.MirroredStrategy()
    # single node with single GPU
    return tf.distribute.get_strategy()


def setup_env(args):
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

    if args.strategy == 'multi_worker_mirrored':
        index = int(os.environ['VK_TASK_INDEX'])
        task_name = os.environ["VC_TASK_NAME"].upper()
        ips = os.environ[f'VC_{task_name}_HOSTS']
        ips = ips.split(',')
        global num_workers
        num_workers = len(ips)
        ips = [f'{ip}:20000' for ip in ips]
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": {
                "worker": ips
            },
            "task": {"type": "worker", "index": index}
        })


def setup_config():
    parser = argparse.ArgumentParser(description='MNIST digits classification using trained model')
    parser.add_argument('--model-dir', help='Directory to model')
    parser.add_argument('--data-dir', help='Directory to MNIST dataset')
    parser.add_argument('--output-dir', help='Directory to save models and logs')
    parser.add_argument(
        '--strategy',
        default='off',
        choices=['off', 'mirrored', 'multi_worker_mirrored'],
        help='TensorFlow distributed training strategies'
    )
    args = parser.parse_args()
    return args


def main():
    args = setup_config()
    # tf2 limitation: Collective ops must be configured at program startup
    strategy = get_strategy(args.strategy)
    setup_env(args)

    with strategy.scope():
        test_images = mnist_test_images(batch_size=64 * num_workers, data_dir=args.data_dir)
        model = tf.keras.models.load_model(args.model_dir)
        model.summary()

    logits = model.predict(test_images, verbose=2)
    probabilities = tf.nn.softmax(logits).numpy()
    predictions = np.argmax(probabilities, 1)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        prediction_results_file = os.path.join(args.output_dir, "prediction_results.npy")
        np.save(prediction_results_file, predictions)
        print(f"Saved prediction results to {prediction_results_file}")


if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)
    main()
