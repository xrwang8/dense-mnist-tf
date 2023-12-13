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


def mnist_dataset(batch_size=64, data_dir=None):
    # load dataset
    if data_dir:
        print(f'Loading mnist data from {data_dir}')
        (x_train, y_train), (x_test, y_test) = load_data(data_dir)
    else:
        print('Loading mnist data from tf.keras.datasets.mnist')
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_dataset, test_dataset


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
    parser = argparse.ArgumentParser(description='Train MNIST digits classification')
    parser.add_argument('--data-dir', help='Directory to MNIST dataset')
    parser.add_argument('--output-dir', help='Directory to save models and logs')
    parser.add_argument('--epochs', default=2, help='Number of epochs')
    parser.add_argument('--eval', action='store_true', help='whether do evaluation after training finished')
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
        # build dataset
        train_dataset, test_dataset = mnist_dataset(
            batch_size=64 * num_workers,
            data_dir=args.data_dir
        )

        # build model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        model.summary()

    # training
    model.fit(train_dataset, epochs=args.epochs, steps_per_epoch=70)

    # evaluation
    if args.eval:
        model.evaluate(test_dataset, verbose=2)

    # save model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save(args.output_dir)
        print(f'Saved model to {args.output_dir}')


if __name__ == '__main__':
    # python train.py --data-dir=/path/to/MNIST/dataset --output-dir=/path/to/output
    print("TensorFlow version:", tf.__version__)
    main()
