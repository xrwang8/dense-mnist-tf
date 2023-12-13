import argparse
import gzip
import json
import os

import numpy as np


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


def setup_config():
    parser = argparse.ArgumentParser(
        description="MNIST digit classification Evaluation using prediction results and labels"
    )
    parser.add_argument('--label-dir', required=True, help='Directory to labels')
    parser.add_argument('--predictions', required=True, help='Path to prediction results numpy-array-like file')
    parser.add_argument('--output-dir', required=True, help='Directory to evaluation result')
    args = parser.parse_args()
    return args


def main():
    args = setup_config()

    (_, _), (_, labels) = load_data(args.label_dir)
    predictions = np.load(args.predictions)

    labels = np.reshape(labels, (-1,))
    predictions = np.reshape(predictions, (-1,))

    accuracy = (labels == predictions).sum() / labels.shape[0]
    print(f'Accuracy is {accuracy}')

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        result_file = os.path.join(args.output_dir, "result.json")
        with open(result_file, 'w') as f:
            json.dump({'accuracy': accuracy}, f)
        print(f"Saved evaluation result to {result_file}")


if __name__ == '__main__':
    main()
