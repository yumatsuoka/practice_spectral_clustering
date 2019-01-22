#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import six
from six.moves.urllib import request

import struct
import gzip


base_path = "./dump/dataset"
file_path = "fmnist.pkl"
kinds = ['train', 't10k']

category_names = ['T-shirt', "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag","Ankle_boot"]


def download():
    #url = 'https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion'
    url = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion'

    for tag in kinds:
        labels_path = tag + '-labels-idx1-ubyte.gz'
        images_path = tag + '-images-idx3-ubyte.gz'
        request.urlretrieve(os.path.join(url, labels_path), os.path.join(base_path, labels_path))
        request.urlretrieve(os.path.join(url, images_path), os.path.join(base_path, images_path))


def load_mnist(kind):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(base_path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(base_path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def load(name=os.path.join(base_path, file_path)):
    print("# Load MNIST dataset")
    if os.path.exists(os.path.join(base_path, file_path)):
        pass
    else:
        main()
    with open(name, 'rb') as data:
        mnist = six.moves.cPickle.load(data)
    return mnist


def main():

    if not os.path.exists(base_path):
        os.system("mkdir -p {}".format(base_path))

    if os.path.exists(os.path.join(base_path, file_path)):
        pass
    else:
        print("# Download MNIST dataset.")
        download()

        print("# Convert training images and labels.")
        train_image_ary, train_label_ary = load_mnist(kinds[0])

        print("# Convert test images and labels.")
        test_image_ary, test_label_ary = load_mnist(kinds[1])

        train = {'data': train_image_ary, 'target': train_label_ary,
                'size': len(train_label_ary), 'categories': len(category_names),
                'category_names': category_names}
        test = {'data': test_image_ary, 'target': test_label_ary,
                'size': len(test_label_ary), 'categories': len(category_names),
                'category_names': category_names}
        data = {'train': train, 'test': test}

        print("# Dump both images as pkl file.")
        with open(os.path.join(base_path, file_path), 'wb') as out_data:
            six.moves.cPickle.dump(data, out_data, -1)

    print("# MNIST dataset ready.")

if __name__ == '__main__':
    main()
