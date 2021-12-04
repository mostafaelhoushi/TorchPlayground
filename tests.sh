#!/usr/bin/env bash

python main.py --task mnist --arch lenet --dry-run --epochs 1
python main.py --task cifar10 --arch resnet18 --dry-run --epochs 1
python main.py --task imagenet --arch resnet18 --dry-run --epochs 1 --data ~/pytorch_examples/imagenet/sample

python main.py --task imdb --arch prajjwal1/bert-tiny --dry-run --epochs 1