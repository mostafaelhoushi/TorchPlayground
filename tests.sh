#!/usr/bin/env bash

BASE_DIR=`pwd`"/"`dirname $0`
EXAMPLES=`echo $1 | sed -e 's/ //g'`

USE_CUDA=$(python -c "import torchvision, torch; print(torch.cuda.is_available())")
case $USE_CUDA in
  "True")
    echo "using cuda"
    CUDA=1
    CUDA_FLAG="--cuda"
    ;;
  "False")
    echo "not using cuda"
    CUDA=0
    CUDA_FLAG=""
    ;;
  "")
    exit 1;
    ;;
esac

ERRORS=""

function error() {
  ERR=$1
  ERRORS="$ERRORS\n$ERR"
  echo $ERR
}

function install_deps() {
  echo "installing requirements"
  cat requirements.txt | \
    sort -u | \
    # testing the installed version of torch, so don't pip install it.
    grep -vE '^torch$' | \
    pip install -r /dev/stdin || \
    { error "failed to install dependencies"; exit 1; }
}

function start() {
  EXAMPLE=${FUNCNAME[1]}
  echo 
  echo
  echo "##########################"
  echo "Running example: $EXAMPLE"
  echo "=========================="
}

function mnist() {
  start
  python main.py --task mnist --arch lenet --dry-run --epochs 1 || error "mnist example failed"
}

function cifar10() {
  start
  python main.py --task cifar10 --arch resnet18 --dry-run --epochs 1 || error "cifar10 example failed"
}

function imagenet() {
  start
  if [[ ! -d "sample/val" || ! -d "sample/train" ]]; then
    mkdir -p sample/val/n
    mkdir -p sample/train/n
    wget "https://upload.wikimedia.org/wikipedia/commons/5/5a/Socks-clinton.jpg" || { error "couldn't download sample image for imagenet"; return; }
    mv Socks-clinton.jpg sample/train/n
    cp sample/train/n/* sample/val/n/
  fi
  python main.py --task imagenet --arch resnet18 --dry-run --epochs 1 --data sample/ || error "imagenet example failed"
}

function infer() {
  start
  python main.py --task cifar10 -i grumpy.jpg || error "cifar10 example failed"
}

function quantization() {
  start
  python main.py -i grumpy.jpg --apot '{"bit": 5}' || error "APOT example failed"
  python main.py -i grumpy.jpg --haq '{"w_bit": 5, "a_bit": 5}' || error "HAQ example failed"
  python main.py --deepshift '{"shift_type": "PS"}' --dry-run --epochs 1 || error "DeepShift example failed"
}

function pruning() {
  start
  python main.py --task cifar10 --dry-run --epochs 1 --prune '{"amount": 0.9, "type": "l1_unstructured"}' || error "L1 unstructured example failed"
  python main.py --task cifar10 --dry-run --epochs 1 --prune '{"amount": 0.9, "type": "ln_structured", "n": 0}' || error "LN structured example failed"
  # FIXME: mistakenly deleted global_prune.py
  # python main.py --task cifar10 --dry-run --epochs 1 --global-prune '{"amount": 0.9, "pruning_method": "L1Unstructured"}' || error "Global Pruning example failed"
}

function tensor_decomposition() {
  start
  python main.py --tucker-decompose '{"ranks":[20,20]}' --task imagenet --pretrained True --arch resnet18 --layer-start 1 -i grumpy.jpg || error "Tucker Decomposition example failed"
  python main.py --depthwise-decompose '{"threshold":0.3}' --task imagenet --pretrained True --arch resnet18 --layer-start 1 -i grumpy.jpg || error "Depthwise Decomposition example failed"
}

function other() {
  start
  python main.py -i grumpy.jpg --convup '{"scale": 2, "mode": "bilinear"}' || error "ConvUp example failed"
}

function data_resizing() {
  start
  python main.py --resize-input '{"size":[15,15]}' --task cifar10 --pretrained False --arch resnet20 --dry-run --epochs 1 || error "data resizing example failed"
  python main.py --resize-input '{"size":[15,15]}' --task cifar10 --pretrained False --arch resnet20 --transform-epoch-start 0 --transform-epoch-end 2 --transform-epoch-step 2 --dry-run --epochs 1 || error "data resizing alternating epochs example failed"
}

function run_all() {
  imagenet
  mnist
  cifar10
  infer
  quantization
  pruning
  tensor_decomposition
  other
  data_resizing
}

# by default, run all examples
if [ "" == "$EXAMPLES" ]; then
  run_all
else
  for i in $(echo $EXAMPLES | sed "s/,/ /g")
  do
    echo "Starting $i"
    $i
    echo "Finished $i, status $?"
  done
fi

if [ "" == "$ERRORS" ]; then
  echo "Completed successfully with status $?"
else
  echo "Some examples failed:"
  printf "$ERRORS"
fi
