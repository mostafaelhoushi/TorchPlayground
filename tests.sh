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

function imdb() {
  start
  python main.py --task imdb --arch prajjwal1/bert-tiny --dry-run --epochs 1 || error "imdb example failed"
}

function run_all() {
  imagenet
  mnist
  cifar10
  imdb
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
