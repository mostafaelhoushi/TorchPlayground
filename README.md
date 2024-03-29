# Getting Started
- Install some libraries:
```
### clone repo ###
cd TorchPlayground
pip install -r requirements.txt
```

# Command Template
```
python main.py --task <imagenet | cifar10 | mnist> --arch <model>
            [--train | --infer <sample path> | --evaluate]
            --<transform> '{<transform parameters>}'
                [--layer-start <num>] [--layer-end <num>]
                [--transform-epoch-start <num>] [--transform-epoch-end <num>] [--transform-epoch-step <num>]
            [--epochs <num>] [--batch-size <num>] [--momentum <num>] [--optimizer <opt>] [--pretrained <true | false>]
            [--lr <num>] [--lr-schedule <scheduler>] [--lr-step-size <num>] [--lr-milestones <nums>]
            [--cpu | --gpu <gpu-id>]
```

There are more options that can be listed by running `python main.py --help`

# Without Transformations

- To infer image:
```
python main.py --task cifar10 -i grumpy.jpg
```

- To train on CIFAR10:
```
python main.py --task cifar10 --epochs 200
```

- To evaluate Imagenet dataset:
```
python main.py --task imagenet --evaluate --data-dir <path to imagenet>
```

- To train on Imagnet:
```
python main.py --data-dir <path to imagenet>
```

# With Model Transformations
<details>
<summary><b>Quantization</b></summary>

- To convert convolution to APoT 5-bit quantized convolution:
```
python main.py -i grumpy.jpg --apot '{"bit": 5}'
```

- To convert convolution and linear layers to HAQ 4-bit quantization:
```
python main.py -i grumpy.jpg --haq '{"w_bit": 5, "a_bit": 5}'
```

- To quantize convolution and linear layers using DeepShift:
```
python main.py --deepshift '{"shift_type": "PS"}'
```
</details>

<details>
<summary><b>Pruning</b></summary>

- Unstructured pruning with 90% sparsity based on L1 norm:
```
python main.py --task cifar10 --epochs 200 --prune '{"amount": 0.9, "type": "l1_unstructured"}'
```

- Structured pruning with 50% filters removed based on L0 norm:
```
python main.py --task cifar10 --epochs 200 --prune '{"amount": 0.9, "type": "ln_structured", "n": 0}'
```

- Global unstructured pruning with 90% sparsity based on L1 norm:
```
python main.py --task cifar10 --epochs 200 --global-prune '{"amount": 0.9, "pruning_method": "L1Unstructured"}'
```
</details>

<details>
<summary><b>Tensor Decomposition</b></summary>

- To perform Tucker decomposition
```
python main.py --data-dir ~/datasets/imagenet --tucker-decompose '{"ranks":[20,20]}' --task imagenet --pretrained True --arch resnet18 --layer-start 1
```

- To perform depthwise decomposition
```
python main.py --data-dir ~/datasets/imagenet --depthwise-decompose '{"threshold":0.3}' --task imagenet --pretrained True --arch resnet18 --layer-start 1
```
</details>

<details>
<summary><b>Other</b></summary>

- To increase stride of convolution and upsample
```
python main.py -i grumpy.jpg --convup '{"scale": 2, "mode": "bilinear"}'
```
</details>

# With Data Transformations

<details>
<summary><b>Input Resize</b></summary>

- To downsize input images:
```
python main.py --resize-input '{"size":[15,15]}' --task cifar10 --pretrained False --arch resnet20
```

- To downsize every other epoch
```
python main.py --resize-input '{"size":[15,15]}' --task cifar10 --pretrained False --arch resnet20 --transform-epoch-start 0 --transform-epoch-end 200 --transform-epoch-step 2
```
</details>