import torch
import torchvision
import importlib

def transform_data(batch, args):
    input, target = batch
    input_transforms = []
    target_transforms = []

    if args.scale_input:
        input_transforms.append(torchvision.transforms.Resize((0,0)))
    
    input_transforms = torchvision.transforms.Compose(input_transforms)
    input = input_transforms(input)

    # TODO: support transforms on target

    batch = input, target
    return batch