import torch
import torch.nn.utils.prune as prune

def convert(module, type="l1_unstructured", **kwargs):
    if type == "l1_unstructured":
        prune.l1_unstructured(module, name='weight', **kwargs)
    else:
        raise Exception("Unknown type")
    
    return module