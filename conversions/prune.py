import torch
import torch.nn.utils.prune

def convert(module, type="l1_unstructured", **kwargs):
    # TODO: for structured pruning, we should also prune bias.
    if type == "random_unstructured":
        torch.nn.utils.prune.random_unstructured(module, name='weight', **kwargs)
    elif type == "l1_unstructured":
        torch.nn.utils.prune.l1_unstructured(module, name='weight', **kwargs)
    elif type == "ln_structured":
        if "dim" not in kwargs:
            # by default we prune along filters dimensions
            kwargs["dim"] = 0
        torch.nn.utils.prune.ln_structured(module, name="weight", **kwargs)
    else:
        raise Exception("Unknown type")
    
    return module