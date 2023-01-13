import torch
import importlib

def transform_model(model, args):
    if args.prune:
        prune = importlib.import_module(f"conversions.prune")
        model, _ = convert(model, torch.nn.Linear, prune.convert, index_start=args.layer_start, index_end=args.layer_end, **args.prune)
        model, _ = convert(model, torch.nn.Conv2d, prune.convert, index_start=args.layer_start, index_end=args.layer_end, **args.prune)
    if args.global_prune:
        global_prune = importlib.import_module(f"conversions.global_prune")
        model = global_prune.convert(model, (torch.nn.Linear, torch.nn.Conv2d), index_start=args.layer_start, index_end=args.layer_end, **args.global_prune)

    if args.svd_decompose:
        tensor_decomposition = importlib.import_module(f"conversions.tensor_decomposition")
        model, _ = convert(model, torch.nn.Linear, tensor_decomposition.svd_decompose_linear, index_start=args.layer_start, index_end=args.layer_end, **args.svd_decompose)
    if args.channel_decompose:
        tensor_decomposition = importlib.import_module(f"conversions.tensor_decomposition")
        model, _ = convert(model, torch.nn.Conv2d, tensor_decomposition.channel_decompose_conv, index_start=args.layer_start, index_end=args.layer_end, **args.channel_decompose)
    if args.spatial_decompose:
        tensor_decomposition = importlib.import_module(f"conversions.tensor_decomposition")
        model, _ = convert(model, torch.nn.Conv2d, tensor_decomposition.spatial_decompose_conv, index_start=args.layer_start, index_end=args.layer_end, **args.spatial_decompose)
    if args.depthwise_decompose:
        tensor_decomposition = importlib.import_module(f"conversions.tensor_decomposition")
        model, _ = convert(model, torch.nn.Conv2d, tensor_decomposition.depthwise_decompose_conv, index_start=args.layer_start, index_end=args.layer_end, **args.depthwise_decompose)
    if args.tucker_decompose:
        tensor_decomposition = importlib.import_module(f"conversions.tensor_decomposition")
        model, _ = convert(model, torch.nn.Conv2d, tensor_decomposition.tucker_decompose_conv, index_start=args.layer_start, index_end=args.layer_end, **args.tucker_decompose)
    if args.cp_decompose:
        tensor_decomposition = importlib.import_module(f"conversions.tensor_decomposition")
        model, _ = convert(model, torch.nn.Conv2d, tensor_decomposition.cp_decompose_conv_other, index_start=args.layer_start, index_end=args.layer_end, **args.cp_decompose)

    if args.apot:
        apot = importlib.import_module(f"conversions.apot")
        model, _ = convert(model, torch.nn.Conv2d, apot.convert, index_start=args.layer_start, index_end=args.layer_end, **args.apot)
    if args.haq:
        haq = importlib.import_module(f"conversions.haq")
        model, _ = convert(model, (torch.nn.Conv2d, torch.nn.Linear), haq.convert, index_start=args.layer_start, index_end=args.layer_end, **args.haq)
    if args.deepshift:
        deepshift = importlib.import_module(f"conversions.deepshift")
        model, _ = convert(model, (torch.nn.Conv2d, torch.nn.Linear), deepshift.convert_to_shift.convert, index_start=args.layer_start, index_end=args.layer_end, **args.deepshift)

    if args.convup:
        convup = importlib.import_module(f"conversions.convup")
        model, _ = convert(model, torch.nn.Conv2d, convup.ConvUp, index_start=args.layer_start, index_end=args.layer_end, **args.convup)
    if args.strideout:
        strideout = importlib.import_module(f"conversions.strideout")
        model, _ = convert(model, torch.nn.Conv2d, strideout.StrideOut, index_start=args.layer_start, index_end=args.layer_end, **args.strideout)
    ## todo: add dilate instead of convup

    return model

def count_layer_type(model, layer_type=torch.nn.Conv2d, count=0):
    for name, module in model._modules.items():
        if isinstance(module, layer_type):
            count += 1
        
        if len(list(module.children())) > 0:
            # recurse
            count += count_layer_type(module, layer_type, 0)
    return count    

def convert(model, layer_from, layer_to, index_start=0, index_end=-1, index=0, **kwargs):
    if index_end < 0:
        index_end = index_end + count_layer_type(model, layer_from)
    for name, module in model._modules.items():
        if isinstance(module, (layer_from)):
            if index >= index_start and index <= index_end:
                module.name = name
                model._modules[name] = layer_to(module, **kwargs)
            index += 1
        
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name], index = convert(module, layer_from, layer_to, index_start, index_end, index, **kwargs)
    return model, index

def register_forward_hook(model, function, layer_types=(torch.nn.Conv2d, torch.nn.Upsample)):
    for name, module in model._modules.items():
        if isinstance(module, layer_types):
            module.register_forward_hook(function)
        
        if len(list(module.children())) > 0:
            # recurse
            register_forward_hook(module, function, layer_types)