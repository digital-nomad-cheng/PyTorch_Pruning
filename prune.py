import time 

import numpy as np
import torch
from torchvision import models


def replace_layers(model, i, indexes, layers):
    # print(i)
    # print(indexes)
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

def prune_conv_layer(model, layer_index, filter_index, use_cuda=False):
    _, conv = list(model.features._modules.items())[layer_index]
    _, norm = list(model.features._modules.items())[layer_index+1]
    _, relu = list(model.features._modules.items())[layer_index+2]

    next_conv = None
    next_norm = None
    next_relu = None

    offset = 1

    while layer_index + offset < len(model.features._modules.items()):
        res = list(model.features._modules.items())[layer_index+offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            next_name, next_conv = res
            break
        offset = offset + 1


    new_conv = torch.nn.Conv2d(in_channels = conv.in_channels, 
                               out_channels = conv.out_channels-1,
                               kernel_size = conv.kernel_size, 
                               stride = conv.stride,
                               padding = conv.padding,
                               dilation = conv.dilation,
                               groups = conv.groups,
                               bias = (conv.bias is not None))
    new_norm = torch.nn.BatchNorm2d(conv.out_channels-1)
    new_relu = torch.nn.ReLU(conv.out_channels-1)
      
    old_conv_weights = conv.weight.data.cpu().numpy()
    new_conv_weights = new_conv.weight.data.cpu().numpy()
    print("layer_index:", layer_index)
    print("old shape:", old_conv_weights.shape)
    print("new shape:", new_conv_weights.shape)
    print("filter index:", filter_index)
    
    # if its the last channel of weight
    if filter_index == old_conv_weights.shape[0]:
        new_conv_weights[:, :, :, :] = old_conv_weights[:filter_index-1, :, :, :]
    else:
        new_conv_weights[:filter_index, :, :, :] = old_conv_weights[:filter_index, :, :, :]
        new_conv_weights[filter_index:, :, :, :] = old_conv_weights[filter_index+1:, :, :, :]
    
    new_conv.weight.data = torch.from_numpy(new_conv_weights)
    if use_cuda:
        new_conv.weight.data = new_conv.weight.data.cuda()

    bias_numpy = conv.bias.data.cpu().numpy()
    bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index:] = bias_numpy[filter_index+1:]
    new_conv.bias.data = torch.from_numpy(bias)
    if use_cuda:
        new_conv.bias.data = new_conv.bias.data.cuda()
    
    old_norm_weights = norm.weight.data.cpu().numpy()
    new_norm_weights = new_norm.weight.data.cpu().numpy()
    # print(old_norm_weights.shape)
    # print(new_norm_weights.shape)
    new_norm_weights[:filter_index] = old_norm_weights[:filter_index]
    new_norm_weights[filter_index:] = old_norm_weights[filter_index+1:]
    new_norm.weight.data = torch.from_numpy(new_norm_weights)


    if not next_conv is None:
        next_new_conv = torch.nn.Conv2d(in_channels=next_conv.in_channels-1,
                out_channels = next_conv.out_channels,
                kernel_size = next_conv.kernel_size,
                stride = next_conv.stride,
                padding = next_conv.padding,
                dilation = next_conv.dilation,
                groups = next_conv.groups,
                bias = (next_conv.bias is not None))

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()
    
        new_weights[:, :filter_index, :, :] = old_weights[:, :filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index+1:, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights)

        if use_cuda:
            next_new_conv.weight.data = next_new_conv.weight.data.cuda()
    
        next_new_conv.bias.data = next_conv.bias.data
    
        next_new_conv.bias.data = next_conv.bias.data
        

    if not next_conv is None:
        features = torch.nn.Sequential(
                *(replace_layers(model.features, i, 
                    [layer_index, layer_index+1, layer_index+2, layer_index+offset], 
                    [new_conv, new_norm, new_relu, next_new_conv])  
                    for i, _ in enumerate(model.features)))
        del model.features
        del conv

        model.features = features
    else: # prunning the last conv layer, affects the first linear layer of the classifier
        model.features = torch.nn.Sequential(
                *(replace_layers(model.features, i, 
                    [layer_index, layer_index+1, layer_index+2], 
                    [new_conv, new_norm, new_relu]) 
                    for i, _ in enumerate(model.features)))
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.modules.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index + 1
        
        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")

        params_per_input_channel = old_linear_layer.in_features // conv.out_channels

        new_linear_layer = torch.nn.Linear(
                old_linear_layer.in_features - params_per_input_channel,
            old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        new_weights[:, :filter_index*params_per_input_channel] = \
            old_weights[:, :filter_index*params_per_input_channel]
        new_weights[:, filter_index*params_per_input_channel:] = \
            old_weights[:, (filter_index+1)*params_per_input_channel:]

        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights)

        if use_cuda:
            new_linear_layer.weight.data = new_linear_layer.weight.data.cuda()

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                    [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model


if __name__ == "__main__":
    model = models.vgg16(pretrained=True)
    model.train()

    t0 = time.time()
    model = prune_conv_layer(model, 28, 10, use_cuda=False)
    print("The prunning took", time.time() - t0)

