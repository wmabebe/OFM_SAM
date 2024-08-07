import torch
import torch.nn as nn
import torch.nn.init as init
from functools import partial
import copy

wanda_sums = {i:[[],[]] for i in range(12)}



# Assuming encoder is already a deep copy of model.vision_encoder
def randomize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            init.kaiming_normal_(layer.weight)  # You can use other initializations too
            if layer.bias is not None:
                init.zeros_(layer.bias)


def mlp_forward_hook(inst, inp, out, layer, lin):
    W = inst.weight  # shape: (3072, 768)

    #print(f'inst : {inst} \t layer : {layer} \t lin : {lin}')
    #print(f'\tW : {W.shape}')

    C_out = W.shape[1]
    l2_norm = inp[0].view(-1,C_out)
    l2_norm = l2_norm.norm(p=2, dim=0)

    #print(f'\tl2_norm : {l2_norm.shape}')

    wanda = W.abs() * l2_norm

    if lin == 1:
        row_sums = torch.abs(wanda).sum(dim=1)
        wanda_sums[layer][0].append(row_sums)
    
    elif lin == 2:
        column_sums = torch.abs(wanda).sum(dim=0)
        wanda_sums[layer][1].append(column_sums)
    
    #print(f'\twanda : {wanda.shape}')

    #return wanda

def movement_reordering(model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    grads = {i:[[],[]] for i in range(12)}

    loss_func = nn.MSELoss()

    encoder = copy.deepcopy(model.vision_encoder).to(device)

    # Randomize the weights of the encoder for non-zero grads
    randomize_weights(encoder)

    encoder.train()
    for idx,(inputs, labels) in enumerate(dataloader):
        data = {'pixel_values': torch.stack([d['pixel_values'].squeeze(0) for d in inputs])}
        print(f'data["pixel_values"] : {data["pixel_values"].shape}')
        output = encoder(data["pixel_values"].to(device))

        pred_embeddings = output[0]
        gts_embeddings = torch.stack(labels).to(device)
        loss = loss_func(gts_embeddings, pred_embeddings)

        loss.backward()

        #Capture grads for lin1 and lin2
        for idx, layer in enumerate(encoder.layers):
            G1 = layer.mlp.lin1.weight.grad
            G2 = layer.mlp.lin2.weight.grad
            row_sums = G1.abs().sum(dim=1) #G1.abs()
            column_sums = G2.abs().sum(dim=0) #G2.abs()
            print(f'Layer : {idx}')
            print("\tlin1 grads:", G1.shape)
            print("\tlin2 grads:", G2.shape)
            grads[idx][0].append(row_sums)
            grads[idx][1].append(column_sums)
        
        # # Zero out gradients
        # for param in encoder.parameters():
        #     if param.grad is not None:
        #         param.grad.zero_()
    
    score_dist = {}
    print(f'Aggregating movement sums')
    for (k,v),layer in zip(grads.items(),encoder.layers):
        grad_row_sums = sum(v[0]) / len(v[0])
        grad_column_sums = sum(v[1]) / len(v[1])

        W1 = layer.mlp.lin1.weight  # shape: (3072, 768)
        W2 = layer.mlp.lin2.weight  # shape: (768, 3072)
        b1 = layer.mlp.lin1.bias

        weight_row_sums = W1.abs().sum(dim=1) #W1.abs() 
        weight_column_sums = W2.abs().sum(dim=0) #W2.abs()
        
        avg_row_sums = grad_row_sums.abs() * weight_row_sums
        avg_column_sums = grad_column_sums.abs() * weight_column_sums

        avg_sums = (avg_row_sums + avg_column_sums) / 2

        score_dist[k] = avg_sums.flatten().tolist()

        _, sorted_indices = avg_sums.sort(descending=True)

        print(f'{k} --> {avg_sums.shape}')
        # print(f'\tgrad_row_sums : {grad_row_sums}')
        # print(f'\tweight_row_sums : {weight_row_sums}')
        # print(f'\tavg_row_sums : {avg_row_sums}')
        # print(f'\tavg_sums : {avg_sums}')

        W1_sorted = W1[sorted_indices, :] #sort rows of W1
        W2_sorted = W2[ :, sorted_indices ] #sort columns of W2
        b1_sorted = b1[sorted_indices] #sort b1


        #Re-order weights and bias of lin1 and lin2
        layer.mlp.lin1.weight.data = W1_sorted
        layer.mlp.lin2.weight.data = W2_sorted
        layer.mlp.lin1.bias.data = b1_sorted
    
    return score_dist

def wanda_reordering(model,dataloader):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = model.vision_encoder.to(device)

    hooks_1, hooks_2 = [],[]


    for idx, layer in enumerate(encoder.layers):
        hook_1 = layer.mlp.lin1.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=1)) #module.register_backward_hook)
        hook_2 = layer.mlp.lin2.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=2))
        hooks_1.append(hook_1)
        hooks_2.append(hook_2)
    
    #encoder = nn.DataParallel(encoder)
    encoder.eval()

    with torch.no_grad():
        for idx,(inputs, labels) in enumerate(dataloader):
            data = {'pixel_values': torch.stack([d['pixel_values'].squeeze(0) for d in inputs])}
            #print(f'data["pixel_values"] : {data["pixel_values"].shape}')
            output = encoder(data["pixel_values"].to(device))
    
        for hook_1,hook_2 in zip(hooks_1,hooks_2):
            hook_1.remove()
            hook_2.remove()

    score_dist = {}
    #print(f'Aggregating wanda sums')
    for (k,v),layer in zip(wanda_sums.items(),encoder.layers):
        avg_sums = ((sum(v[0]) / len(v[0])) + (sum(v[1]) / len(v[1]))) / 2
        #print(f'{k} --> {avg_sums.shape}')

        score_dist[k] = avg_sums.flatten().tolist()

        _, sorted_indices = avg_sums.sort(descending=True)

        W1 = layer.mlp.lin1.weight  # shape: (3072, 768)
        W2 = layer.mlp.lin2.weight  # shape: (768, 3072)
        b1 = layer.mlp.lin1.bias

        W1_sorted = W1[sorted_indices, :] #sort rows of W1
        W2_sorted = W2[ :, sorted_indices ] #sort columns of W2
        b1_sorted = b1[sorted_indices] #sort b1


        #Re-order weights and bias of lin1 and lin2
        layer.mlp.lin1.weight.data = W1_sorted
        layer.mlp.lin2.weight.data = W2_sorted
        layer.mlp.lin1.bias.data = b1_sorted
    
    return score_dist


def magnitude_reordering(sam_vit_layers):

    score_dist = {}
    
    for i, layer in enumerate(sam_vit_layers):

        W1 = layer.mlp.lin1.weight  # shape: (3072, 768)
        W2 = layer.mlp.lin2.weight  # shape: (768, 3072)
        b1 = layer.mlp.lin1.bias
        
        row_sums = W1.sum(dim=1)
        column_sums = W2.sum(dim=0)
        avg_sums = (row_sums + column_sums) / 2
        score_dist[i] = avg_sums.flatten().tolist()

        _, sorted_indices = avg_sums.sort(descending=True)

        W1_sorted = W1[sorted_indices, :] #sort rows of W1
        W2_sorted = W2[ :, sorted_indices ] #sort columns of W2
        b1_sorted = b1[sorted_indices] #sort b1


        #Re-order weights and bias of lin1 and lin2
        layer.mlp.lin1.weight.data = W1_sorted
        layer.mlp.lin2.weight.data = W2_sorted
        layer.mlp.lin1.bias.data = b1_sorted
    
    return score_dist


def mask_layers(model, layer_indices_to_mask):
    """
    Masks specified layers in the model by setting their parameters to zero.

    Args:
        model (torch.nn.Module): The model containing the layers.
        layer_indices_to_mask (list): List of layer indices to mask.

    Returns:
        torch.nn.Module: The modified model with masked layers.
    """
    for idx, layer in enumerate(model.vision_encoder.layers):
        if idx in layer_indices_to_mask:
            # Zero out the parameters in the attention sub-layer
            layer.attn.qkv.weight.data.zero_()
            layer.attn.qkv.bias.data.zero_()
            layer.attn.proj.weight.data.zero_()
            layer.attn.proj.bias.data.zero_()

            # Zero out the parameters in the MLP sub-layer
            layer.mlp.lin1.weight.data.zero_()
            layer.mlp.lin1.bias.data.zero_()
            layer.mlp.lin2.weight.data.zero_()
            layer.mlp.lin2.bias.data.zero_()

            # Zero out the LayerNorm parameters if desired (optional)
            layer.layer_norm1.weight.data.zero_()
            layer.layer_norm1.bias.data.zero_()
            layer.layer_norm2.weight.data.zero_()
            layer.layer_norm2.bias.data.zero_()

    return model

def remove_layers(model, layer_indices_to_remove):
    """
    Removes specified layers from the model by their indices.

    Args:
        model (torch.nn.Module): The model containing the layers.
        layer_indices_to_remove (list): List of layer indices to remove.

    Returns:
        torch.nn.Module: The modified model with specified layers removed.
    """
    # Sort the indices in descending order to avoid index shifting issues
    layer_indices_to_remove = sorted(layer_indices_to_remove, reverse=True)
    
    # Iterate over the indices and remove the corresponding layers
    for idx in layer_indices_to_remove:
        del model.vision_encoder.layers[idx]
    
    return model


def sam_weight_reorder(model, dataloader=None, method='magnitude'):
    """_summary_

    Args:
        model (torch.module): Pytorch model
        order (int, optional): Order used to compute importance. Defaults to 0.

    Returns:
        torch.module: Model
    """

    if method == 'wanda':
        score_dist = wanda_reordering(model, dataloader)

    elif method == 'magnitude':
        vision_encoder = model.vision_encoder.cpu()
        sam_vit_layers = vision_encoder.layers
        score_dist = magnitude_reordering(sam_vit_layers)
    elif method == 'movement':
        score_dist = movement_reordering(model,dataloader)
        
    return model, score_dist