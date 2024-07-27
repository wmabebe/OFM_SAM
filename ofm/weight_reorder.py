def sam_weight_reorder(model, order=0):
    """_summary_

    Args:
        model (torch.module): Pytorch model
        order (int, optional): Order used to compute importance. Defaults to 0.

    Returns:
        torch.module: Model
    """

    vision_encoder = model.vision_encoder.cpu()

    sam_vit_layers = vision_encoder.layers

    score_dist = {}

    for i, layer in enumerate(sam_vit_layers):

        W1 = layer.mlp.lin1.weight  # shape: (3072, 768)
        W2 = layer.mlp.lin2.weight  # shape: (768, 3072)
        b1 = layer.mlp.lin1.bias

        if order == 0:
            row_sums = W1.sum(dim=1)
            score_dist[i] = row_sums

        _, sorted_indices = row_sums.sort(descending=True)

        W1_sorted = W1[sorted_indices, :] #sort rows of W1
        W2_sorted = W2[ :, sorted_indices ] #sort columns of W2
        b1_sorted = b1[sorted_indices] #sort b1


        #Re-order weights and bias of lin1 and lin2
        layer.mlp.lin1.weight.data = W1_sorted
        layer.mlp.lin2.weight.data = W2_sorted
        layer.mlp.lin1.bias.data = b1_sorted

    return model, score_dist