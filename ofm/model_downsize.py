import numpy as np
import copy
from torch import nn
import torch
import time
from .utils import calculate_params,structured_pruning
from peft import (
    PeftModel,
    PeftConfig,
    inject_adapter_in_model,
)
from typing import List


__all__ = [
    "bert_module_handler",
    "vit_module_handler",
    "vit_peft_module_handler",
    "arc_config_sampler",
    "sam_module_handler",
    "T5_module_handler",
    "distilbert_module_handler",
    "mamba_module_handler",
]


def copy_weights_to_subnet(subnet, org_model):
    """
    Copies the weights from original foundation model to scaled subnet where the parameter names match.
    Only the overlapping parts of the weights are copied when the dimensions in the subnet
    are less than or equal to those in the larger model.

    Parameters:
    subnet (torch.nn.Module): The smaller model to which the weights will be copied.
    org_model (torch.nn.Module): The foundation model from which the weights will be sourced.

    Usage:
    This function is useful in extract subnet from pre-trained foundation model scenarios where a smaller model is initialized
    with weights from certain layers of a larger, pre-trained model.
    """

    for sm_param_name, sm_param in subnet.named_parameters():
        if sm_param_name in dict(org_model.named_parameters()):
            lg_param = dict(org_model.named_parameters())[sm_param_name]
            if all(
                sm_dim <= lg_dim
                for sm_dim, lg_dim in zip(sm_param.shape, lg_param.shape)
            ):
                # Create a slice object for each dimension to copy the corresponding weights
                slices = tuple(
                    slice(0, min(sm_dim, lg_dim))
                    for sm_dim, lg_dim in zip(sm_param.shape, lg_param.shape)
                )
                sm_param.data.copy_(lg_param.data[slices])


def check_weight_copy_correctness(subnet, org_model):
    """
    Checks if the weights have been correctly copied from the larger model to the smaller model.

    Parameters:
    smaller_model (torch.nn.Module): The smaller model with copied weights.
    larger_model (torch.nn.Module): The larger model from which the weights were sourced.

    Returns:
    bool: True if the weights are correctly copied, False otherwise.

    Usage:
    Useful for verifying the correctness of a weight copying process in model adaptation or transfer learning.
    """

    for sm_param_name, sm_param in subnet.named_parameters():
        if sm_param_name in dict(org_model.named_parameters()):
            lg_param = dict(org_model.named_parameters())[sm_param_name]

            # Compare shapes
            if not all(
                sm_dim <= lg_dim
                for sm_dim, lg_dim in zip(sm_param.shape, lg_param.shape)
            ):
                return False

            # Compare values
            slices = tuple(
                slice(0, min(sm_dim, lg_dim))
                for sm_dim, lg_dim in zip(sm_param.shape, lg_param.shape)
            )
            if not torch.all(sm_param == lg_param[slices]):
                return False

    return True


def arc_config_sampler(elastic_config:dict, n_layer=12,smallest=False,largest=False)->dict:
    """Generate subnet architecture configuration based on the provided configuration.

    Args:
        atten_out_space (list[int]): Attention head output hidden size space, NOT the hidden space.
        inter_hidden_space (list[int]): Intermediate dense hidden layer size space.
        residual_hidden_space (list[int]): Attention (input size) and Intermediate layer (out size) hidden size.
        n_layer (int, optional): Number of multi-head attention layers. Defaults to 12.
        smallest (bool, optional): Either return smallest subnet configuration. Defaults to False.

    Returns:
        dic: Subnet architecture configure.
    """
    
    arc_config = {}
    np.random.seed(int(time.time()))  # Set the seed to the current time

    assert smallest == False or largest == False  # Only one can be true
    assert len(elastic_config) - 1 == n_layer
    
    for layer in range(n_layer):
        layer = str(layer)
        if smallest:
            inter_hidden = min(elastic_config[layer]["inter_hidden_space"])
            atten_out = min(elastic_config[layer]["atten_out_space"])
        elif largest:
            inter_hidden = max(elastic_config[layer]["inter_hidden_space"])
            atten_out = max(elastic_config[layer]["atten_out_space"])
        else:
            inter_hidden = np.random.choice(elastic_config[layer]["inter_hidden_space"]).item()
            atten_out = np.random.choice(elastic_config[layer]["atten_out_space"]).item()

        arc_config[f"{int(layer)}"] = {
            "atten_out": atten_out,
            "inter_hidden": inter_hidden,
            "residual_hidden": elastic_config[layer]["residual_hidden"],
        }
    
    def sample_layer_indices(target_layer_idx, prob):
        import random
        sampled_layer_idx = []
        for idx, p in zip(target_layer_idx, prob):
            if random.random() < p:
                sampled_layer_idx.append(idx)
        return sampled_layer_idx

    layer_elastic = elastic_config['layer_elastic']
    if smallest:
        remove_layer_idx = layer_elastic['elastic_layer_idx']
    elif largest:
        remove_layer_idx = []
    else:
        remove_layer_idx = sample_layer_indices(layer_elastic['elastic_layer_idx'], layer_elastic['remove_layer_prob'])
    
    arc_config['remove_layer_idx'] = remove_layer_idx
    
    
    return arc_config

def clip_module_handler(model, arc_config):
    from transformers.models.clip.modeling_clip import (
        CLIPEncoderLayer,
    )

    text_arc_config, vision_arc_config = arc_config
    subnet = copy.deepcopy(model).cpu()
    text_encoder_layers = subnet.text_model.encoder.layers
    vision_encoder_layers = subnet.vision_model.encoder.layers

    new_text_config = copy.deepcopy(subnet.config.text_config)
    new_vision_config = copy.deepcopy(subnet.config.vision_config)

    subnet.config.text_architecture = text_arc_config
    subnet.config.vision_architecture = vision_arc_config

    for i, (layer, key) in enumerate(zip(text_encoder_layers, text_arc_config)):
        arc = text_arc_config[key]

        new_text_config.intermediate_size = arc["inter_hidden"]

        new_layer = CLIPEncoderLayer(config=new_text_config)

        subnet.text_model.encoder.layers[i] = new_layer

    for i, (layer, key) in enumerate(zip(vision_encoder_layers, vision_arc_config)):
        arc = vision_arc_config[key]

        new_vision_config.intermediate_size = arc["inter_hidden"]

        new_layer = CLIPEncoderLayer(config=new_vision_config)

        subnet.vision_model.encoder.layers[i] = new_layer

    copy_weights_to_subnet(subnet, model)

    return subnet, calculate_params(subnet)


def mamba_module_handler(model, arc):
    from transformers.models.mamba.modeling_mamba import (
        MambaCache as OriginalMambaCache,
    )

    class MambaCache(OriginalMambaCache):
        def __init__(self, config, batch_size, dtype=torch.float16, device=None):
            self.seqlen_offset = 0
            self.dtype = dtype
            intermediate_size = config.intermediate_size
            ssm_state_size = config.state_size
            conv_kernel_size = config.conv_kernel
            if hasattr(config, "architecture"):
                self.conv_states = {
                    i: torch.zeros(
                        batch_size,
                        config.architecture[layer_arc]["inter_hidden"],
                        conv_kernel_size,
                        device=device,
                        dtype=dtype,
                    )
                    for i, layer_arc in zip(
                        range(config.num_hidden_layers), config.architecture
                    )
                }
                self.ssm_states = {
                    i: torch.zeros(
                        batch_size,
                        config.architecture[layer_arc]["inter_hidden"],
                        ssm_state_size,
                        device=device,
                        dtype=dtype,
                    )
                    for i, layer_arc in zip(
                        range(config.num_hidden_layers), config.architecture
                    )
                }
            else:
                self.conv_states = {
                    i: torch.zeros(
                        batch_size,
                        intermediate_size,
                        conv_kernel_size,
                        device=device,
                        dtype=dtype,
                    )
                    for i in range(config.num_hidden_layers)
                }
                self.ssm_states = {
                    i: torch.zeros(
                        batch_size,
                        intermediate_size,
                        ssm_state_size,
                        device=device,
                        dtype=dtype,
                    )
                    for i in range(config.num_hidden_layers)
                }

    import transformers.models.mamba.modeling_mamba

    transformers.models.mamba.modeling_mamba.MambaCache = MambaCache

    from transformers.models.mamba.modeling_mamba import MambaBlock

    
    new_model = copy.deepcopy(model)
    new_model.config.architecture = arc

    for idx, (layer, layer_arc) in enumerate(zip(model.backbone.layers, arc)):

        layer_config = copy.deepcopy(new_model.config)
        layer_config.intermediate_size = arc[layer_arc]["inter_hidden"]
        new_layer = MambaBlock(config=layer_config, layer_idx=idx)

        copy_weights_to_subnet(new_layer, layer)

        new_model.backbone.layers[idx] = new_layer

    total_params = calculate_params(new_model)
    copy_weights_to_subnet(new_model, model)

    return new_model, total_params


def bert_module_handler(model, arc_config):
    from transformers.models.bert.modeling_bert import (
        BertSelfAttention,
        BertSelfOutput,
        BertIntermediate,
        BertOutput,
        BertEmbeddings,
        BertPooler,
    )
    from transformers import BertConfig

    BertLayerNorm = nn.LayerNorm

    class BertSelfAttention(BertSelfAttention):
        def __init__(self, config):
            super().__init__(config)

            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = config.attention_head_size
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    class BertSelfOutput(BertSelfOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(
                config.attention_head_size * config.num_attention_heads,
                config.hidden_size,
            )
            self.LayerNorm = BertLayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )

    class BertIntermediate(BertIntermediate):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    class BertOutput(BertOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
            self.LayerNorm = BertLayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )

    subnetwork = copy.deepcopy(model).cpu()

    bert_layers = subnetwork.bert.encoder.layer

    new_config = BertConfig.from_dict(model.config.to_dict())

    for i, (layer, key) in enumerate(zip(bert_layers, arc_config)):
        arc = arc_config[key]
        # new_config.hidden_size = arc  # Set to the new output dimension
        new_config.attention_head_size = (
            arc["atten_out"] // new_config.num_attention_heads
        )  # Ensure it divides evenly
        new_config.intermediate_size = arc["inter_hidden"]
        new_config.hidden_size = arc["residual_hidden"]

        new_attention_layer = BertSelfAttention(config=new_config)
        new_out_layer = BertSelfOutput(config=new_config)
        new_inter_layer = BertIntermediate(config=new_config)
        new_dens_out_layer = BertOutput(config=new_config)

        layer.attention.self = new_attention_layer
        layer.attention.output = new_out_layer
        layer.intermediate = new_inter_layer
        layer.output = new_dens_out_layer

    new_embeddings = BertEmbeddings(new_config)
    subnetwork.bert.embeddings = new_embeddings

    # Sequence classification model
    if hasattr(subnetwork, "classifier"):
        new_pooler = BertPooler(new_config)
        new_classifier = nn.Linear(
            new_config.hidden_size, model.classifier.out_features
        )
        subnetwork.bert.pooler = new_pooler
        subnetwork.classifier = new_classifier
    # Question answering model
    if hasattr(subnetwork, "qa_outputs"):
        new_qa_outputs = nn.Linear(
            new_config.hidden_size, model.qa_outputs.out_features
        )
        subnetwork.qa_outputs = new_qa_outputs

    subnetwork.config = new_config
    copy_weights_to_subnet(subnetwork, model)

    total_params = calculate_params(subnetwork)

    return subnetwork, total_params


def vit_module_handler(model, arc_config):
    from transformers.models.vit.modeling_vit import (
        ViTSelfAttention,
        ViTSelfOutput,
        ViTIntermediate,
        ViTOutput,
        ViTEmbeddings,
    )
    from transformers import ViTConfig
    from torch import nn

    class ViTSelfAttention(ViTSelfAttention):
        def __init__(self, config: ViTConfig):
            super().__init__(config)

            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = config.attention_head_size
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            self.query = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )
            self.key = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )
            self.value = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )

            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    class ViTSelfOutput(ViTSelfOutput):
        def __init__(self, config: ViTConfig):
            super().__init__(config)
            self.dense = nn.Linear(
                config.attention_head_size * config.num_attention_heads,
                config.hidden_size,
            )
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    class ViTIntermediate(ViTIntermediate):
        def __init__(self, config: ViTConfig):
            super().__init__(config)
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    class ViTOutput(ViTOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

    subnetwork = copy.deepcopy(model).cpu()

    vit_layers = subnetwork.vit.encoder.layer
    new_config = ViTConfig.from_dict(model.config.to_dict())

    for i, (layer, key) in enumerate(zip(vit_layers, arc_config)):
        arc = arc_config[key]
        # new_config.hidden_size = arc  # Set to the new output dimension
        new_config.attention_head_size = (
            arc["atten_out"] // new_config.num_attention_heads
        )  # Ensure it divides evenly
        new_config.intermediate_size = arc["inter_hidden"]
        new_config.hidden_size = arc["residual_hidden"]

        new_attention_layer = ViTSelfAttention(config=new_config)
        new_out_layer = ViTSelfOutput(config=new_config)
        new_inter_layer = ViTIntermediate(config=new_config)
        new_dens_out_layer = ViTOutput(config=new_config)
        layernorm_before = nn.LayerNorm(
            new_config.hidden_size, eps=new_config.layer_norm_eps
        )
        layernorm_after = nn.LayerNorm(
            new_config.hidden_size, eps=new_config.layer_norm_eps
        )

        layer.attention.attention = new_attention_layer
        layer.attention.output = new_out_layer
        layer.intermediate = new_inter_layer
        layer.output = new_dens_out_layer
        layer.layernorm_before = layernorm_before
        layer.layernorm_after = layernorm_after

    new_embeddings = ViTEmbeddings(new_config)
    new_layernorm = nn.LayerNorm(new_config.hidden_size, eps=new_config.layer_norm_eps)
    new_classifier = nn.Linear(new_config.hidden_size, model.classifier.out_features)

    subnetwork.vit.embeddings = new_embeddings
    subnetwork.vit.layernorm = new_layernorm
    subnetwork.classifier = new_classifier

    subnetwork.config = new_config
    copy_weights_to_subnet(subnetwork, model)

    total_params = calculate_params(subnetwork)

    return subnetwork, total_params


def swin_module_handler(model, arc_config):
    from transformers.models.swin.modeling_swin import (
        SwinStage,
        SwinPatchMerging,
        SwinDropPath,
        SwinIntermediate,
        SwinLayer,
        SwinAttention,
        SwinOutput,
        SwinSelfAttention,
        SwinSelfOutput,
        ACT2FN,
    )

    class SwinOutput(SwinOutput):
        def __init__(self, config, input_dim, dim, dense_out=None):
            super().__init__(config, dim)
            self.dense = nn.Linear(input_dim, dim)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    class SwinIntermediate(SwinIntermediate):
        def __init__(self, config, dim, out_dim):
            super().__init__(config, dim)
            self.dense = nn.Linear(dim, out_dim)
            if isinstance(config.hidden_act, str):
                self.intermediate_act_fn = ACT2FN[config.hidden_act]
            else:
                self.intermediate_act_fn = config.hidden_act

    class SwinSelfOutput(SwinSelfOutput):
        def __init__(self, config, dim, out_dim):
            super().__init__(config, dim)
            self.dense = nn.Linear(dim, out_dim)
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    class SwinAttention(SwinAttention):
        def __init__(self, config, dim, out_dim, num_heads, window_size):
            super().__init__(config, dim, num_heads, window_size)
            self.self = SwinSelfAttention(config, dim, num_heads, window_size)
            self.output = SwinSelfOutput(config, dim, out_dim)
            self.pruned_heads = set()

    class SwinLayer(SwinLayer):
        def __init__(
            self,
            config,
            dim,
            atten_out,
            interm_out,
            input_resolution,
            num_heads,
            shift_size=0,
        ):
            super().__init__(config, dim, input_resolution, num_heads, shift_size)

            self.attention = SwinAttention(
                config, dim, atten_out, num_heads, window_size=self.window_size
            )
            self.drop_path = (
                SwinDropPath(config.drop_path_rate)
                if config.drop_path_rate > 0.0
                else nn.Identity()
            )
            self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
            self.intermediate = SwinIntermediate(config, atten_out, interm_out)
            self.output = SwinOutput(config, interm_out, dim)

    subnet = copy.deepcopy(model).cpu()
    swin_backbone_layers = model.swin.encoder.layers[-2].blocks
    subnet.config.ofm_architecture = arc_config
    new_config = copy.deepcopy(subnet.config)

    for i, (layer, key) in enumerate(zip(swin_backbone_layers, arc_config)):
        arc = arc_config[key]
        new_layer = SwinLayer(
            config=new_config,
            dim=new_config.embed_dim * 2**2,  # only on third (index 2) stage
            atten_out=arc["atten_out"],
            interm_out=arc["inter_hidden"],
            input_resolution=layer.input_resolution,
            num_heads=new_config.num_heads[2],
            shift_size=0 if (i % 2 == 0) else new_config.window_size // 2,
        )
        subnet.swin.encoder.layers[-2].blocks[i] = new_layer
    total_params = calculate_params(subnet)
    subnet.config.num_parameters = total_params

    copy_weights_to_subnet(subnet, model)

    return subnet, total_params


def sam_module_handler(model, arc_config):
    from transformers.models.sam.modeling_sam import (
        SamVisionAttention,
        SamMLPBlock,
        SamVisionLayer,
    )
    from transformers import SamVisionConfig

    sub_model = copy.deepcopy(model).cpu()
    vision_encoder = copy.deepcopy(sub_model.vision_encoder).cpu()

    sam_vit_layers = vision_encoder.layers

    class SamVisionAttention(SamVisionAttention):
        def __init__(self, config, window_size):
            import torch

            super().__init__(config, window_size)
            input_size = (
                (
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                )
                if window_size == 0
                else (window_size, window_size)
            )

            self.num_attention_heads = config.num_attention_heads
            # head_dim = config.hidden_size // config.num_attention_heads
            head_dim = config.attention_head_size

            self.scale = head_dim**-0.5
            self.dropout = config.attention_dropout

            self.qkv = nn.Linear(
                config.hidden_size,
                head_dim * self.num_attention_heads * 3,
                bias=config.qkv_bias,
            )
            self.proj = nn.Linear(
                head_dim * self.num_attention_heads, config.hidden_size
            )

            self.use_rel_pos = config.use_rel_pos
            if self.use_rel_pos:
                if input_size is None:
                    raise ValueError(
                        "Input size must be provided if using relative positional encoding."
                    )

                # initialize relative positional embeddings
                self.rel_pos_h = nn.Parameter(
                    torch.zeros(2 * input_size[0] - 1, head_dim)
                )
                self.rel_pos_w = nn.Parameter(
                    torch.zeros(2 * input_size[1] - 1, head_dim)
                )

    class SamMLPBlock(SamMLPBlock):
        def __init__(self, config):
            super().__init__(config)

            self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
            self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
    
    ## Remove layer index
    remove_layer_idx = arc_config['remove_layer_idx']
    print(f'arc_config : {arc_config}')
    # for i, (layer, key) in enumerate(zip(sam_vit_layers, arc_config)):
    for i, layer in enumerate(sam_vit_layers):
        if i in remove_layer_idx:
            # for layer been pruned, simply skip it here
            continue
        arc = arc_config[str(i)]
        new_config = SamVisionConfig.from_dict(vision_encoder.config.to_dict())

        # new_config.hidden_size = arc  # Set to the new output dimension
        new_config.attention_head_size = (
            arc["atten_out"] // new_config.num_attention_heads
        )  # Ensure it divides evenly

        new_config.mlp_dim = arc["inter_hidden"]
        new_attention_layer = SamVisionAttention(
            config=new_config,
            window_size=(
                new_config.window_size if i not in new_config.global_attn_indexes else 0
            ),
        )

        new_mlp = SamMLPBlock(config=new_config)

        layer.attn = new_attention_layer
        layer.mlp = new_mlp

    sub_model.vision_encoder = vision_encoder
    
    
    

    copy_weights_to_subnet(sub_model, model)
    sub_model, _, _ = structured_pruning(sub_model,remove_layer_idx,model.vision_encoder.config.global_attn_indexes)

    total_params = calculate_params(sub_model)

    return sub_model, total_params


def t5_module_handler(model, arc_config):
    from transformers.models.t5.modeling_t5 import (
        T5Config,
        T5LayerSelfAttention,
        T5LayerCrossAttention,
        T5LayerFF,
        T5LayerNorm,
    )

    subnetwork = copy.deepcopy(model).cpu()
    # return subnetwork, calculate_params(subnetwork)
    encoder_layers = subnetwork.encoder.block
    new_config = T5Config.from_dict(model.config.to_dict())

    for i, (layer, key) in enumerate(zip(encoder_layers, arc_config)):
        arc = arc_config[key]
        # new_config.hidden_size = arc  # Set to the new output dimension
        new_config.d_kv = arc["atten_out"] // new_config.num_heads
        new_config.d_ff = arc["inter_hidden"]
        new_config.d_model = arc["residual_hidden"]

        layer.layer[0] = T5LayerSelfAttention(
            new_config, has_relative_attention_bias=bool(i == 0)
        )
        layer.layer[1] = T5LayerFF(new_config)

    decoder_layers = subnetwork.decoder.block
    for i, (layer, key) in enumerate(zip(decoder_layers, arc_config)):
        arc = arc_config[key]
        # new_config.hidden_size = arc  # Set to the new output dimension
        new_config.d_kv = arc["atten_out"] // new_config.num_heads
        new_config.d_ff = arc["inter_hidden"]
        new_config.d_model = arc["residual_hidden"]

        # layer.layer[0] = T5LayerSelfAttention(
        #     new_config, has_relative_attention_bias=bool(i == 0)
        # )
        # layer.layer[1] = T5LayerCrossAttention(new_config)
        layer.layer[2] = T5LayerFF(new_config)

    subnetwork.shared = nn.Embedding(new_config.vocab_size, new_config.d_model)
    # subnetwork.encoder.embed_tokens = subnetwork.shared
    # subnetwork.decoder.embed_tokens = subnetwork.shared
    subnetwork.encoder.final_layer_norm = T5LayerNorm(
        new_config.d_model, eps=new_config.layer_norm_epsilon
    )
    subnetwork.decoder.final_layer_norm = T5LayerNorm(
        new_config.d_model, eps=new_config.layer_norm_epsilon
    )
    subnetwork.lm_head = nn.Linear(
        new_config.d_model, new_config.vocab_size, bias=False
    )
    subnetwork.config = new_config
    copy_weights_to_subnet(subnetwork, model)

    total_params = calculate_params(subnetwork)
    return subnetwork, total_params


def roberta_module_handler(model, arc_config):
    from transformers.models.roberta.modeling_roberta import (
        RobertaSelfAttention,
        RobertaSelfOutput,
        RobertaIntermediate,
        RobertaOutput,
        RobertaEmbeddings,
        RobertaConfig,
        RobertaClassificationHead,
    )

    class RobertaSelfAttention(RobertaSelfAttention):
        def __init__(self, config):
            super().__init__(config)

            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = config.attention_head_size
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    class RobertaSelfOutput(RobertaSelfOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(
                config.attention_head_size * config.num_attention_heads,
                config.hidden_size,
            )
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    subnetwork = copy.deepcopy(model).cpu()
    roberta_layers = subnetwork.roberta.encoder.layer
    new_config = RobertaConfig.from_dict(model.config.to_dict())

    for i, (layer, key) in enumerate(zip(roberta_layers, arc_config)):
        arc = arc_config[key]
        new_config.attention_head_size = (
            arc["atten_out"] // new_config.num_attention_heads
        )
        new_config.intermediate_size = arc["inter_hidden"]
        new_config.hidden_size = arc["residual_hidden"]

        new_attention_layer = RobertaSelfAttention(config=new_config)
        new_out_layer = RobertaSelfOutput(config=new_config)
        new_inter_layer = RobertaIntermediate(config=new_config)
        new_dens_out_layer = RobertaOutput(config=new_config)

        layer.attention.self = new_attention_layer
        layer.attention.output = new_out_layer
        layer.intermediate = new_inter_layer
        layer.output = new_dens_out_layer

    new_embeddings = RobertaEmbeddings(new_config)
    subnetwork.roberta.embeddings = new_embeddings

    if hasattr(subnetwork, "classifier"):
        new_classifer = RobertaClassificationHead(new_config)
        subnetwork.classifier = new_classifer
    if hasattr(subnetwork, "qa_outputs"):
        new_qa_outputs = nn.Linear(new_config.hidden_size, new_config.num_labels)
        subnetwork.qa_outputs = new_qa_outputs

    subnetwork.config = new_config
    copy_weights_to_subnet(subnetwork, model)

    total_params = calculate_params(subnetwork)

    return subnetwork, total_params


def distilbert_module_handler(model, arc_config):
    from transformers.models.distilbert.modeling_distilbert import (
        DistilBertConfig,
        Embeddings,
    )

    subnetwork = copy.deepcopy(model).cpu()
    distilbert_layers = subnetwork.distilbert.transformer.layer
    new_config = DistilBertConfig.from_dict(model.config.to_dict())

    for i, (layer, key) in enumerate(zip(distilbert_layers, arc_config)):
        arc = arc_config[key]
        new_config.attention_head_size = (
            arc["atten_out"] // new_config.num_attention_heads
        )
        new_config.dim = arc["residual_hidden"]
        new_config.hidden_dim = arc["inter_hidden"]

        layer.attention.q_lin = nn.Linear(
            new_config.dim,
            new_config.attention_head_size * new_config.num_attention_heads,
        )
        layer.attention.k_lin = nn.Linear(
            new_config.dim,
            new_config.attention_head_size * new_config.num_attention_heads,
        )
        layer.attention.v_lin = nn.Linear(
            new_config.dim,
            new_config.attention_head_size * new_config.num_attention_heads,
        )
        layer.attention.out_lin = nn.Linear(
            new_config.attention_head_size * new_config.num_attention_heads,
            new_config.dim,
        )
        layer.sa_layer_norm = nn.LayerNorm(new_config.dim, eps=layer.sa_layer_norm.eps)

        layer.ffn.lin1 = nn.Linear(new_config.dim, new_config.hidden_dim)
        layer.ffn.lin2 = nn.Linear(new_config.hidden_dim, new_config.dim)
        layer.output_layer_norm = nn.LayerNorm(
            new_config.dim, eps=layer.output_layer_norm.eps
        )

    new_embeddings = Embeddings(new_config)
    subnetwork.distilbert.embeddings = new_embeddings

    if hasattr(subnetwork, "pre_classifier"):
        new_pre_classifier = nn.Linear(new_config.dim, new_config.dim)
        subnetwork.pre_classifier = new_pre_classifier
    if hasattr(subnetwork, "classifier"):
        new_classifier = nn.Linear(new_config.dim, new_config.num_labels)
        subnetwork.classifier = new_classifier
    if hasattr(subnetwork, "qa_outputs"):
        new_qa_outputs = nn.Linear(new_config.dim, new_config.num_labels)
        subnetwork.qa_outputs = new_qa_outputs

    subnetwork.config = new_config
    copy_weights_to_subnet(subnetwork, model)

    total_params = calculate_params(subnetwork)
    return subnetwork, total_params


def vit_peft_module_handler(model: PeftModel, peft_config: PeftConfig, arc_config):
    from transformers.models.vit.modeling_vit import (
        ViTSelfAttention,
        ViTSelfOutput,
        ViTIntermediate,
        ViTOutput,
    )
    from transformers import ViTConfig

    class ViTSelfAttention(ViTSelfAttention):
        def __init__(self, config: ViTConfig):
            super().__init__(config)

            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = config.attention_head_size
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            self.query = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )
            self.key = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )
            self.value = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )

            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    class ViTSelfOutput(ViTSelfOutput):
        def __init__(self, config: ViTConfig):
            super().__init__(config)
            self.dense = nn.Linear(
                config.attention_head_size * config.num_attention_heads,
                config.hidden_size,
            )
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    class ViTIntermediate(ViTIntermediate):
        def __init__(self, config: ViTConfig):
            super().__init__(config)
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    class ViTOutput(ViTOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

    subnetwork = copy.deepcopy(model).cpu()

    vit_layers = subnetwork.vit.encoder.layer

    for i, (layer, key) in enumerate(zip(vit_layers, arc_config)):
        arc = arc_config[key]
        new_config = ViTConfig.from_dict(model.config.to_dict())
        # new_config.hidden_size = arc  # Set to the new output dimension
        new_config.attention_head_size = (
            arc["atten_out"] // new_config.num_attention_heads
        )  # Ensure it divides evenly
        new_config.intermediate_size = arc["inter_hidden"]
        new_attention_layer = ViTSelfAttention(config=new_config).requires_grad_(False)
        new_out_layer = ViTSelfOutput(config=new_config).requires_grad_(False)
        new_inter_layer = ViTIntermediate(config=new_config).requires_grad_(False)
        new_dens_out_layer = ViTOutput(config=new_config).requires_grad_(False)

        if any(
            item in peft_config.target_modules for item in ["query", "key", "value"]
        ):
            new_attention_layer = inject_adapter_in_model(
                peft_config, new_attention_layer
            )

        if "dense" in peft_config.target_modules:
            new_out_layer = inject_adapter_in_model(peft_config, new_out_layer)
            new_inter_layer = inject_adapter_in_model(peft_config, new_inter_layer)
            new_dens_out_layer = inject_adapter_in_model(
                peft_config, new_dens_out_layer
            )

        layer.attention.attention = new_attention_layer
        layer.attention.output = new_out_layer
        layer.intermediate = new_inter_layer
        layer.output = new_dens_out_layer

    copy_weights_to_subnet(subnetwork, model)
    # total_params = calculate_params(subnetwork)
    trainable_params, all_param = subnetwork.get_nb_trainable_parameters()

    return subnetwork, trainable_params
