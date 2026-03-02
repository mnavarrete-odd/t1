"""
Custom DINOv3 ViT Model for transformers.
This implements the DINOv3 architecture to load the local model weights.
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Optional, Tuple, Union


class DINOv3ViTConfig(PretrainedConfig):
    model_type = "dinov3_vit"
    
    def __init__(
        self,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=1536,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_register_tokens=4,
        rope_theta=100.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_register_tokens = num_register_tokens
        self.rope_theta = rope_theta


class DINOv3ViTModel(PreTrainedModel):
    config_class = DINOv3ViTConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Calculate number of patches
        num_patches = (config.image_size // config.patch_size) ** 2
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # Register tokens
        self.register_tokens = nn.Parameter(
            torch.zeros(1, config.num_register_tokens, config.hidden_size)
        )
        
        # Position embedding (without registers for compatibility)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.hidden_size)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.attention_dropout,
            activation=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.post_init()
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        batch_size = pixel_values.shape[0]
        
        # Patch embedding
        x = self.patch_embed(pixel_values)  # [B, hidden, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding (to CLS + patches)
        x = x + self.pos_embed
        
        # Add register tokens after position embedding
        registers = self.register_tokens.expand(batch_size, -1, -1)
        x = torch.cat((x[:, :1], registers, x[:, 1:]), dim=1)
        
        # Pass through encoder
        x = self.encoder(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Extract CLS token and pool
        cls_token = x[:, 0]
        
        if not return_dict:
            return (x, cls_token)
        
        return BaseModelOutputWithPooling(
            last_hidden_state=x,
            pooler_output=cls_token,
        )
    
    def forward_features(self, pixel_values):
        """Extract features without pooling - for compatibility"""
        output = self.forward(pixel_values, return_dict=True)
        return output.last_hidden_state
