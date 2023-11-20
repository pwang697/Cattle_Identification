import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from timm.models.vision_transformer import VisionTransformer
import torch.nn.functional as F


class TripletViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embedding_size=16):
        super(TripletViT, self).__init__()
        self.model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,  # Number of input channels (RGB images)
            num_classes=num_classes,
            embed_dim=embedding_size,  # Output embedding size
            depth=12,  # Number of transformer blocks
            num_heads=1,  # Number of attention heads
            mlp_ratio=4.0,  # Ratio of MLP hidden dim to embedding dim
            qkv_bias=True,  # Use bias in qkv (query, key, value) linear layers
            drop_rate=0.1,  # Dropout rate
            attn_drop_rate=0.0,  # Attention dropout rate
            drop_path_rate=0.1,  # Stochastic depth rate
            norm_layer=nn.LayerNorm,  # Normalization layer
        )

        # Additional layers for embedding
        self.fc_embedding = nn.Linear(self.model.num_classes, embedding_size)

    def forward_sibling(self, x):
        x = self.model(x)
        # x = x.view(x.size(0), -1)
        # x = x['embedding']  # Extract the 'embedding' tensor from the output dictionary
        x_embedding = self.fc_embedding(x)
        return x_embedding

    def forward(self, input1, input2, input3):
        embedding_vec_1 = self.forward_sibling(input1)
        embedding_vec_2 = self.forward_sibling(input2)
        embedding_vec_3 = self.forward_sibling(input3)

        return embedding_vec_1, embedding_vec_2, embedding_vec_3