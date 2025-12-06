from einops import  repeat
from einops.layers.torch import Rearrange
from torch import nn,randn,cat
from torch.nn import Module
from Transformer import Transformer

class ViT(Module):
    def __init__(self, *,
                 image_size,
                 patch_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool = 'cls', # 'cls', 'mean'
                 channels = 3,
                 dim_head = 64,
                 dropout = 0.,
                 emb_dropout = 0.):
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        if pool == 'cls':
            num_cls_tokens = 1
        else:
            #'mean'
            num_cls_tokens = 0

        self.to_patch_embedding = self._init_patch_embedding(patch_dim,dim,patch_height,patch_width)
        self.cls_token = nn.Parameter(randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(randn(num_patches + num_cls_tokens, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def _init_patch_embedding(self,patch_dim,dim,patch_height,patch_width):
        return nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, image):
        batch = image.shape[0]
        x = self.to_patch_embedding(image)
        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        x = cat((cls_tokens, x), dim = 1)
        seq = x.shape[1]

        x = self.transformer(self.dropout(x + self.pos_embedding[:seq]))

        if self.pool == 'mean':
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        return self.mlp_head(self.to_latent(x))