import torch
import torch.nn as nn
import einops

class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, token_dim=256, num_heads=3):
        super().__init__()

        #self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        #self.to_key = nn.Linear(149, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.w_g_new = nn.Parameter(torch.randn(1,149))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        #self.final = nn.Linear(token_dim * num_heads, 768)

    def forward(self, x,y):
        #query = self.to_query(x)#BxNxD 32x149x768
        #key = self.to_key(y)#32x1x768
        query=x
        key=y#32x1x768

        #query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        #key = torch.nn.functional.normalize(key, dim=-1) #BxNxD

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        #query_weight_new = self.w_g_new @ query # BxNx1 (1xD @ BxNxD)
        A = query_weight * self.scale_factor # BxNx1 32x149x1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1 32x149x1

        G = torch.sum(A * query, dim=1) # BxD 32x768

        '''G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD 32x1x768'''
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[2]
        ) # BxNxD 32x768x768自己想的


        #out = self.Proj(G * key) + query #BxNxD
        #out = self.Proj(G * key) #BxNxD
        out = self.Proj(key@G) #BxNxD

        #out = self.final(out) # BxNxD

        return out
