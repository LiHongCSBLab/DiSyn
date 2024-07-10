from typing import List

import torch
from torch import nn
from torch.nn import functional as F


class DiSyn(nn.Module):

    def __init__(self, shared_encoder, adapter, decoder, input_dim: int, latent_dim: int, alpha: float = 1.0,
                 hidden_dims: List = None, drop_out: float = 0.1, noise_flag: bool = False, norm_flag: bool = False):
        super(DiSyn, self).__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.noise_flag = noise_flag
        self.drop_out = drop_out
        self.norm_flag = norm_flag

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.shared_encoder = shared_encoder
        self.adapter = adapter
        self.decoder = decoder

        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                # nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(self.drop_out)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    # nn.Dropout(0.1),
                    # nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(self.drop_out)
                )
            )
        modules.append(nn.Dropout(self.drop_out))
        modules.append(nn.Linear(hidden_dims[-1], latent_dim, bias=True))

        self.private_encoder = nn.Sequential(*modules)

    def p_encode(self, input):
        if self.noise_flag and self.training:
            latent_code = self.private_encoder(input + torch.randn_like(input, requires_grad=False) * 0.1)
        else:
            latent_code = self.private_encoder(input)

        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def s_encode(self, input):
        latent_code = self.shared_encoder(input)
        latent_code = self.adapter(latent_code)

        return F.normalize(latent_code, p=2, dim=1)

    def encode(self, input):
        p_latent_code = self.p_encode(input)
        s_latent_code = self.s_encode(input)
        # (private, shared)
        return torch.cat((p_latent_code, s_latent_code), dim=1)

    def decode(self, z):
        # embed = self.decoder(z)
        # outputs = self.final_layer(embed)
        outputs = self.decoder(z)

        return outputs

    def forward(self, input):
        z = self.encode(input)
        return [input, self.decode(z), z]
