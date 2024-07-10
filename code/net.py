from typing import List
from gradient_reversal import RevGrad
import torch
from torch import nn
from torch.nn import functional as F


class FC(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List = None, drop_out: float = 0.1, gradient_reverse=False):
        super(FC, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [32, 64, 128, 256, 512]
        self.gradient_reverse = gradient_reverse
        self.drop_out = drop_out

        modules = []

        if gradient_reverse:
            modules.append(RevGrad())

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                nn.SELU(),
                nn.Dropout(self.drop_out)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    nn.SELU(),
                    nn.Dropout(self.drop_out)
                )
            )

        self.module = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim, bias=True)
        )

    def forward(self, input):
        self_embed = self.module(input)
        output = self.output_layer(self_embed)

        return output


class DSN(nn.Module):

    def __init__(self, shared_encoder, decoder, input_dim: int, latent_dim: int, alpha: float = 1.0,
                 hidden_dims: List = None, drop_out: float = 0.1, **kwargs) -> None:
        super(DSN, self).__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.drop_out = drop_out

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

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
                    nn.ReLU(),
                    nn.Dropout(self.drop_out)
                )
            )
        modules.append(nn.Dropout(self.drop_out))
        modules.append(nn.Linear(hidden_dims[-1], latent_dim, bias=True))

        self.shared_encoder = shared_encoder
        self.decoder = decoder
        self.private_encoder = nn.Sequential(*modules)

    def forward(self, input):
        z = self.encode(input)
        return [input, self.decode(z), z]

    def p_encode(self, input):
        latent_code = self.private_encoder(input)
        return F.normalize(latent_code, p=2, dim=1)

    def s_encode(self, input):
        latent_code = self.shared_encoder(input)
        return F.normalize(latent_code, p=2, dim=1)

    def encode(self, input):
        p_latent_code = self.p_encode(input)
        s_latent_code = self.s_encode(input)

        return torch.cat((p_latent_code, s_latent_code), dim=1)

    def decode(self, z):
        # embed = self.decoder(z)
        # outputs = self.final_layer(embed)
        outputs = self.decoder(z)

        return outputs




class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        encoded_input = self.encode(input)
        encoded_input = nn.functional.normalize(encoded_input, p=2, dim=1)
        output = self.decoder(encoded_input)

        return output

    def encode(self, input):
        return self.encoder(input)

    def decode(self, z):
        return self.decoder(z)
