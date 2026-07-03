import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.masks import make_synaptic_mask, make_cable_mask, count_mask_params


class MaskedLinear(nn.Module):
    """
    A linear layer that enforces sparse connectivity via a binary mask.
    Only connections where mask=1 are allowed to have non-zero weights.
    """
    def __init__(self, in_features, out_features, mask):
        super(MaskedLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # learnable parameters
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # mask is fixed — not a learnable parameter
        self.register_buffer('mask', mask)

        # apply mask at initialisation
        with torch.no_grad():
            self.weight.data *= self.mask

    def forward(self, x):
        # apply mask to weights before every forward pass
        return F.linear(x, self.weight * self.mask, self.bias)

    def enforce_mask(self):
        """
        Zero out any weights that should not exist.
        Must be called after every optimizer step.
        This is Equation 3 from the paper applied to weights.
        """
        with torch.no_grad():
            self.weight.data *= self.mask


class DendriticANN(nn.Module):
    """
    Dendritic ANN with two sparse masked hidden layers.

    Architecture:
        Input (1024)
            ↓ synaptic mask (sparse)
        Dendritic layer — LeakyReLU
            ↓ cable mask (sparse)
        Somatic layer — LeakyReLU
            ↓ fully connected
        Output (10)
    """
    def __init__(self, n_somata, n_dendrites_per_soma,
                 sampling_mode='random', n_synapses=16,
                 n_inputs=1024, n_classes=10, image_size=32):
        super(DendriticANN, self).__init__()

        self.n_somata            = n_somata
        self.n_dendrites_per_soma = n_dendrites_per_soma
        self.n_dendrites         = n_somata * n_dendrites_per_soma
        self.sampling_mode       = sampling_mode

        # build masks
        synaptic_mask = make_synaptic_mask(
            n_dendrites  = self.n_dendrites,
            n_inputs     = n_inputs,
            n_synapses   = n_synapses,
            mode         = sampling_mode,
            image_size   = image_size
        )
        cable_mask = make_cable_mask(n_somata, n_dendrites_per_soma)

        # layer 1: input → dendrites (sparse)
        self.dendritic_layer = MaskedLinear(
            n_inputs, self.n_dendrites, synaptic_mask
        )

        # layer 2: dendrites → somata (sparse)
        self.somatic_layer = MaskedLinear(
            self.n_dendrites, n_somata, cable_mask
        )

        # layer 3: somata → output (fully connected)
        self.output_layer = nn.Linear(n_somata, n_classes)

        # activation
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        # flatten image: [batch, 1, 32, 32] → [batch, 1024]
        x = x.view(x.size(0), -1)

        # dendritic layer
        x = self.activation(self.dendritic_layer(x))

        # somatic layer
        x = self.activation(self.somatic_layer(x))

        # output layer
        x = self.output_layer(x)
        return x

    def enforce_masks(self):
        """Call after every optimizer step to maintain sparsity."""
        self.dendritic_layer.enforce_mask()
        self.somatic_layer.enforce_mask()

    def count_parameters(self):
        # for masked layers, only count non-zero weights (active connections)
        # plus all bias terms
        total = 0
    
        # dendritic layer: active synaptic weights + biases
        total += int(self.dendritic_layer.mask.sum().item())
        total += self.dendritic_layer.bias.numel()

        # somatic layer: active cable weights + biases
        total += int(self.somatic_layer.mask.sum().item())
        total += self.somatic_layer.bias.numel()

        # output layer: fully connected, count everything
        total += self.output_layer.weight.numel()
        total += self.output_layer.bias.numel()
    
        return total
    
def find_matching_config(target_params, n_synapses, n_somata_options=None, dendrites_per_soma_range=None):
    """Search for n_somata and n_dendrites_per_soma that gets closest to target_params."""
    if n_somata_options is None:
        n_somata_options = [256, 512, 1024]
    if dendrites_per_soma_range is None:
        dendrites_per_soma_range = range(1, 128)

    best = None
    best_diff = float('inf')

    for n_somata in n_somata_options:
        for n_dps in dendrites_per_soma_range:
            n_dendrites = n_somata * n_dps
            params = (n_dendrites * n_synapses   # synaptic weights
                    + n_dendrites                 # synaptic biases
                    + n_dendrites                 # cable weights
                    + n_somata                    # somatic biases
                    + n_somata * 10 + 10)         # output layer
            diff = abs(params - target_params)
            if diff < best_diff:
                best_diff = diff
                best = (n_somata, n_dps, params)

    return best


if __name__ == "__main__":
    print("Testing dANN variants...\n")

    configs = [
        ('random',  16, 'dANN-R'),
        ('lrf_4x4', 16, 'dANN-LRF-4x4'),
        ('lrf_4x8', 32, 'dANN-LRF-4x8'),
    ]

    for mode, n_synapses, name in configs:
        model = DendriticANN(
            n_somata             = 512,
            n_dendrites_per_soma = 8,
            sampling_mode        = mode,
            n_synapses           = n_synapses
        )
        dummy = torch.randn(8, 1, 32, 32)
        out   = model(dummy)
        print(f"{name}")
        print(f"  Trainable parameters: {model.count_parameters():,}")
        print(f"  Input shape:          {dummy.shape}")
        print(f"  Output shape:         {out.shape}")
        print()

    # find parameter-matched configurations
    target = 399_114
    print(f"\nSearching for configs matching vANN ({target:,} params)...\n")

    for n_synapses, name in [(16, 'dANN-R / dANN-LRF-4x4'), 
                              (32, 'dANN-LRF-4x8')]:
        n_somata, n_dps, actual = find_matching_config(target, n_synapses)
        print(f"{name}:")
        print(f"  n_somata={n_somata}, n_dendrites_per_soma={n_dps}")
        print(f"  Estimated params: {actual:,}  "
              f"(diff from target: {abs(actual-target):,})")
        print()