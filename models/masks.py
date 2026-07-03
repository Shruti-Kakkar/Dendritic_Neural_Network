import torch


def make_synaptic_mask(n_dendrites, n_inputs=1024, 
                        n_synapses=16, mode='random', 
                        image_size=32):
    """
    Creates the synaptic mask (input → dendrite).
    Shape: (n_dendrites, n_inputs)
    Each row has exactly n_synapses ones, rest zeros.
    
    modes:
        'random' : randomly pick n_synapses pixels per dendrite
        'lrf_4x4': pick a random center, take 4x4 neighborhood (16 pixels)
        'lrf_4x8': pick a random center, take 4x8 neighborhood (32 pixels)
    """
    mask = torch.zeros(n_dendrites, n_inputs)

    for d in range(n_dendrites):

        if mode == 'random':
            idx = torch.randperm(n_inputs)[:n_synapses]
            mask[d, idx] = 1.0

        elif mode == 'lrf_4x4':
            pixels = []
            # keep trying until we get exactly 16 pixels
            while len(pixels) < n_synapses:
                # pick random center pixel
                cx = torch.randint(0, image_size, (1,)).item()
                cy = torch.randint(0, image_size, (1,)).item()
                pixels = []
                for i in range(cx, min(cx + 4, image_size)):
                    for j in range(cy, min(cy + 4, image_size)):
                        pixels.append(i * image_size + j)
            mask[d, pixels[:n_synapses]] = 1.0

        elif mode == 'lrf_4x8':
            pixels = []
            while len(pixels) < n_synapses:
                cx = torch.randint(0, image_size, (1,)).item()
                cy = torch.randint(0, image_size, (1,)).item()
                pixels = []
                for i in range(cx, min(cx + 4, image_size)):
                    for j in range(cy, min(cy + 8, image_size)):
                        pixels.append(i * image_size + j)
            mask[d, pixels[:n_synapses]] = 1.0

    return mask


def make_cable_mask(n_somata, n_dendrites_per_soma):
    """
    Creates the cable mask (dendrite → soma).
    Shape: (n_somata, n_total_dendrites)
    Each soma connects only to its own dendrites.
    """
    n_dendrites = n_somata * n_dendrites_per_soma
    mask = torch.zeros(n_somata, n_dendrites)

    for s in range(n_somata):
        start = s * n_dendrites_per_soma
        end   = start + n_dendrites_per_soma
        mask[s, start:end] = 1.0

    return mask


def count_mask_params(synaptic_mask, cable_mask, n_somata, n_classes=10):
    """
    Count trainable parameters in a dANN with these masks.
    Useful for parameter matching with vANN.
    """
    n_dendrites = synaptic_mask.shape[0]

    # layer 1: synaptic weights + biases
    syn_params = int(synaptic_mask.sum().item()) + n_dendrites

    # layer 2: cable weights + biases
    cable_params = int(cable_mask.sum().item()) + n_somata

    # layer 3: fully connected output + biases
    out_params = n_somata * n_classes + n_classes

    return syn_params + cable_params + out_params


if __name__ == "__main__":
    # quick check
    print("Testing mask generation...")

    # synaptic masks
    mask_r    = make_synaptic_mask(256, mode='random',  n_synapses=16)
    mask_lrf4 = make_synaptic_mask(256, mode='lrf_4x4', n_synapses=16)
    mask_lrf8 = make_synaptic_mask(256, mode='lrf_4x8', n_synapses=32)

    print(f"Synaptic mask (random)  shape: {mask_r.shape}")
    print(f"Synaptic mask (lrf_4x4) shape: {mask_lrf4.shape}")
    print(f"Synaptic mask (lrf_4x8) shape: {mask_lrf8.shape}")

    # verify each dendrite gets exactly the right number of inputs
    print(f"Inputs per dendrite (random):  "
          f"{mask_r.sum(dim=1).unique().item()}")
    print(f"Inputs per dendrite (lrf_4x4): "
          f"{mask_lrf4.sum(dim=1).unique().item()}")
    print(f"Inputs per dendrite (lrf_4x8): "
          f"{mask_lrf8.sum(dim=1).unique().item()}")

    # cable mask
    cable = make_cable_mask(n_somata=32, n_dendrites_per_soma=8)
    print(f"\nCable mask shape: {cable.shape}")
    print(f"Each soma connects to exactly "
          f"{cable.sum(dim=1).unique().item()} dendrites")

    # parameter count
    n_params = count_mask_params(mask_r, cable, n_somata=32)
    print(f"\nEstimated dANN parameters: {n_params:,}")