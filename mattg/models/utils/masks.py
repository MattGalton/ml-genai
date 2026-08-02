from typing import List

import torch


def make_generator(seed: int | None = None) -> torch.Generator:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def make_nade_masks(n_visible, n_hidden):
    # hidden degrees in [1, n_visible - 1]
    degrees = torch.arange(1, n_hidden + 1) % (n_visible - 1)
    degrees = degrees + 1
    # input -> hidden: (H, D)
    m_in = (torch.arange(n_visible).unsqueeze(0) < degrees.unsqueeze(1)).float()
    # hidden -> output: (D, H)
    m_out = (degrees.unsqueeze(0) < torch.arange(1, n_visible + 1).unsqueeze(1)).float()
    return m_in, m_out


def make_made_masks(layer_sizes: List[int], generator: torch.Generator | None = None) -> List[torch.Tensor]:
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must contain at least input and output sizes")

    input_size = layer_sizes[0]
    output_size = layer_sizes[-1]
    hidden_sizes = layer_sizes[1:-1]

    input_degrees = torch.arange(1, input_size + 1)
    output_degrees = (torch.arange(output_size) % input_size) + 1

    degrees = [input_degrees]
    for hidden_size in hidden_sizes:
        hidden_degrees = torch.randint(
            low=1,
            high=input_size,
            size=(hidden_size,),
            generator=generator,
        )
        degrees.append(hidden_degrees)
    degrees.append(output_degrees)

    masks = []
    for idx, (in_degrees, out_degrees) in enumerate(zip(degrees[:-1], degrees[1:])):
        final_layer = idx == len(degrees) - 2
        if final_layer:
            # The final layer must enforce causality strictly
            mask = (out_degrees[:, None] > in_degrees[None, :]).float()
        else:
            # This is an input -> hidden or hidden -> hidden layer.
            # Enforce causality weakly
            mask = (out_degrees[:, None] >= in_degrees[None, :]).float()
        masks.append(mask)

    return masks


def make_made_conditional_masks(layer_sizes: List[int], num_classes: int, generator: torch.Generator | None = None):
    """Assumes the last num_classes inputs are a one-hot class label."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if layer_sizes[0] < num_classes:
        raise ValueError("input size must be at least num_classes")

    masks = make_made_masks(layer_sizes, generator)
    # Allow label inputs to reach every hidden unit in the first layer.
    # The Linear layer does Ax+b, where x has size i. This means A has size o x i.
    # Therefore, each row of the mask determines what input features can influence the feature in the output layer.
    # To make sure that ALL output features are able to be influenced by the class input,
    # we must set the final num_classes columns to 1.0 for the first mask.
    masks[0][:, -num_classes:] = 1.0
    return masks
