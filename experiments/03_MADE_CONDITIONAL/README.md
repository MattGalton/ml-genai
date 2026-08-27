This experiment implements 
**Masked Autoencoder for Density Estimation (MADE)**, 
following the autoregressive models introduced in Stanford's [CS236: Deep Generative Models](https://www.youtube.com/watch?v=XZ0PMRWXBEU&list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8).

## Autoregressive Models

These models factorise the joint distribution using the chain rule:

$$
p(x_1,\ldots,x_n) = \prod_{i=1}^{n} p(x_i \mid x_1,\ldots,x_{i-1}).
$$

Note that the conditional distribution for each variable $x_i$ depends **only on the variables preceding it**.
This is an important feature that must be preserved, called causality.

More generally, each variable can be conditioned on any subset of variables 
that precede it in a valid ordering, 
provided the resulting dependency graph is a directed acyclic graph (DAG). This can be seen as
an approximation to the full chain rule disintegration of the joint density.

## MADE Model

MADE uses a feed-forward neural network while enforcing the autoregressive dependency structure through carefully constructed masks.

The conditional distribution for each pixel is:

$$
p(x_i = 1 \mid x_1,\ldots,x_{i-1})
=
\sigma(a_i),
$$

where $a_i$ is the output logit produced by the masked neural network.

The key idea is that the network is allowed to be fully connected internally, but the masks restrict which variables can influence each output.

This allows a standard neural network to represent the entire set of autoregressive conditional distributions in a single forward pass.

### Autoregressive Masking

The masks are constructed using a **degree** assigned to every input, hidden, and output unit.

The input pixels are assigned degrees according to their position in the autoregressive ordering:

$$
m(x_i) = i.
$$

Hidden units are assigned degrees sampled from:

$$
\{1,\ldots,n-1\}.
$$

The masks between layers are then constructed from these degrees.

For hidden layers, connections are allowed when the degree of the receiving unit is greater than or equal to the degree of the sending unit:

$$
m_{\mathrm{out}} \geq m_{\mathrm{in}}.
$$

For the final layer, the constraint is made strict:

$$
m_{\mathrm{out}} > m_{\mathrm{in}}.
$$

This strict inequality ensures that the prediction for $x_i$ cannot depend on $x_i$ itself.

Together, these constraints enforce the autoregressive factorisation:

$$
p(x_1,\ldots,x_n)
=
p(x_1)
p(x_2\mid x_1)
\cdots
p(x_n\mid x_1,\ldots,x_{n-1}).
$$


### Mask Resampling

An interesting feature of this implementation is that the MADE masks are periodically regenerated during training.

The masks are initially generated using a fixed random seed. Every N training epochs, the mask degrees are regenerated using a new seed derived from the original seed and the update count.

The network weights are retained while the connectivity pattern changes.

This exposes the model to different valid autoregressive orderings of the hidden units during training while preserving the same input/output ordering.

## Dataset

We use the binarised MNIST dataset.

Each image is $28\times28$ pixels and is flattened into a vector of
784 binary variables using raster-scan ordering, yielding

$$
x_1,x_2,\ldots,x_{784},
$$

starting at the top-left corner of the image and proceeding across each
row before moving to the next row.

## Loss Function

The model is trained by maximising the likelihood. For binary images, this
is equivalent to binary cross-entropy summed over the pixels.


## Sampling

The autoregressive structure also gives a straightforward way to generate
new images.

Pixels are sampled sequentially from the learned conditional
distributions:

$$
x_1 \sim p(x_1),
$$

then

$$
x_2 \sim p(x_2\mid x_1),
$$

and so on until

$$
x_{784}
\sim
p(x_{784}\mid x_1,\ldots,x_{783}).
$$

Each newly sampled pixel becomes part of the conditioning information
used to generate the next pixel.

## MADE vs NADE

The NADE experiment also uses masked neural networks to model an autoregressive distribution, but the two architectures approach the problem differently.

NADE uses a particular masked architecture in which a hidden representation is shared across the conditional distributions.

MADE instead constructs a conventional multi-layer feed-forward network and uses masks to enforce the autoregressive dependency structure throughout the network.