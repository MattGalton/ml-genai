This experiment implements a **Masked Autoencoder for Distribution Estimation (MADE)**, following the autoregressive models introduced in Stanford's [CS236: Deep Generative Models](https://www.youtube.com/watch?v=XZ0PMRWXBEU&list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8).

MADE models a high-dimensional probability distribution by decomposing it into a product of conditional distributions:

$$
p(x_1,\ldots,x_n)
=
\prod_{i=1}^{n}
p(x_i \mid x_1,\ldots,x_{i-1}).
$$

For binarised MNIST, I use a raster-scan ordering of the pixels, from the top-left pixel $x_1$ to the bottom-right pixel $x_{784}$.

## Model

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

## Architecture

The current implementation uses the following architecture:

$$
784 \rightarrow 1024 \rightarrow 1024 \rightarrow 784.
$$

The network therefore consists of:

- a masked linear layer from the 784 input pixels to 1024 hidden units;
- a ReLU activation;
- a second masked linear layer from 1024 to 1024 hidden units;
- a ReLU activation; and
- a final masked linear layer producing 784 output logits.

The final output contains one logit for each pixel.

The masks ensure that the output corresponding to $x_i$ can only depend on:

$$
x_1,\ldots,x_{i-1}.
$$

## Autoregressive Masking

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

## Mask Resampling

An interesting feature of this implementation is that the MADE masks are periodically regenerated during training.

The masks are initially generated using a fixed random seed. Every 25 training epochs, the mask degrees are regenerated using a new seed derived from the original seed and the update count.

The network weights are retained while the connectivity pattern changes.

This exposes the model to different valid autoregressive orderings of the hidden units during training while preserving the same input/output ordering.

The mask update frequency is:

$$
25\text{ epochs}.
$$

The initial random seed is fixed to make the experiment reproducible.

## Why Masks?

A naive feed-forward network could allow information from $x_i$ or later pixels to flow into the prediction for $x_i$.

That would violate the autoregressive factorisation.

The masks solve this by setting prohibited weights to zero:

$$
W_{\mathrm{masked}}
=
W \odot M,
$$

where $M$ is a binary mask and $\odot$ denotes element-wise multiplication.

The resulting network can therefore be evaluated in a single forward pass while still representing all of the conditional distributions required by the autoregressive model.

## MADE vs NADE

The NADE experiment also uses masked neural networks to model an autoregressive distribution, but the two architectures approach the problem differently.

NADE uses a particular masked architecture in which a hidden representation is shared across the conditional distributions.

MADE instead constructs a conventional multi-layer feed-forward network and uses masks to enforce the autoregressive dependency structure throughout the network.

In this implementation, MADE has two hidden layers:

$$
784 \rightarrow 1024 \rightarrow 1024 \rightarrow 784.
$$

This provides substantially more hidden capacity than the current NADE implementation.

Both models ultimately produce 784 logits corresponding to the conditional distributions of the 784 MNIST pixels.

## Training

The model is trained by maximising the log-likelihood of the training data, or equivalently minimising the negative log-likelihood:

$$
\mathcal{L}
=
-\sum_{i=1}^{n}
\log p(x_i\mid x_1,\ldots,x_{i-1}).
$$

For binary MNIST pixels, each conditional distribution is Bernoulli, so the loss can be expressed as binary cross-entropy over the predicted logits.

The experiment uses the Adam optimiser with a learning rate of:

$$
\eta = 0.001.
$$

Training is configured for up to 124 epochs.

## Dataset

The experiment uses the **binarised MNIST** dataset.

Each image contains $28\times28=784$ binary pixels. The pixels are flattened using a raster-scan ordering.

The model therefore learns an ordering-dependent autoregressive distribution over the image.

## Sampling

Because the model represents an autoregressive distribution, new images can be generated using ancestral sampling.

The pixels are sampled sequentially according to:

$$
x_i \sim p(x_i\mid x_1,\ldots,x_{i-1}).
$$

## Results

See the generated experiment page for:

- training and validation curves;
- validation loss;
- bits-per-dimension (BPD);
- validation perplexity; and
- generated samples at different training epochs.

The generated samples can be explored interactively by moving the epoch slider on the experiment page.