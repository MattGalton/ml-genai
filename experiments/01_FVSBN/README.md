This experiment implements a **Fully Visible Sigmoid Belief Network (FVSBN)**,
following the autoregressive models introduced in Stanford's
[CS236: Deep Generative Models](https://www.youtube.com/watch?v=XZ0PMRWXBEU&list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8).

## Autoregressive Models

These models factorise the joint distribution using the chain rule:

$$
p(x_1,\ldots,x_n)
=
\prod_{i=1}^{n}
p(x_i \mid x_1,\ldots,x_{i-1}).
$$

Note that the conditional distribution for each pixel $x_i$ depends **only on the pixels preceding it**.
This is an important feature that must be preserved, called causality.

More generally, each variable can be conditioned on any subset of variables 
that precede it in a valid ordering, 
provided the resulting dependency graph is a directed acyclic graph (DAG). This can be seen as
an approximation to the full chain rule disintegration of the joint density.

## FVSBN Model

The first pixel is modelled using a Bernoulli distribution:

$$
p(x_1 = 1) = \alpha_1,
\qquad
p(x_1 = 0) = 1-\alpha_1.
$$

For every subsequent pixel, the conditional probability is parameterised
using a sigmoid:

$$
p(x_i = 1 \mid x_1,\ldots,x_{i-1}) = \sigma\left(\alpha_0^{(i)} + \sum_{j=1}^{i-1}\alpha_j^{(i)}x_j \right),
$$

where

$$
\sigma(z) = \frac{1}{1+e^{-z}}.
$$

In other words, every pixel is predicted from the pixels that precede it
in the raster-scan ordering.

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

Unlike evaluating the likelihood, sampling cannot be performed for all
pixels independently in a single step: each pixel depends on the samples
generated before it.

## Why FVSBN?

FVSBN provides a useful baseline for understanding autoregressive density
estimation.

Its main advantage is simplicity: there is very little machinery between
the mathematical factorisation and the implementation.

The downside is that the model has a large number of parameters and does
not learn a hierarchical hidden representation of the data. Each
conditional is essentially a logistic regression over all preceding
pixels.

This makes FVSBN a useful starting point before moving to more expressive
architectures such as **NADE** and **MADE**.