This experiment implements 
**Neural Autoregressive Density Estimation (NADE)**, 
following the autoregressive models introduced in Stanford's [CS236: Deep Generative Models](https://www.youtube.com/watch?v=XZ0PMRWXBEU&list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8).

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

## NADE Model

Unlike the Fully Visible Sigmoid Belief Network (FVSBN), where each conditional distribution is parameterised directly using the preceding variables, NADE introduces a hidden representation.

For each variable (pixel!) $x_i$, the conditional distribution is parameterised by a neural network whose inputs are restricted to the preceding variables:

$$
p(x_i = 1 \mid x_1,\ldots,x_{i-1})
=
\sigma(a_i),
$$

where the output logit $a_i$ is produced by the masked neural network.

The model consists of a masked linear layer followed by a ReLU activation and a second masked linear layer:

$$
x
\rightarrow
\operatorname{MaskedLinear}
\rightarrow
\operatorname{ReLU}
\rightarrow
\operatorname{MaskedLinear}
\rightarrow
a.
$$

The sigmoid converts each output logit into a Bernoulli probability.

The important constraint is that the prediction for $x_i$ must not depend on $x_i$ itself, or on any subsequent variables.
This autoregressive property is enforced using masks on the linear layers.

### Architecture

The current implementation uses a single hidden layer,so the network consists of

- a masked linear layer from the visible variables to the hidden units;
- a ReLU non-linearity; and
- a second masked linear layer from the hidden units to the output variables.

The output contains one logit for each variable.

The masks ensure that the output corresponding to $x_i$ can only depend on:

$$
x_1,\ldots,x_{i-1}.
$$

### Masking

The key idea in NADE is that the autoregressive dependency structure is encoded directly into the network weights.

The first masked linear layer prevents a hidden unit from seeing pixels that occur later in the raster-scan ordering.

The second masked linear layer prevents the output for $x_i$ from depending on $x_i$ or any subsequent pixel.

This means that although the model is implemented using ordinary linear layers, the effective computation respects the autoregressive factorisation.

The masks are constructed by assigning each hidden unit a degree that determines which input variables it is allowed to depend on.

#### Degrees

Each hidden unit is assigned a degree indicating the latest input variable that it is allowed to depend on.

For a hidden unit with degree $d$, the input mask allows connections from variables satisfying:

$$
j < d.
$$

The output mask then ensures that the logit for $x_i$ can only receive information from hidden units whose degree satisfies:

$$
d < i.
$$

Together these constraints enforce the required autoregressive ordering.

### NADE vs FVSBN

The FVSBN experiment directly parameterises each conditional distribution using the preceding pixels:

$$
p(x_i=1\mid x_{<i})
=
\sigma
\left(
\alpha_0^{(i)}
+
\sum_{j<i}\alpha_j^{(i)}x_j
\right).
$$

NADE instead introduces a learned hidden representation:

$$
x_{<i}
\rightarrow
h
\rightarrow
p(x_i\mid x_{<i}).
$$

This gives the model a non-linear representation while preserving the same autoregressive dependency structure.

In both cases, the raster ordering determines which variables are allowed to influence each prediction.

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
