>These are the overall frameworks or recipes for generating new data. They define the *strategy* for learning the data distribution and generating samples
#### TL;DR
>**The normalizing constant $Z_θ$** is crucial for accurate probability calculations in generative models. For discrete data (e.g. classification, token prediction), $Z_θ$ is easily computed, the space of possible outcomes is finite and often manageable. This makes calculating $Z_{\theta}$ feasible, leading to accurate probability density functions (PDFs) and strong model performance. 
>**However**, with continuous data (e.g. image generation, audio waveforms, time-series), calculating $Z_θ$ is intractable, making it hard to obtain true PDFs and hindering performance. Score-based models solve this by learning the gradient of the log density (score function) instead of the density itself, bypassing $Z_θ$ and enabling impressive results in continuous data generation.

# How to create a generative model
[[PDF|What is PDF?]]

Suppose we are given a dataset $\{\mathbf{x}_1, \mathbf{x}_2, \cdots, \mathbf{x}_N\}$, where each point is drawn independently from an underlying data distribution $p(x).$ Given this dataset, the goal of generative modeling is to fit a model to the data distribution such that we can synthesize new data points at will by sampling from the distribution.

In order to build such a generative model, we first need a way to represent a probability distribution. One such way, as in likelihood-based models, is to directly model the [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) (p.d.f.) or [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function) (p.m.f.). Let $f_\theta(\mathbf{x}) \in \mathbb{R}$ be a real-valued function parameterized by a learnable parameter $θ$. We can define a p.d.f. via
$$
p_{\theta}(\mathbf{x})=\frac{e^{-f_{\theta}(\mathbf{x})}}{Z_{\theta}}
$$

where 
- $Z_{\theta} = \int e^{-f_{\theta}(\mathbf{x})} \mathrm{d} \mathbf{x}$ a normalizing constant to make area = 1  
- $e^{-f_{\theta}(\mathbf{x})}$ for non-negative value, $f_{\theta}(\mathbf{x})$ also called an unnormalized probabilistic model, or energy-based model
- $\theta$ is a learnable parameter

We can train $p_\theta(\mathbf{x})$ by maximizing the log-likelihood of the data
$$
\max _{\theta} \sum_{i=1}^{N} \log p_{\theta}\left(\mathbf{x}_{i}\right)
$$

but to compute $p_\theta(\mathbf{x})$ we need to find $Z_{\theta}$ which is hard for several key reasons:

1. **High-Dimensional Integration**: $Z_{\theta}$ requires computing the integral $\int e^{-f_{\theta}(\mathbf{x})} \mathrm{d} \mathbf{x}$ over the entire input space. For high-dimensional data (like images), this becomes computationally intractable due to the curse of dimensionality.
2. **Parameter Dependence**: The normalizing constant $Z_{\theta}$ depends on the model parameters $\theta$. This means it needs to be recomputed every time the parameters are updated during training, making the optimization process extremely expensive.
3. **Non-Analytical Form**: For complex $f_{\theta}(\mathbf{x})$ (like neural networks), the integral typically doesn't have a closed-form solution. This means numerical methods would be needed, which are often impractical for high-dimensional spaces.
4. **Global Computation**: Computing $Z_{\theta}$ requires integrating over the entire input space, not just the observed data points. This makes it particularly challenging for high-dimensional data where most of the space contains negligible probability mass.

# Generative models can be split into two main types

> **Explicit Density:** Tries to explicitly model $p(x)$ (e.g., VAEs, Flow models).
> 
> **Implicit Density:** Learns to generate samples without explicitly modeling $p(x)$ (e.g., GANs, Diffusion Models).

|           Category            |           Subcategory           | $Z_{\theta}$ |                        Example Models                        |
| :---------------------------: | :-----------------------------: | :----------: | :----------------------------------------------------------: |
| Explicit Density (Likelihood) |        Tractable Density        |  Computable  | Autoregressive (GPT, PixelCNN),  Normalizing Flows (RealNVP) |
|                               |       Approximate Density       | Approximated |               Variational Autoencoders (VAEs)                |
|  Implicit Density (Sampling)  | Generative Adversarial Networks |   Avoided    |                   GANs (StyleGAN, BigGAN)                    |
|                               |        Diffusion Models         |   Avoided    |       DDPMs, Score-based (Stable Diffusion, DALL-E 2)        |
>  **Note**: [[Transformer|Transformers]] are not generative model, but it's a **neural network architecture** like a tool to integrate within these generative models.

### 1. likelihood-based models (Explicit Density)

**Definition:** These models directly define a probability distribution $p_{\theta}(x)$ over the data space, parametrized by $\theta$. The goal is to find parameters $\theta$ that maximize the likelihood of the observed data.

**Normalizing Constant ($Z_{\theta}$):** The key challenge here is often the normalizing constant $Z_{\theta}$, which ensures that the distribution integrates to 1. 

#### Subcategories
1. **Tractable Density Models:** These models are designed such that $Z_{\theta}$ is either analytically computable or can be efficiently estimated. Examples include:
	-  **Autoregressive Models:** They decompose the joint probability of a data point into a product of conditional probabilities which each conditional probability are normalized via softmax(discrete data), making the likelihood tractable.  
		$$
		p(x) = \prod_{i=1}^{n} p(x_i | x_1, ..., x_{i-1})
		$$
		 
	- **Normalizing Flows:** These models learn a series of invertible transformations to map a simple base distribution (like a Gaussian) to the complex data distribution. The change of variables formula makes it possible to compute the likelihood exactly. Examples include: **RealNVP, Glow**
2. **Approximate Density Models:** $Z_{\theta}$ is intractable to compute directly. These models rely on approximations:
	-  **Variational Autoencoders (VAEs):** They use variational inference to approximate the posterior distribution and the marginal likelihood (evidence) by maximizing a lower bound called the Evidence Lower Bound (ELBO). The ELBO can be seen as a way to deal with the intractable $Z_{\theta}$.

### 2. Implicit Density Models

**Definition:** These models don't explicitly define a probability density function. Instead, they learn a process to generate samples from the target distribution, often without having access to $p_{\theta}(x)$ or $Z_{\theta}$.

**Normalizing Constant ($Z_{\theta}$):**  Implicit models generally avoid dealing with $Z_{\theta}$ altogether.

#### Subcategories
1. **Generative Adversarial Networks (GANs):** GANs train two networks – a generator and a discriminator – in an adversarial game. The generator learns to produce realistic samples to fool the discriminator, while the discriminator learns to distinguish between real and generated samples. The generator implicitly learns to sample from the data distribution. Examples include: **DCGAN, StyleGAN, BigGAN**
2. **Diffusion Models:** Inspired by non-equilibrium thermodynamics, diffusion models learn to reverse a gradual noising process. They start with the data distribution and progressively add noise until the data is completely destroyed. The model then learns to reverse this process, generating samples by gradually removing noise from a pure noise input.
	-  [[Score-based models|Score-based Generative Models]]: Learn the score function (gradient of the log-density) of the data distribution at different noise levels. Sampling is done via methods like Langevin dynamics. Examples include: **Stable Diffusion, Imagen, DALL-E 2, DDPMs**



