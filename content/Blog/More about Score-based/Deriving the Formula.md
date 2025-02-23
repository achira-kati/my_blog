# Translate to conditional probability

The [[Multivariate normal distribution]] is given by the formula
$$
\mathcal{N}(\mathbf{y};\mathbf{\mu},\mathbf{\Sigma})=\frac{1}{\sqrt{(2\pi)^d|\mathbf{\Sigma}|}}\exp{\left(-\frac{1}{2}(\mathbf{y}-\mathbf{\mu})^\top\mathbf{\Sigma}^{-1}(\mathbf{y}-\mathbf{\mu})\right)}
$$
where:

- $d$ is the dimension of the vector,
- $\boldsymbol{\mu}$ is the mean vector,
- $\boldsymbol{\Sigma}$ is the covariance matrix, and
- $|\boldsymbol{\Sigma}|$ denotes the determinant of $\boldsymbol{\Sigma}$.

In your case, the model is
$$
\tilde{x} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

This means that conditioned on $x$, the random variable $\tilde{x}$ is distributed as a multivariate normal with:
- Mean: $\boldsymbol{\mu} = x$
- Covariance: $\boldsymbol{\Sigma} = \sigma^2 I.$

Substitute these into the multivariate normal formula:
1. **Determinant and Inverse of the Covariance:**  
    Since $\boldsymbol{\Sigma} = \sigma^2 I$ (a diagonal matrix with each diagonal entry equal to $\sigma^2$):
    - The determinant is $|\sigma^2 I| = (\sigma^2)^d = \sigma^{2d}.$
    - The inverse is $(\sigma^2 I)^{-1} = \frac{1}{\sigma^2} I.$
2. **Plug into the Formula:** 
$$
p(\tilde{x} \mid x) = \frac{1}{\sqrt{(2\pi)^d \sigma^{2d}}} \exp\left(-\frac{1}{2} (\tilde{x} - x)^\top \left(\frac{1}{\sigma^2} I\right) (\tilde{x} - x)\right)
$$
3. **Simplify:**
	1. The identity matrix $I$ doesn't change the vector, so
	$$
	(\tilde{x} - x)^\top I (\tilde{x} - x) = (\tilde{x} - x)^\top (\tilde{x} - x) = \|\tilde{x} - x\|^2
	$$
	
	2. Thus, the exponent simplifies to 
	$$
	-\frac{1}{2\sigma^2}\|\tilde{x} - x\|^2
	$$

	3. Final expression 
	$$
	p(\tilde{x} \mid x) = \frac{1}{(2\pi \sigma^2)^{d/2}} \exp\left(-\frac{1}{2\sigma^2} \|\tilde{x} - x\|^2\right)
	$$

In summary, starting from $\tilde{x} = x + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, the conditional probability $p(\tilde{x} \mid x)$ is
$$
p(\tilde{x} \mid x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I) = \frac{1}{(2\pi \sigma^2)^{d/2}} \exp\left(-\frac{1}{2\sigma^2} \|\tilde{x} - x\|^2\right)
$$



> Why 
> $$
> ( \tilde { x } - x)^{T}(\tilde { x } - x) = \| \tilde { x }-x\|^{2}
> $$
> 
> Answer:
> The Euclidean distance between two vectors $x$ and $\tilde{x}$ is defined as the length of the vector difference $\tilde{x} - x.$ Concretely, if 
> $$
> \tilde{x} - x = \begin{bmatrix} \tilde{x}_1 - x_1 \\ \tilde{x}_2 - x_2 \\ \vdots \\ \tilde{x}_n - x_n \end{bmatrix}
> $$
> then the Euclidean distance is given by 
> $$
> \|\tilde{x} - x\| = \sqrt{(\tilde{x}_1 - x_1)^2 + (\tilde{x}_2 - x_2)^2 + \cdots + (\tilde{x}_n - x_n)^2}
> $$
> When you square this distance, you get 
> $$
> \|\tilde{x} - x\|^2 = (\tilde{x}_1 - x_1)^2 + (\tilde{x}_2 - x_2)^2 + \cdots + (\tilde{x}_n - x_n)^2
> $$
> This is exactly the same as computing the dot product of the vector $\tilde{x} - x$ with itself: 
> $$
> (\tilde{x} - x)^\top (\tilde{x} - x) = (\tilde{x}_1 - x_1)^2 + (\tilde{x}_2 - x_2)^2 + \cdots + (\tilde{x}_n - x_n)^2
> $$
> Thus, we have 
> $$
> (\tilde{x} - x)^\top (\tilde{x} - x) = \|\tilde{x} - x\|^2
> $$


# From SDE to conditional probability

The key idea is to solve the stochastic differential equation (SDE) by integrating its diffusion term and then computing the variance of the resulting stochastic integral. Here’s a step-by-step explanation:
### 1. Writing the SDE and Its Integral Form
You start with the SDE: 
$$
d \mathbf{x}(t) = \sigma^t \, d\mathbf{w}(t), \quad t\in[0,1]
$$
where $\mathbf{w}(t)$ is a standard Wiener process (Brownian motion).
Integrating both sides from 0 to $t$ gives: 
$$
\mathbf{x}(t) = \mathbf{x}(0) + \int_0^t \sigma^s \, d\mathbf{w}(s)
$$

### 2. Distribution of the Stochastic Integral
The integral
$$
\int_0^t \sigma^s \, d\mathbf{w}(s)
$$

is a stochastic integral with a deterministic integrand. By properties of such integrals, it is normally distributed with mean zero and covariance given by the Itô isometry:
$$
\text{Cov}\left[\int_0^t \sigma^s \, d\mathbf{w}(s)\right] = \int_0^t (\sigma^s)^2 \, ds \; \mathbf{I}
$$

where $\mathbf{I}$ is the identity matrix.
### 3. Evaluating the Variance Integral

We need to compute the integral
$$
\int_0^t \sigma^{2s} \, ds
$$

This can be computed by rewriting the integrand in exponential form:
$$
\sigma^{2s} = e^{2s\log \sigma}
$$

Thus,
$$
\int_0^t \sigma^{2s} \, ds = \int_0^t e^{2s\log \sigma} \, ds
$$

This is an elementary integral:
$$
\int_0^t e^{2s\log \sigma} \, ds = \left[\frac{e^{2s\log \sigma}}{2\log \sigma}\right]_0^t = \frac{e^{2t\log \sigma} - 1}{2\log \sigma} = \frac{\sigma^{2t} - 1}{2\log \sigma}
$$

### 4. Putting It All Together
Since
$$
\mathbf{x}(t) = \mathbf{x}(0) + \int_0^t \sigma^s \, d\mathbf{w}(s)
$$

and the stochastic integral is Gaussian with mean 0 and covariance $\frac{\sigma^{2t} - 1}{2\log \sigma} \mathbf{I},$ it follows that: 
$$
\mathbf{x}(t) \mid \mathbf{x}(0) \sim \mathcal{N}\Big(\mathbf{x}(0), \frac{\sigma^{2t} - 1}{2\log \sigma}\mathbf{I}\Big)
$$
Thus, the transition probability density is:
$$
p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) = \mathcal{N}\Big(\mathbf{x}(t); \mathbf{x}(0), \frac{1}{2\log \sigma}(\sigma^{2t} - 1) \mathbf{I}\Big)
$$

# Plug conditional probability into loss function

## 1. Gaussian Conditional Distribution and Its Score
The conditional distribution of $\mathbf{x}(t)$ given $\mathbf{x}(0)$ is a Gaussian:
$$
p(\mathbf{x}(t) \mid \mathbf{x}(0)) = \mathcal{N}\Big(\mathbf{x}(t);\, \mathbf{x}(0), \, \text{variance}\cdot\mathbf{I}\Big)
$$
where the variance is given by 
$$
\text{variance} = \frac{1}{2 \log \sigma}(\sigma^{2t} - 1)
$$
For a Gaussian distribution with mean $\mu$ and covariance $\sigma^2\mathbf{I},$ its probability density function is: 
$$
p(\mathbf{x}) = \frac{1}{(2\pi\sigma^2)^{d/2}} \exp\Bigg(-\frac{\|\mathbf{x}-\mu\|^2}{2\sigma^2}\Bigg)
$$
Taking the logarithm, we get:
$$
\log p(\mathbf{x}) = -\frac{d}{2}\log(2\pi\sigma^2) - \frac{\|\mathbf{x}-\mu\|^2}{2\sigma^2}
$$
The score function is defined as the gradient of the log-density with respect to $\mathbf{x}$:
$$
\nabla_{\mathbf{x}} \log p(\mathbf{x}) = \nabla_{\mathbf{x}} \Bigg(-\frac{\|\mathbf{x}-\mu\|^2}{2\sigma^2}\Bigg) = -\frac{1}{\sigma^2} (\mathbf{x} - \mu)
$$

In our setting:
- The "mean" $\mu$ is $\mathbf{x}(0).$
- The variance is $\frac{1}{2 \log \sigma}(\sigma^{2t} - 1).$

Thus, the score (the gradient of the log-density) becomes:
$$
\nabla_{\mathbf{x}(t)} \log p(\mathbf{x}(t) \mid \mathbf{x}(0)) = -\frac{\mathbf{x}(t)-\mathbf{x}(0)}{\frac{1}{2 \log \sigma}(\sigma^{2t}-1)}
$$
in other words:
$$
\nabla_{\mathbf{x}(t)} \log p(\mathbf{x}(t) \mid \mathbf{x}(0)) =-\frac{\mathbf{x}(t)-\mathbf{x}(0)}{\text{variance}}
$$


## 2. Plugging into the Loss Function
The loss function is defined as:
$$
\mathcal{L}(\theta) = \min_\theta \mathbb{E}_{t\sim \mathcal{U}(0, T)} [\lambda(t) \mathbb{E}_{\mathbf{x}(0) \sim p_0(\mathbf{x})}\mathbf{E}_{\mathbf{x}(t) \sim p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))}[ \|s_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))\|_2^2]]
$$

Substituting the score: 
$$
\|s_\theta(\mathbf{x}(t), t) - (-\frac{\mathbf{x}(t)-\mathbf{x}(0)}{\text{variance}})\|_2^2
$$

## 3. Complete loss function

Consider the loss
$$
\|s_\theta(\mathbf{x}(t), t) - (-\frac{\mathbf{x}(t)-\mathbf{x}(0)}{\text{variance}})\|_2^2
$$
we know that magnitude (scale) of true score varies according to time $(t)$ since variance increases with time $(t)$.
$$
\text{variance} = \frac{1}{2 \log \sigma}(\sigma^{2t} - 1)
$$
This means the model must predict the correct scale for each time $t$ to balance the loss across all time steps. However, the model will likely struggle with this, so we need to assist it in finding the correct scale for each $t$. To do that:
1. The **"typical"** difference $\mathbf{x}(t)-\mathbf{x}(0)$ is roughly $\sqrt{\text{variance}}$ since 
	$$
	\mathbf{x}(t)-\mathbf{x}(0) \sim \mathcal{N}\Big(0,\, \text{variance}\cdot\mathbf{I}\Big)
	$$
	that means typical size (or scale) is $\sqrt{\text{variance}}$. In other words, although individual samples vary, most values of $\mathbf{x}(t)-\mathbf{x}(0)$ are on the order of $\sqrt{\text{variance}}$. "On the order of" is a shorthand for saying "approximately proportional to" or "roughly of the same scale as."

2. The **"typical"** magnitude of true score: Since the score is
	$$
	-\frac{\mathbf{x}(t)-\mathbf{x}(0)}{\text{variance}}
	$$
	its magnitude is roughly 
	$$
	\frac{\|\mathbf{x}(t)-\mathbf{x}(0)\|}{\text{variance}}
	$$
	Given that $\|\mathbf{x}(t)-\mathbf{x}(0)\|$ is typically on the order of $\sqrt{\text{variance}}$, the typical magnitude of the true score becomes approximately 
	$$
	\frac{\sqrt{\text{variance}}}{\text{variance}} = \frac{1}{\sqrt{\text{variance}}}.
	$$

	that means on average at each time $t$, it will have magnitude (scale) equal to $\frac{1}{\sqrt{ \text{variance} }}$

3. We simply divide the output of the model by $\sqrt{\text{variance}}$ then we will have match score.

	Lastly, we choose the **weighting function** 
	$$
	\lambda(t) = \text{variance}=\sigma^2
	$$
	to avoid division (reason [[Avoided computation|here]]), so our loss now: 
	$$
	\sigma(t)^2 \left\| s_\theta(\mathbf{x}(t), t) + \frac{\mathbf{x}(t)-\mathbf{x}(0)}{\sigma(t)^2} \right\|_{2}^{2}
	$$
	therefore 
	$$
	\left\| \sigma(t)^2s_\theta(\mathbf{x}(t), t) + \text{noise} \right\|_{2}^{2} \quad \text{noise}=\mathbf{x}(t)-\mathbf{x}(0)
	$$

# How to compute step size for PC
The step size is chosen adaptively to balance the contribution of the score (signal) and the injected noise in the Langevin MCMC update. Recall that the Langevin update is: 
$$
\mathbf{x} \leftarrow \mathbf{x} + \epsilon\, s_\theta(\mathbf{x}, t) + \sqrt{2\epsilon}\, \mathbf{z}
$$
where:

- $s_\theta(\mathbf{x}, t)$ is the score (i.e., an estimate of $\nabla_{\mathbf{x}} \log p_t(\mathbf{x}))$
- $\epsilon$ is the step size, and
- $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is standard Gaussian noise.

The goal is to set $\epsilon$ such that the update from the score is proportional to the noise level, scaled by a desired signal-to-noise ratio (SNR). That is, we want the magnitude of the score update to be $\text{snr}$ times the magnitude of the noise update.

1. **Magnitude of the Score Update:**  
    The change due to the score is approximately 
    $$
    \Delta x_{\text{score}} = \epsilon \, \| s_\theta(\mathbf{x}, t) \|
    $$
    where $\| s_\theta(\mathbf{x}, t) \|$ is the norm of the score.
2. **Magnitude of the Noise Update:**  
    The noise term has a typical magnitude of 
    $$
    \Delta x_{\text{noise}} = \sqrt{2\epsilon}\, \text{noise\_norm}
    $$
    where $\text{noise\_norm}$ is an estimate of the norm of a standard Gaussian noise vector (compute from $\sqrt{\prod \text{dimensions}}$ of $\mathbf{x}$ this is the typical norm of a standard Gaussian vector in that space)
3. **Balancing the Two Terms:**  
    To enforce a desired signal-to-noise ratio ($\text{snr}$), we set: 
    $$
    \epsilon \, \| s_\theta(\mathbf{x}, t) \| = \text{snr} \times \sqrt{2\epsilon}\, \text{noise\_norm}
    $$
    
4. **Solving for $\epsilon$:**  
    Rearranging, we have: 
    $$
    \| s_\theta(\mathbf{x}, t) \| = \text{snr} \sqrt{2\epsilon}\, \text{noise\_norm} \quad \Longrightarrow \quad \sqrt{\epsilon} = \frac{\text{snr} \, \sqrt{2}\, \text{noise\_norm}}{\| s_\theta(\mathbf{x}, t) \|}
    $$
    Squaring both sides gives: 
    $$
    \epsilon = 2 \left( \frac{\text{snr} \, \text{noise\_norm}}{\| s_\theta(\mathbf{x}, t) \|} \right)^2
    $$
    
So the step size is adaptively set to ensure that the update from the score is $\text{snr}$ times the size of the noise update, balancing the two contributions in the Langevin MCMC step. This adaptive choice helps maintain stability and improves the quality of the refined sample during the corrector step.


