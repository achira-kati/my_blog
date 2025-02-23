This post builds on the ideas presented in the [video](https://www.youtube.com/watch?v=B4oHJpEJBAA) and the accompanying [paper blog](https://yang-song.net/blog/2021/score/).

> **TL;DR;**
> A score-based model is built to estimate $p(x)$ by using the score function. For any input point in the data space, the model predicts its score, which indicates the direction to higher probability (understanding the score function's meaning is key). Inference uses Langevin dynamics sampling, iteratively applying the model (ideally many times) to move the input towards higher probability areas, resulting in a final sample from the highest probability area.

Recall that we still have [[Generative Models#How to create a generative model|problem]] on $Z_{\theta}$.

# Score function 
Instead of modeling the density function directly, we model the score function. The **score function** of a distribution $p(x)$ is defined as 
$$
\nabla _ { \mathbf { x } } \log { p ( \mathbf { x } ) }
$$
Our model aims to estimate this quantity by approximating 
$$
\mathbf{s}_\theta(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log p(\mathbf{x})
$$
an approach known as a **score-based model**. After that we [do some math](https://youtu.be/B4oHJpEJBAA?t=193&si=7OT6wmWI87YZ36tp) to get
$$
\begin{array} { r } { \mathbf { s } _ { \theta } ( \mathbf { x } ) = \nabla _ { \mathbf { x } } \log p _ { \theta } ( \mathbf { x } ) = - \nabla _ { \mathbf { x } } f _ { \theta } ( \mathbf { x } ) - \underbrace { \nabla _ { \mathbf { x } } \log Z _ { \theta } } _ { = 0 } = - \nabla _ { \mathbf { x } } f _ { \theta } ( \mathbf { x } ) . } \end{array}
$$
Note that $\mathbf{s}_\theta(\mathbf{x})$ is independent of the normalizing constant $Z_\theta$, which is highly advantageous.
# Score matching
The score function, $\nabla _ { \mathbf { x } } \log { p ( \mathbf { x } ) } ,$ , clearly indicates the direction in which the data probability increases, in other words, score function can tell you for any point in your data space in which direction you have to move to get closer to actual data points (probability increase) 

![[langevin.gif|center]]

In case of images we can start with random Gaussian image and score will tell you which direction we should move to get closer to image manifold (target image).

To do that we just train score-base model to minimize model and the data distribution:
$$
\frac{1}{2}\mathbb { E } _ { p ( \mathbf { x } ) } [ | | \nabla _ { \mathbf { x } } \log p ( \mathbf { x } ) - \mathbf { s } _ { \boldsymbol { \theta } } ( \mathbf { x } ) | | _ { 2 } ^ { 2 } ]
$$
**Note:** we use [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance#Higher_dimensions) to find distance between two vectors but formula is the same as [L2 Norm/Euclidean norm](<https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm>) so we can write $||p-d||_{2}$ and we commonly square this distance for (1) Ensure positive values. (2) Penalizes larger differences more. So final formula is $||p-d||_{2}^{2}$ then we add expected value (average) $\mathbb { E } _ { p ( \mathbf { x } ) }$ to get average across all points with respect to $p(x)$

However, we do not know $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ (the data score) or $p(x)$ (the data probability density function). Fortunately, by [performing some mathematical derivations](https://youtu.be/B4oHJpEJBAA?si=tYTsDtWpm4155Evn&t=343), we can eliminate $p(x)$ and arrive at the following expression: 
$$
\frac{1}{2}\mathbb{E}_{p( \mathbf { x } )} \left[  s_{\theta}\left(  x\right)
^{2}\right]  +\mathbb{E}_{p(x)} \left[  \nabla_{x}s_{\theta
}\left(  x\right)  \right]
$$
we can train minimizing this formula, so both term should be zero:
1.  $s_{\theta}\left(  x\right)^{2}=0$ means score are 0 or we at a data point.
2. $\nabla_{x}s_{\theta}\left(  x\right)=0$  gradient of score at the data point is 0 this means it should be a local maximum.

> **Note:** $\mathbb{E}_{p(x)}$ is still requiring $p(x)$ in formula, but in practice expectation is average so we can train approximate using sample from $p(x)$ (our training data).

So now our **goal** is to learn a model that given some point in our data space, predict the direction where we should move to get closer to data, this is much easier than learn to predict PDF function, since score function is don't need the normalizing constant $Z_{\theta}$. 

![[Pasted image 20241209145614.png|center|300x300]]


But we ran in to new problems:
1. **Expensive Training** ($\nabla_{x}s_{\theta}\left(  x\right)$);  When input space is large, you need to compute gradient for each input dimension to do that we need to do backpropagation for all input variable.
2. **Low Coverage of Data Space** ($s_{\theta}\left(  x\right)^{2}$); Our model is not trained on input from the entire space (of course, we train on some data) so when we have input that come from random position outside data space that we trained on, model will be inaccurate in other word model don't know correct direction, you can see that estimated scores are only accurate in high density regions (score point towards center). 

> *Note:* Why $\nabla_{x}s_{\theta}\left(  x\right)$ is computable (but expensive) while $Z_{\theta} = \int e^{-f_{\theta}(\mathbf{x})} \mathrm{d} \mathbf{x}$ is not computable ?
> Answer: Integration is total area under the curve so we must compute integral over the entire input of $e^{-f_{\theta}(x)}$ which have high dimensional in other hand differential only take specific point $x$. 

![[Pasted image 20241207235049.png|center]]


# Noise Perturbation
To overcome **Low Coverage of Data Space** problem; We just add some Gaussian noise to data point 
$$
\tilde { x } = x + \epsilon
$$
where
$$
\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )
$$

Now we denote our pdf as (new noise-perturbed pfd is depending on $\sigma$ which is noise level)
$$
p ( x ) \xrightarrow [ ] { \mathrm { n o i s e } } \, p _ { \sigma } ( \tilde { x } )
$$

and we train **Noise Perturbed Objective**
$$
\frac { 1 } { 2 } \mathbb { E } _ { p _ { \sigma } ( \tilde { x } ) } [ \| \nabla _ { \tilde { x } } \log p _ { \sigma } ( \tilde { x } ) - s _ { \theta } ( \tilde { x } ) \| _ { 2 } ^ { 2 } ]
$$

here is what happens when we perturb a mixture of two Gaussian perturbed by additional Gaussian noise.

![[Pasted image 20241209144033.png]]

This improve the accuracy of model on low density region, but we still got tradeoff here

![[Screenshot 2567-12-09 at 22.55.48.png]]

If we use high $\sigma$ (more noise):
- Cover more low density regions for better score estimation (resolve **Low Coverage of Data Space**). 
- But it over-corrupts the data and alters it significantly from the original distribution

If we use low $\sigma$ (less noise):
- Facing **Low Coverage of Data Space** problem.
- Not over-corrupts the original data.
# Connection to Denoising Autoencoders
someone found the connection between them and can overcome **Expensive Training** problem by [reformulate](https://youtu.be/B4oHJpEJBAA?si=6937TjMkcWprCBVh&t=822) score-matching objective from
$$
\frac { 1 } { 2 } \mathbb { E } _ { p _ { \sigma } ( \tilde { x } ) } [ \| \nabla _ { \tilde { x } } \log p _ { \sigma } ( \tilde { x } ) - s _ { \theta } ( \tilde { x } ) \| _ { 2 } ^ { 2 } ]
$$

to this:
$$
\frac { 1 } { 2 } \mathbb { E } _ { \boldsymbol { x } \sim p ( \boldsymbol { x } ) \, , \tilde { \boldsymbol { x } } \sim p _ { \sigma } ( \tilde { \boldsymbol { x } } | \boldsymbol { x } ) } [ | | s _ { \theta } ( \tilde { \boldsymbol { x } } ) \! - \! \nabla _ { \tilde { \boldsymbol { x } } } \log p _ { \sigma } ( \tilde { \boldsymbol { x } } | \boldsymbol { x } ) | | _ { 2 } ^ { 2 } ]
$$


This is huge, since normally in score-matching objective we don't know $p _ { \sigma } ( \tilde { x } )$ so we can't compute $\nabla _ { \tilde { x } } \log p _ { \sigma } ( \tilde { x } )$, we do some math as describe above, and we got **Expensive Training** problem, but with this new objective we just need to know $p _ { \sigma } ( \tilde { \boldsymbol { x } } | \boldsymbol { x } )$ and we know that in denoising autoencoders!

![[Screenshot 2567-12-09 at 20.36.23.png]]

We just need to find ${ p } _ { \sigma } ( \tilde { { x } } | { x } )$ by [[Deriving the Formula#Translate to conditional probability|using Multivariate Gaussian distribution]] then [compute](https://youtu.be/B4oHJpEJBAA?si=Op7T9Kj9EiUBVPkh&t=1116) $\nabla _ { \tilde { \boldsymbol { x } } } \log p _ { \sigma } ( \tilde { \boldsymbol { x } } | \boldsymbol { x })$ and our new objective is to minimize this:  
$$
\frac { 1 } { 2 } \mathbb { E } _ { { x } \sim { p } ( { x } ) } \, , \tilde { { x } } { \sim } { p } _ { \sigma } ( \tilde { { x } } | { x } ) \big [ \big | \big | s _ { \theta } \big ( \tilde { { x } } \big ) + \frac { 1 } { { \sigma } ^ { 2 } } \epsilon \big | \big | _ { 2 } ^ { 2 } \big ]
$$
This is [beautiful](https://youtu.be/B4oHJpEJBAA?si=8aaUd8GEpXJarlXA&t=1227),  $\frac { 1 } { { \sigma } ^ { 2 } } \epsilon$  is noise that we add to input and model need to predict $s _ { \theta } \big ( \tilde { { x } })=-\frac { 1 } { { \sigma } ^ { 2 } } \epsilon$ in other word model need to predict direction(score) that back to original data point.

![[Diffusion_Models_From_Scratch_Score_Based_Generative_Models_Explained_Math_Explained.gif|center]]

# How to generate new sample ?
After training our score-based model we can do get denoise sample by iteratively predict new direction using **Langevin Dynamics Sampling**: 
$$
\begin{array} { c } { { \tilde { x } _ { i + 1 } \leftarrow \tilde { x } _ { i } + \alpha \cdot s _ { \theta } ( \tilde { x _ { i } } ) + \sqrt { 2 \alpha } \cdot \epsilon } } \\ { { i = 0 , 1 , \cdot \cdot \cdot , K } } \end{array}
$$
At each step, we update the current sample $\tilde{x}_i$ by adding the product of a small step size $\alpha$ and the estimated score $s_\theta(\tilde{x}_i)$. We also add the term $\sqrt{2\alpha}\,\epsilon$ to introduce noise, which prevents all samples from collapsing to a single point. So our sample will become closer to data manifold.

![[Screenshot 2567-12-09 at 22.24.58.png|center|300x300]]


# Multiple Noise Perturbation
Recall that in **Low Coverage of Data Space** problem we have tradeoff when select $\sigma$, to achieve the best of both worlds, we use multiple scales of noise perturbations simultaneously.

We start from pre-specify $\sigma$ between $[0.01, 25]$ which have total length $L$ with increasing standard deviations $\sigma_{1} < \sigma_{2} < \dots < \sigma_{L}$ now got:
$$
\tilde { x } = x + \epsilon
$$
where 
$$
\begin{array} {c}
\epsilon \sim \mathcal { N } ( 0 , \sigma _ {i} ^ { 2 } I) \\
i=1,2,\dots,L
\end{array}
$$
now $\sigma_{i}$ is difference noise, and we add those noise as model input to give more information to model
$$
s _ { \theta } ( \tilde { x }, i )
$$
and jointly train model for all noises
$$
\sum _ { i = 1 } ^ { L } \lambda ( i ) \mathbb { E } _ { p _ { \sigma _ { i } } ( \mathbf { x } ) } [ \| \nabla _ { \mathbf { x } } \log p _ { \sigma _ { i } } ( \mathbf { x } ) - \mathbf { s } _ { \theta } ( \mathbf { x } , i ) \| _ { 2 } ^ { 2 } ]
$$
After model is trained, we can use same sample technique except we start with the largest noise $i=L,L-1,\dots,1$ since we know that large noise help model in low density region, This method is called **annealed Langevin dynamics**, also model that trained on difference noised scales is called **Noise Conditional Score Networks (NCSNs)**.

![[Screenshot 2567-12-09 at 23.28.35.png|center|400x400]]

In first paper they use 1000 noise scales.


# Link to Stochastic Process
[[SDEs and ODEs|What is SDEs]]
#### How score-matching link to SDEs
In [[Score-based models#Multiple Noise Perturbation|Multiple Noise Perturbation]] we use 1000 scales but If we want to cover as much noise scale as possible like 0.001 very small scale to very large scale like $\infty$ we no longer can write as discrete scale anymore (i.e., 1, 2, ..., 1000 scales) it will be much better If we turn into function of time $X(t)$ where $t$ is time that control noise added to image (like control scale), so $X(t)$ is also known as **Stochastic Process**, how?
1. We start at our added noise data: 
$$
\tilde { x } = x + \epsilon \quad \epsilon \sim \mathcal { N } ( 0 , \sigma _ {i} ^ { 2 } I)
$$
2. Compare to SDEs equation 
$$
d X _ { t } = \underbrace {f ( t , X _ { t } ) d t}_{deterministic\space(Drift)} + \underbrace {\sigma ( t , X _ { t } ) d W _ { t }}_{stochastic\space(Diffusion)}
$$
3. You can see that change in $x$ is just by adding noise $(\epsilon)$ in other word change in $x$ only influence by stochastic term. We can write SDEs as: 
$$
dx=\sigma(t,X_{t})dW_{t}
$$
4. We know that stochastic term (noise that we add) is only change with time $t$ so we can remove input $X_{t}$. Now we can write SDEs as: 
$$
dx=\sigma(t)dW_{t}
$$
5. If we got forward SDE, we can get reverse SDE: 
$$
\mathrm { d } x = [ f ( x , t ) - g ^ { 2 } ( t ) \underbrace{\nabla _ { x } \log p _ { \sigma } ( x )}_{score function} ] \ \mathrm { d } t + g ( t ) \ \mathrm { d } W_{t}
$$
you can see that it use score function to reverse

**Note** that in many diffusion model will have technique to add noise differently, so detail in above step is not exactly the same, but idea and step is the same.
![[Pasted image 20241210215252.png|center]] ^f3e2ff

This method offers several advantages over using multiple discrete noise levels:
1. Continuous noise handling
2. Improved generalization
3. More accurate score estimation
4. Better sampling through continuous annealing
5. Flexibility in designing the noise schedule
6. Closer connection to the underlying data distribution

# Summary 
## Score-function
The score function, by definition, gives you the gradient of the log probability density function at a given point. In other words, it tells you the direction in which the data density increases most steeply. Using just the score function, you can:
- **Guide Sampling:**  
    By following the gradient directions (e.g., via Langevin dynamics), you can iteratively refine a random noise input into a sample that resembles data drawn from the target distribution.
- **Denoising:**  
    In tasks like denoising, the score function indicates how to adjust a noisy input to move it toward higher probability regions, effectively “cleaning” the data.
- **Solve Inverse Problems:**  
    The gradient information can be used to guide optimization processes to recover or reconstruct signals from corrupted or incomplete data.

They can be applied to both audio and images

## Example Image Generation
[Code](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing) for this example.

If we want to generate image (new sample), we all know that score-function alone only give direction at given point, so we need to do sampling to make new sample before we link to SDE we use [[Score-based models#How to generate new sample ?|Langevin Dynamics Sampling]] but after we link to SDE we can use reverse SDE instead.

![[Pasted image 20241210215252.png|center]]

Steps:
1. Define how we perturb our data (add noise to sample), in this example we use 
$$
\tilde { x } = x + \epsilon \quad \epsilon \sim \mathcal { N } ( 0 , \sigma^ { 2 } I)
$$
2. Define forward SDE from step (1) using forward SDE formula. In this case we got 
$$
d \mathbf{x}(t) = \sigma^t \, d\mathbf{w}(t), \quad t\in[0,1]
$$

3.  [[Deriving the Formula#From SDE to conditional probability|Solve]] forward SDE to get conditional probability, which will be plug into loss function later.
4. [[Deriving the Formula#Plug conditional probability into loss function|Plug conditional probability]] from step (2) into [[Score-based models#New loss function|loss function]].
5. Train [[Score-based models#Build Time-Dependent Score-Based Model|score-based model]] using loss from (4).
6. After trained model, we create new sample (image) using these [[Score-based models#Create new sample|methods]].

Other diffusion paper (e.g. DDPM, DDIM) design step (1) differently. 
### New loss function
Recall this [[Score-based models#Connection to Denoising Autoencoders|loss function]].
$$
\mathcal{L}(\theta; \sigma) =\frac { 1 } { 2 } \mathbb { E } _ { \boldsymbol { x } \sim p ( \boldsymbol { x } ) \, , \tilde { \boldsymbol { x } } \sim p _ { \sigma } ( \tilde { \boldsymbol { x } } | \boldsymbol { x } ) } [ | | s _ { \theta } ( \tilde { \boldsymbol { x } } ) \! - \! \nabla _ { \tilde { \boldsymbol { x } } } \log p _ { \sigma } ( \tilde { \boldsymbol { x } } | \boldsymbol { x } ) | | _ { 2 } ^ { 2 } ]
$$

this loss only use one noise level $(\sigma)$, from [[Score-based models#Multiple Noise Perturbation|this section]] we learned that we need to train with multiple noise and this link to [[Score-based models#How score-matching link to SDEs|SDEs]], so loss become
$$
\mathcal{L}(\theta) = \min_\theta \mathbb{E}_{t\sim \mathcal{U}(0, T)} [\lambda(t) \mathbb{E}_{\mathbf{x}(0) \sim p_0(\mathbf{x})}\mathbf{E}_{\mathbf{x}(t) \sim p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))}[ \|s_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))\|_2^2]]
$$
These are changes:
1. **Replace $x$ with stochastic process $x(t)$**: $x(t)$ means noisy image at time $t$ while $x(0)$ is original image.
2. **Replacing fixed $σ$ with time $t$**:  $t$ becomes the parameter controlling the noise level range from 0 (original image) to $T$ (noisy image). (recall this [[Score-based models#^f3e2ff|image]])
3. **Making noise distribution time-dependent**: $p_σ(\tilde{x}|x)$ becomes $p_{0t}(x(t)|x(0))$.
4. **Making the score model time-dependent**: $s_θ(\tilde{x})$ becomes $s_θ(x(t), t)$.
5. **Averaging the loss over different times (noise levels)**: Introducing $\mathbb{E}_{t\sim \mathcal{U}(0, T)}$
6. **Weighting function $λ(t)$**: Added to solve this [[Deriving the Formula#3. Complete loss function|problem]]

### Build Time-Dependent Score-Based Model
Can be any architecture as long as input and output have same dimension, but effective one is U-net. Our model ideally produces $s_\theta(\mathbf{x}(t), t)$ so, input is $x$ (image) and $t$ (time or noise level) but how can we make model understand $t$ ?

Recall from [[Score-based models#How score-matching link to SDEs|link to SDEs]] section we know that $t$ is time in stochastic process which as $t$ increase noise added to image, to make model understand time $t$ the easiest way is just adding $t$ to every intermediate layer in U-net, but 
1. we can't code time as `0,1,2,..T`, since $t$ is continuous in $[0, T]$
2. we can't add single integer like `0,1,...,T` we must represent time $t$ as vector instead, since neural network won't learn much from single integer.

To solve these problems we must create **projection function** that can map any time $t$ (continuous) to higher dimension vector so we can add this vector to U-net, and it can learn time $t$ information, but what projection function we should use ?

To create such a **projection function** we must do like this:
1. In order to convert 1D time $t$ to high-dimension 
	1. Create fixed random N-dimension vector.
	2. We can multiply any time $t$ with that vector.
	3. We will get vector that represent time $t$, difference $t$ will always have difference vector.
2. From (1) is just linear function this is not enough, since time can be complex and nonlinear. We further use [[Fourier Transform|Fourier features]] which is better to capture complex function, it is crucial that they are designed to be sufficiently expressive to encode the temporal information effectively.
3. what I mean is our **projection function** return $[\sin(2\pi \vec{x}), \cos(2\pi \vec{x})]$ where $\vec{x}$ is vector from (1). 

> **Note**:
> Projection means mapping data into a new space.


Lastly we normalize output of network by $\frac{1}{\sqrt{\text{variance}}}$ reason [[Deriving the Formula#3. Complete loss function|here]].


### Create new sample
Recall that for any SDE of the form 
$$
d \mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}
$$
the reverse-time SDE is given by 
$$
d \mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})] dt + g(t) d \bar{\mathbf{w}}
$$

Since we have chosen the forward SDE to be 
$$
d \mathbf{x} = \sigma^t d\mathbf{w}, \quad t\in[0,1]
$$
The reverse-time SDE is given by 
$$
d\mathbf{x} = -\sigma^{2t} \nabla_\mathbf{x} \log p_t(\mathbf{x}) dt + \sigma^t d \bar{\mathbf{w}}
$$

To sample from our time-dependent score-based model $s_\theta(\mathbf{x}, t)$, we first draw a sample from the prior distribution $p_1 \approx \mathbf{N}\bigg(\mathbf{x}; \mathbf{0}, \frac{1}{2}(\sigma^{2} - 1) \mathbf{I}\bigg)$, and then solve the reverse-time SDE with numerical methods.
#### 1. Sampling with Numerical SDE Solvers (Euler–Maruyama)
For an SDE of the form 
$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}
$$
the Euler–Maruyama update rule is 
$$
\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \mathbf{f}(\mathbf{x}_t, t)\Delta t + g(t) \sqrt{\Delta t}\,\mathbf{z}
$$
where $\mathbf{z}$ is a sample from a standard normal distribution, $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
In the context of the reverse-time SDE, the method approximates
$$
d\mathbf{x} = -\sigma^{2t} s_\theta(\mathbf{x}, t) dt + \sigma^t d \bar{\mathbf{w}}
$$
with the discretized iteration: 
$$
\mathbf{x}_{t-\Delta t} = \underbrace{\mathbf{x}_t + \sigma^{2t} s_\theta(\mathbf{x}_t, t)\Delta t}_{Drift} + \underbrace{ \sigma^t\sqrt{\Delta t}\,\mathbf{z}_t}_{Diffusion}
$$

- $s_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ is the estimated score function.
- $g(t)$ is the diffusion coefficient.
- $\Delta t$ is the time step.
- $\mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is Gaussian noise.

We can loop this to generate new sample
#### 2. Sampling with Predictor-Corrector Methods

We combine **Predictor** (Euler–Maruyama) with **Corrector** ([[Score-based models#How to generate new sample ?|Langevin MCMC]]). Recall the classical Langevin MCMC update is given by 
$$
\mathbf{x}_{i+1} = \mathbf{x}_{i} + \epsilon \nabla_\mathbf{x} \log p(\mathbf{x}_i) + \sqrt{2\epsilon} \mathbf{z}_i
$$

but instead of try to get next time step from this, we **refine current time step** and use Euler–Maruyama to get next time step instead, so Corrector update is 
$$
\mathbf{x} \leftarrow \mathbf{x} + \epsilon\, s_\theta(\mathbf{x}, t) + \sqrt{2\epsilon}\,\mathbf{z}
$$

[[Deriving the Formula#How to compute step size for PC|this is detail]] of how to select step size $\epsilon$, then after we refine using **Corrector**, we use **Predictor** to get next time step.
In summary:
1. **Refines the sample using Langevin MCMC (corrector step)** to reduce discretization error.
2. **Propagates the refined sample using an Euler–Maruyama update (predictor step)**.

#### 3. Sampling with Numerical ODE Solvers

In this [paper](https://openreview.net/forum?id=PxTIG12RRHS) they found that, For any Forward SDE of the form 
$$
d \mathbf{x} = \mathbf{f}(\mathbf{x}, t) d t + g(t) d \mathbf{w}
$$

and have Reverse SDE:
$$
d\mathbf{x} = \Big[\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})\Big] dt + g(t) d\bar{\mathbf{w}}
$$

It turns out that Reverse SDE have an associated ODE 
$$
d \mathbf{x} = \bigg[\mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})\bigg] dt
$$

which is known as the **probability flow ODE**. Despite being deterministic (i.e., without the random noise term), this ODE has trajectories whose marginal distributions $p_t(\mathbf{x})$ match those of the original SDE. This means that if you solve this ODE from time $T$ to $0$, starting with a sample from $p_T$, you'll obtain a sample from $p_0$ (typically the data distribution).

Below is a schematic figure showing how trajectories from this probability flow ODE differ from SDE trajectories, while still sampling from the same distribution.
![[Pasted image 20250222022145.png|center]]

Therefore, we can start from a sample from $p_T$, integrate the ODE in the reverse time direction, and then get a sample from $p_0$. In particular, for the SDE in our running example, we can integrate the following ODE from $t=T$ to $0$ for sample generation 
$$
d\mathbf{x} = -\frac{1}{2}\sigma^{2t} s_\theta(\mathbf{x}, t) dt
$$

