![video](https://youtu.be/spUNpyF58BY?si=gB23eqWE1uKQ9wfi)

**Discrete Fourier Transform (DFT)**  
Converts a time-domain signal to a frequency-domain representation and back via the Inverse DFT (IDFT):

$$
\begin{aligned} X_{k} &= \sum_{n=0}^{N-1} x_n \, e^{-i\,2\pi \,k\,n/N}, \quad k = 0, 1, \dots, N-1, \\ x_{n} &= \frac{1}{N}\sum_{k=0}^{N-1} X_k \, e^{i\,2\pi \,k\,n/N}, \quad n = 0, 1, \dots, N-1. \end{aligned}
$$

### Key Points:
1. **Bijection**: The DFT maps an **$N$-dimensional time-domain signal** to an **$N$-dimensional frequency-domain signal**, ensuring invertibility.  
2. **Frequency Bins**: The index $k$ represents frequencies $f_k = \frac{k \cdot F_s}{N}$ Hz, where $F_s$ is the sampling rate.  
3. Example: If $F_s = N$ (e.g., 1000 Hz sampling for 1000 samples), then $f_k = k$ Hz.  
4. Generally, frequencies depend on $F_s$ and $N$, not just $k$.  
