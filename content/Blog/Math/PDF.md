PDF (Probability Density Function) is a function that describes the relative likelihood of a continuous random variable taking on a specific value. Here are the key points about PDFs:

1. Mathematical notation: For a random variable $X$, the PDF is often denoted as $f_X(x)$ or just $f(x)$
2. ==Properties==:
	* Non-negative: $f(x) \geq 0$ for all $x$
		* Probability is always non-negative in real world
		* Negative probability doesn't make physical sense
	* Total area equals 1: $\int_{-\infty}^{\infty} f(x)dx = 1$
		* The integral represents total probability across all possible values
		* By axioms of probability, total probability must equal 1
		* If area was > 1, we'd have >100% probability
		* If area was < 1, we'd have missing probability
		* This property ensures probabilities are properly normalized
3. Probability calculation:
	* For an interval $[a,b]$, probability is: $P(a \leq X \leq b) = \int_a^b f(x)dx$
	* For a single point: $P(X = x) = 0$ for continuous distributions
4. Common examples:
	* Normal distribution: $f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
	* Uniform distribution: $f(x) = \frac{1}{b-a}$ for $x \in [a,b]$

The PDF is different from PMF (Probability Mass Function), which is used for discrete random variables.