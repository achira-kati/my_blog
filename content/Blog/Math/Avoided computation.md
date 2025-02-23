Operations that can lead to numerical instability or computational issues are generally avoided or reformulated. Here are a few examples:

- **Division by small numbers:**  
    When the denominator can be near zero, division may cause very large values or gradients, which destabilizes training. This is why, in our example, multiplying by the noise standard deviation is preferred over direct division.
    
- **Exponentiation of large values:**  
    Exponentiating large numbers can lead to overflow or numerical instability. Many implementations use tricks like the log-sum-exp trick to handle this.
    
- **Logarithms of small or non-positive numbers:**  
    Taking the logarithm of very small numbers can result in large negative values or undefined behavior if the input is zero or negative.
    
- **Subtraction of nearly equal numbers:**  
    This can cause cancellation errors, where significant digits are lost, leading to reduced numerical precision.
    

In summary, people tend to avoid or carefully manage operations like division (especially when the divisor might be small), exponentiation, logarithms under risky conditions, and operations that can lead to cancellation errors in order to ensure stability and reliability during training.