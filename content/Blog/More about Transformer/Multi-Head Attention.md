
This is core component of transformer. 
![Multi head attention](MultiHeadAttention.png)

## Overview

Multi-head attention divides the model dimension ($Q$, $K$, $V$) by the number of heads, applies attention independently to each head, concatenates the results, and finally applies a learned linear transformation ($W^O$) to produce the final output.

### Why Multiple Heads?
Input sequences contain various types of relationships and patterns:
- Head 1 might capture grammatical structure
- Head 2 might focus on semantic relationships
- Deeper heads might learn more complex, abstract patterns

The final learned parameter $W^O$ combines and refines these different aspects into a coherent representation, so 
$$
MultiHead(Q,K,V)=Concat(head_1, ..., head_h)W_o
$$ 

## Attention Mechanisms
### One-Head Attention
#### Components and Dimensions

| Component | Shape                 | Description                      |
| --------- | --------------------- | -------------------------------- |
| Query (Q) | (seq_length, d_model) | Input sequence transformation    |
| Key (K)   | (seq_length, d_model) | Used to compute attention scores |
| Value (V) | (seq_length, d_model) | Used to compute final output     |
#### Learnable Parameters

| Parameter | Shape              | Purpose              |
| --------- | ------------------ | -------------------- |
| $W^Q$     | (d_model, d_model) | Query transformation |
| $W^K$     | (d_model, d_model) | Key transformation   |
| $W^V$     | (d_model, d_model) | Value transformation |
#### Attention formula
$Attention(Q,K,V) = softmax({QK^T \over {\sqrt{d_k}}})V$  

#### Process Steps
1. Apply linear transformation to get $Q\prime$ $K\prime$ $V\prime$ .
2. Do dot product of $Q\prime$ and $K\prime$. 
	-  For self-attention: Think of this like find attention score or find ideas, relation within input sequence (e.g. gramma, abstract ideas, we hope fully model learn something complex as we going deeper(more heads))
	- For cross-attention: similar to self-attention except we $Q\prime$ and $K\prime$ are form difference sequence like in translation task, so this is like find attention score from both sequence (e.g. ideas of how we translate Thai to English)
3. Divide by  $\sqrt{d_k}$ : Just for numerical stability.
4. Apply `softmax`, so $softmax({QK^T \over \sqrt{d_k}})$:
	- Converts attention scores into probabilities (values between 0 and 1)
1. Do dot product of $softmax({QK^T \over \sqrt{d_k}})$ and $V$: 
	- Think of this like after we know ideas, relation for each word (attention score $QK^T$) then what exactly should we added to the embedding to make some reflection, so this is like dot product with $V$ to make new representations $\Delta \vec{E}$ 
2. Now we got new representations for all word $\Delta \vec{E}$  that can be add to original words $\vec{E}$  (in Add&Norm layer), so we can get better, more meaningful words.
### Practical Example
Consider the sequence "A blue cat", word "cat" embedding is $\vec{E_3}$ then model compute self-attention, start with $QK^T$ and now model know attention score or an ideas that is this cat is a weird blue cat then model compute ${QK^T \over \sqrt{d_k}}V$ to get new vector that reflect this knowledge $\Delta \vec{E_3}$, now we $\vec{E_3} + \Delta \vec{E_3}$ (done in Add&Norm) , so we get new $\vec{E_3}$ that change meaning from "cat" to "blue cat".

| $\vec{E_3}$ meaning before | $\vec{E_3}$ meaning after |
| :------------------------: | :-----------------------: |
|        ![[cat.png]]        |     ![[blue_cat.png]]     |

## Masking Types
### Decoder Self-Attention Masking (causal mask)

```python
[ 1 0 0 ]  # First position can only look at itself
[ 1 1 0 ]  # Second position can look at first and itself
[ 1 1 1 ]  # Third position can look at everything up to itself
```
-  When compute $QK^T$, sets upper triangle to `-inf` before `softmax`
- Prevents model from seeing future tokens during training
- Essential for autoregressive generation
### Padding Masking

```python
[ 1 1 1 0 0 ]
[ 1 1 1 0 0 ]
[ 1 1 1 0 0 ]
```
- Applied to both encoder and decoder
- Masks padding tokens to prevent them from contributing to attention

## Video
- [Best video to get intuition](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=1314s)