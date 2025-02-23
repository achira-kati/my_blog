## TL;DR;
1. Self-Attention Mechanism:
	- Replaces traditional RNNs/LSTMs with attention
	- Allows direct modeling of relationships between all words in a sequence
	- Computes attention using Queries, Keys, and Values (Q, K, V)

2. Multi-Head Attention:
	- Runs multiple attention mechanisms in parallel
	- Captures different types of relationships between words
	- Combines results for richer representations

3. Architecture:
	- Encoder-Decoder structure
	- Positional encodings to maintain word order information
	- Feed-forward neural networks
	- Layer normalization and residual connections

##  Example translation task

### 1. What data will look like
ทำไมไม่ยกตัวอย่างภาษาไทยง่ายๆคือภาษาไทยมันไม่มีการแบ่งวรรคเหมือนภาษาอื่นเวลานับ token เดียวงง

```json
{ "de": "Erstes Kapitel", "en": "CHAPTER I" }
```

this will be tokenized (this case just split by space) then 

```python
class TranslationDataset(Dataset):
	...
	def __getitem__(self, idx):
		...
		return { 
		'encoder_input': encoder_input, # token of ["Erstes", "Kapitel"]
		'decoder_input': decoder_input, # token of ["<SOS>", "CHAPTER", "I"]
		'label': label,                 # token of ["CHAPTER", "I", "<EOS>"]
		'encoder_mask': encoder_mask,   # padding mask
		'decoder_mask': decoder_mask    # padding & causal mask (anti-cheat) 
		}
```

Why is the dataset in this format? read through each data flow through each architectural component you will get answer. 

### 2. Go to each architectural component
![[transformer_model.png]]

Start with encoder left side.

#### 2.1 Encoder Input
In this translation example, the encoder input is `[132, 142]`, which represents the tokens `["Erstes", "Kapitel"]`. The encoder processes these tokens to generate a feature that is later passed to the decoder.
#### 2.2 Encoder Input Embedding. [[What is Embedding ?]]
Just turn input sequence into vector of model dimension `(Batch, seq_length, d_model)`, in this example shape will be `(1, 2, 512)` if use $d_{model} = 512$, batch is 1 because we have only one sentence.

#### 2.3 Positional Encoding
- To make transformer understand order of the sequence. Can be use many ways depend on sequence dimension and length (e.g. 1D, 2D fixed sinusoidal position encoding, RoPE, Learnable position encoding, etc.). In this case use 1D fixed sinusoidal position encoding then add to input embedding.
- Shape is still `(1, 2, 512)`, since position embedding just add to input embedding. 

#### 2.4 Encoder Layer (1 Block)
1. [[Multi-Head Attention]]: 
	- Core component of transformer. 
	- Used to compute attention score(ideas, relation) within input sequence called ==self-attention==, since $Q,K,V$ is same input sequence.
	- This layer is where `encoder_mask` is used for mask padding.
	- Outputs new representations of each word ($\Delta \vec{E}$)
	- Shape is still`(1, 2, 512)`, since it's just new more meaningful embedding. 
2. Add&[[Normalization]]: 
	- Add original embedding with new representations embedding (result from multi-head attention), this add also called residual connections.
	- Then we do normalization (this case use layer norm) to keep value in a reasonable range (prevent exploding & vanishing gradients).
	- Shape is still`(1, 2, 512)`
	- Note is if the word is masked, result from multi-head attention is 0 then we add to original embedding $0+\vec{E} = \vec{E}$ , that means model not learn anything from multi-head attention. 
	
3. Feed Forward:
	- MLP network that cover 2/3 of transformer model.
	- Have 2 linear layer called up and down projection
	- Up projection increase dimension from 512 ($d_{model}$) to 2048 ($d_{ff}$), Shape is now`(1, 2, 2048)`
	- Down projection decrease dimension back from 2048 ($d_{ff}$) to 512 ($d_{model}$), then shape goes back to `(1, 2, 512)`
	- Ideas is we hopefully model will capture more complex ideas, relation from result of multi-head attention. 
	- ==Some research show that facts live here.==
1. Add&[[Normalization]]: same as 2.
	- Shape is still`(1, 2, 512)`

#### 2.5 Encoder 
Just repeat encoder layer $N$ times, depends on how many you want. We done with encoder now Next, start with decoder right side. No matter how many head we still have shape `(1, 2, 512)`
#### 2.6 Decoder Input
Recall that dataset is like this
```json
{ "de": "Erstes Kapitel", "en": "CHAPTER I" }
```
The decoder input is `["<SOS>", "CHAPTER", "I"]`. We add `<SOS>` (start of sentence) so that the decoder recognizes the beginning of a sentence. When translating a new sentence such as `"zweites Kapitel"`, we simply feed `<SOS>` to initiate the process.

#### 2.7 Decoder Embedding
Same structure as encoder embedding but use different parameters. Shape is `(1, 3, 512)`

#### 2.8 Positional Encoding
Same thing, since use 1D fixed sinusoidal position encoding. Same value added to position $1,..,L$ $L$ is seq_length. Shape is still `(1, 3, 512)`
#### 2.9 Decoder Layer (1 Block)
1. Masked Multi-head Attention:
	- Self-attention, since $Q,K,V$ is same input sequence.
	- Same as in encoder except we use `decoder_mask`, that not only mask padding but also mask future token, so model will not cheating when compute self-attention. [[Multi-Head Attention#Masking Types|More detail ]]
	- Shape is still `(1, 3, 512)`
2. Add&Norm:
	- Same as in encoder.
	- Shape is still `(1, 3, 512)`
3. Multi-head attention:
	- This is ==cross-attention==, since $Q$ from decoder but $K,V$ from encoder.
	- This is where decoder use knowledge from encoder to predict next token, in this example decoder knows how to write new word in English from knowledge that encoder give from sentence in German.
	- $Q$  shape is `(1, 3, 512)`, from decoder
	- $K, V$ shape is `(1, 2, 512)`, from encoder
	- `seq_length` can be vary but `batch_size` must equal.
	- Output shape is `(1, 3, 512)`
4. Add&Norm:
	- Same as in encoder.
	- Shape is still `(1, 3, 512)`
5. Feed Forward:
	- Same as in encoder.
	- Shape is still `(1, 3, 512)`
6. Add&Norm:
	- Same as in encoder.
	- Shape is still `(1, 3, 512)`

#### 2.10 Decoder
Just repeat decoder layer $N$ times, depends on how many you want. Shape is still `(1, 3, 512)`

#### 2.11 Generate output
1. Linear:
	- Called projection layer because it project from $d_{model}=512$ back to vocab size depends on what tokenizer that you use in very first step
	- Shape is now from `(1, 3, 512)` to `(1, 3, vocab_size)`
2. Softmax:
	- Then we apply `softmax` to convert all vocab into probabilities in order to predict next token
	- Shape is still `(1, 3, vocab_size)`
3. We now can get predict token in multiple way, simplest one is called "Greedy Search" just find `argmax` or token that give us maximum possibility. Other strategies [here](https://huggingface.co/docs/transformers/main/en/generation_strategies#decoding-strategies)
	- Shape now is `(1, 3, 1)`, since we do `argmax`.
4. We hopefully those 3 token from  `(1, 3, 1)` are word `["CHAPTER", "I", "<EOS>"]` that will match our label `["CHAPTER", "I", "<EOS>"]`.

### 3. When Training
You can see that decoder take input `"Erstes Kapitel"` to encoder and generate output `["CHAPTER", "I", "<EOS>"]` shape `(1, 3, 1)` from decoder and we have label `["CHAPTER", "I", "<EOS>"]` so we can compute loss each token, then we can train to minimize to loss.

### 4. When inference
Example we will translate `"zweites Kapitel"`, then we feed this to encoder, wait until encoder done, now we have feature to feed to decoder when it compute cross-attention, then we start decoder by feed token `<SOS>`  to decoder it will start generate next token and we keep looping until it generate `<EOS>`. [Animated process](https://jalammar.github.io/images/t/transformer_decoding_2.gif)
```
[<SOS>]                                # timestamp 0
[<SOS>, "second"]                      # timestamp 1
[<SOS>, "second", "chapter"]           # timestamp 2
[<SOS>, "second", "chapter", <EOS>]    # timestamp 3
```
Note that we can use many [decoding methods](https://medium.com/nlplanet/two-minutes-nlp-most-used-decoding-methods-for-language-models-9d44b2375612) to select next token. In this example use simplest greedy search which choose next token that have maximum probability. 
# Useful links
1. [Full implementation guide](https://www.youtube.com/watch?v=ISNdQcPhsts)
2. [How a Transformer works at inference vs training time](https://www.youtube.com/watch?v=IGu7ivuy1Ag)
3. [Best video to get intuition](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=1314s)
4. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)