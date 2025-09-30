---
title: "To ken izers"
date: 2025-29-09
draft: false
toc: false
mathjax: true
tags:
  - NLP
  - LLM
---
# An overview of commonly used tokenizers

Langauge models have taken the world by storm, thanks to transformers ([Vaswani et al.](https://arxiv.org/abs/1706.03762)). A fundamental question when modeling language is how to convert words into numerical representations that computers can understand. One obvious way is to think of each unique word as a one-hot encoded vector of dimension equal to the size of our vocabulary. This is a bit of a problem since there are over 150,000 words in the english language! Surely we can do better than assigning a unique vector to each word in our vocabulary. Behold, tokenizers. Tokenizers are basically algorithms that help prepare a vocabulary for language models (hopefully one that is smaller than the set of unique words in the corpus). Let's jump right into different tokenizers that are commonly used today.

## Byte-Pair Encoding (BPE)

[Byte-Pair Encoding](https://arxiv.org/abs/1508.07909) is an iterative algorithm that builds a vocabulary by identifying the most frequently occuring pairs of symbols in our "fragmented corpus" and merging them to produce a new (sub)word. We start with a base vocabulary consisting of the unique characters appearing in our corpus, after applying some pre-tokenization (lowercasing all words, removing punctuation, etc.). Let's say our corpus is 
```python
"Hello there, how are you?"
```
Then, our base vocabulary $V_0$ would be `['h','e','l','o','t','h','r','w','a','y','u']` and fragmented corpus $C_0$ would be`[('h', 'e', 'l', 'l', 'o'), ('t', 'h', 'e', 'r', 'e'), ('h', 'o', 'w'), ('a', 'r', 'e'), ('y', 'o', 'u')]`. In the first iteration of the algorithm, the most frequently appearing pair of characters in the corpus would be merged and added to the vocabulary. The fragmented corpus would then be modified by performing the same merge across the words as well. In our case, the most frequently occurring pair is `'h', 'e'`. We merge it to create a new subword `'he'` and add it to the vocabulary. Our new vocabulary $V_1$ is `['h','e','l','o','t','h','r','w','a','y','u','he']` and the fragmented corpus $C_1$ is `[('he', 'l', 'l', 'o'), ('t', 'he', 'r', 'e'), ('h', 'o', 'w'), ('a', 'r', 'e'), ('y', 'o', 'u')]`. Repeating this for one more step, we get $V_2$ is `['h','e','l','o','t','h','r','w','a','y','u','he','re']` and $C_2$ is `[('he', 'l', 'l', 'o'), ('t', 'he', 're'), ('h', 'o', 'w'), ('a', 're'), ('y', 'o', 'u')]`. We stop when we've exhausted the fragmented corpus (all single words) or we've reached the required vocabulary size (hyperparameter chosen by us). Usually, an end-of-sentence `<EOS>` and catch-all unknown `<UNK>` tokens are included in the vocabulary. GPT-1 uses BPE with a vocabulary size of 40,000.

The order of merges is important since it decides how a new sentence is tokenized.  Note that we have arbitrarily broken ties for the most frequent pair here. Given a new sentence, we tokenize it by applying merge rules in the same order they were added to the vocabulary. For example. the word 'Cherry' would be tokenized as
```python
'cherry' -> '<UNK>' 'h' 'e' 'r' 'r' 'y' -> '<UNK>' 'he' 'r' 'r' 'y'
```
Since the character `'c'` was not in the vocabulary, it was tokenized with the catch-all token `<UNK>`.

More recently, byte level BPE has replaced character level BPE. Our base vocabulary (`0` - `255`) is the UTF-8 representation of the characters in the corpus (for example, `'h'` is `68`, `'e'` is `65`). To merge two symbols, say `'h'` and `'e'`, we add `'he'` as `256` to our vocabulary. The advantage of this is universal representation. Any character or symbol can be represented with atmost 4 bytes using the UTF-8 encoding. We apply the learned merge rules sequentially till we get the final tokenization. This means no more `<UNK>` tokens!

## WordPiece Encoding

WordPiece is also an iterative algorithm similar to BPE except for three differences:
* All characters and subwords in the fragmented corpus (except prefixes) are encoded with `##` before the word. For example, the word `'how'` in $C_0$ is `('h', '##o', '##w')`. `'##o'` and `'##w'` would be merged as `'##ow'`.
* The pair to be merged is chosen using a score 
$$ \text{score} = \text{freq of pair} / (\text{freq of first element}\times \text{freq of second element})$$ The score is designed to prioritize merging pairs that occur together frequently but don't occur by themselves in other words. For example, consider the pairs `('ad', '##vantage')` and `('un', '##able')`. `'un'` and `'able'` occur very frequently as parts of other words (undo, capable, etc.) but `'ad'` and `'vantage'` probably occur less frequently so adding `'advantage'` to the vocabulary before `unable` makes sense (assuming the two words occur fairly frequently in the corpus to begin with).
* The merge rules are not stored, only the final vocabulary.

A new word is tokenized by recursively finding the longest prefix of the word from the vocabulary. If any part of the remaining word can't be tokenized with the vocabulary, we tokenize the entire word as `'<UNK>'`. Here are two examples to illustrate. We don't go into the corpus or the vocabulary to sidestep tedious details.
```python
'heresay' -> 'he' '##resay' -> 'he' '##re' '##say'  
```
```python
'totem' -> 'to' '##tem' -> 'to' '##te' '##m' -> 'to' '##te' '<UNK>' -> '<UNK>' 
```

## Unigram Encoding

BPE and WordPiece tokenization algorithms start with a small base vocabulary and iteratively build it up. On the other hand, Unigram encoding starts with a large vocabulary and iteratively removes words till the desired size is reached.

A Unigram language model is a model that assumes each token to be independent of the rest of the context, i.e., $P(x_t \vert x_{t-1},...,x_1) = P(x_t)$. The probability of seeing a token $x$ at any position is only dependent on how frequently it appears in the corpus $(\text{freq of token} / \text{sum of freq of all tokens})$. Extending this to words, the probability of seeing a word $w = (x_1x_2...x_k)$ under the unigram model is $\prod_{i=1}^{k} P(x_i \vert x_{\\<i}) = \prod_i P(x_i)$. The unigram tokenization algorithm starts with a large base vocabulary $V_0$, usually constructed using BPE on characters for a large number of merge steps, or considering every subword of the words in our corpus. We then compute a loss over the corpus as follows
* For each word $w$ in the corpus, the loss associated with this word under a unigram model with vocabulary $V_i$ is the probability of the best tokenization of the word w.r.t. the vocabulary. That is, $$ L(w) = \max_{\tau} \prod_{t\in \tau} P(t)$$ where $\tau$ is every possible tokenization of the word under the vocabulary.
* The loss of the whole corpus $(w_1, w_2,...,w_N)$ is defined as $$L = -\log \prod_{w_i} L(w_i) = -\sum_{w_i} \log L(w_i)$$ We use negative log probabilities since the product of probabilities could become arbitrarily small.

In each iteration of the algorithm, we find the token whose removal would lead to the smallest increase in the loss. We stop when we reach the desired vocabulary size. Given a new word, its tokenization is the one corresponding to the best probability under the vocabulary, i.e., $\argmax_{\tau} \prod_{t \in \tau}P(t)$.

In practice, the [_Viterbi algorithm_](https://en.wikipedia.org/wiki/Viterbi_algorithm) is used to find the compute the loss. We will look at a simple DP algorithm the find the best tokenization of a word here. Given a word $w$, we get that $$L(w[:i]) = \min_{sfx \in V} \left\{L(w[:i-j]) \cdot P(sfx) \right\} $$ where $sfx$ is all possible suffixes of $w[:i]$ in the vocabulary and $j$ is the length of the corresponding suffix. Memoization of this gives us the DP algorithm. This can be implemented efficiently using a _trie_ data structure. We will not be looking at how the tokens to be removed are chosen at each iteration since it is out of the scope of this blog post.

A naive implementation of all these tokenizers in python [can be found here](https://github.com/shankram/LLMs-from-scratch/blob/main/Tokenizers/Tokenizers.ipynb).