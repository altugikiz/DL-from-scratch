# 02-makemore-bigram — Notes

Learning log for building a bigram character-level language model, following Andrej
Karpathy's makemore series (video 1). Reference repo: github.com/karpathy/makemore.

## 1. The dataset

`names.txt` — 32,033 English first names, one per line (`emma`, `olivia`, `ava`, ...),
downloaded from the makemore repo.

```python
words = open('names.txt', 'r').read().splitlines()
```

## 2. Bigram idea

A bigram model asks: "given the current character, what's likely to come next?" — only
looking one character back, nothing further. For `emma`, the consecutive character pairs
are `e-m, m-m, m-a`.

**Start/end token:** wrap each word with a special character (`.`) on both sides —
`.emma.` — so bigrams also capture "what character does a name start with" (`. -> e`) and
"what character does a name end on" (`a -> .`). Using one shared symbol for both start and
end (rather than two separate special tokens) is a deliberate simplification — context
already disambiguates which one it is.

```python
w = words[0]
chs = ['.'] + list(w) + ['.']
for ch1, ch2 in zip(chs, chs[1:]):
    print(ch1, ch2)   # . e / e m / m m / m a / a .
```

`zip(chs, chs[1:])` pairs each character with the next one by zipping the list against a
version of itself shifted by one — a compact way to generate consecutive pairs.

## 3. Counting bigrams across the whole dataset

First pass: a plain dict counting every `(ch1, ch2)` pair across all 32,033 words.

```python
b = {}
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
```

Most common bigrams: `n.` (6763), `a.` (6640), `an` (5438), `.a` (4410), `e.` (3983) — names
very often end in `n` or `a`, and very often start with `a`. This is pure counting/statistics
at this stage, not yet anything resembling a trained model.

## 4. From a dict to a `torch.Tensor` matrix

A dict is fine to inspect by eye, but not for doing math (normalizing into probabilities,
sampling) efficiently. Built a 27×27 matrix instead — 26 letters + 1 special `.` token — so
row = "current character", column = "next character".

```python
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}   # inverse mapping, needed later to turn sampled
                                           # indices back into characters

import torch
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        N[ix1, ix2] += 1
```

Verified `N[stoi['a'], stoi['n']] == 5438`, matching the dict-based count exactly — confirms
the matrix was built correctly.

**Note on `torch.Tensor` vs the `Value` engine from 01-micrograd:** `Value` worked with one
scalar at a time — fine for learning, far too slow at real scale (millions of numbers).
`torch.Tensor` is PyTorch's equivalent of `Value`, but built to hold and operate on entire
blocks of numbers at once, in fast (C++/GPU-backed) code. Note: "tensor" is just the general
math term for a multi-dimensional array (scalar/vector/matrix/higher-dim); `torch.Tensor`
is PyTorch's own implementation of that concept — unrelated to TensorFlow's own (separate,
incompatible) `tf.Tensor`, despite the shared vocabulary.

Visualized the matrix as a heatmap with matplotlib (darker = more frequent), each cell
labeled with the bigram string and its count — confirmed visually that `.a`, `n.`, `a.`,
`an` are the darkest (most frequent) cells, and rare letters like `q` are almost entirely
pale.

## 5. From counts to probabilities

A row of `N` holds raw frequency counts, not probabilities — probabilities must sum to 1
across all possible outcomes. Normalize by dividing each row by its own sum:

```python
p = N[0].float()      # row 0 = what follows the start token '.'
p = p / p.sum()        # p.sum() == 1.0 exactly
```

**Row vs column normalization matters and isn't interchangeable.** The question we want to
answer is "given the current character, what's the probability distribution over the next
character" — that means fixing a row (current char) and normalizing across its columns
(candidate next chars). Normalizing by column instead would answer a different question
("given this character came next, what was probably before it") — the reverse direction,
not useful for left-to-right generation.

Needed `.float()` before dividing — integer division would truncate any ratio below 1 down
to 0, destroying the probability information (e.g. `5438 // 6640` gives `0`, not `~0.82`).

## 6. Sampling — generating new names

Given a probability distribution over "next character," use `torch.multinomial` to draw a
random sample according to those probabilities (higher-probability characters get picked
more often, but it's not deterministic).

```python
g = torch.Generator().manual_seed(2147483647)   # fixed seed = reproducible randomness

out = []
ix = 0
while True:
    p = N[ix].float()
    p = p / p.sum()
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
        break
print(''.join(out))
```

Starts at `ix = 0` (the `.` token, i.e. "start of word"), repeatedly samples the next
character from the current row's distribution, stops when `.` is sampled again (end of
word).

Generated 10 samples: `cexze.`, `momasurailezitynn.`, `konimittain.`, `llayn.`, `ka.`, `da.`,
`staiyaubrtthrigotai.`, `moliellavo.`, `ke.`, `teda.`

Mixed quality — some short outputs (`ka`, `da`, `ke`) look plausibly name-like; longer ones
(`momasurailezitynn`) fall apart. Root cause: bigram only looks one character back, so it
has no sense of overall word length/shape or longer patterns — it can chain together
locally-plausible pairs that add up to something globally implausible.

## 7. Open questions / next steps

- Need a way to *quantitatively* score how good this model is (not just eyeballing
  generated samples) — negative log likelihood (NLL), which plays the same role loss played
  in 01-micrograd's training loop, but here to score a fixed counting-based model rather
  than to drive gradient-based updates (yet).
- The eventual goal (per the video description) is to evolve this counting-based bigram
  model into a neural-network version trained with gradient descent, and later into deeper
  architectures (MLP, then RNN/Transformer) that use more than one character of context.