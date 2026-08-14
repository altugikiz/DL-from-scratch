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

## 8. Scoring the model: negative log likelihood (NLL)

Eyeballing generated samples is subjective. Need a quantitative score for "how good is this
model" — negative log likelihood, which plays the same role `loss` played in 01-micrograd's
training loop.

**Why "likelihood"?** If the model is good, it should assign high probability to bigrams
that actually occur in the real data. The likelihood of the whole dataset (under the model)
is the product of the individual bigram probabilities (independent events multiply):

```
p(emma) = p(.e) × p(em) × p(mm) × p(ma) × p(a.)
```

**Why take the log?** Multiplying hundreds of thousands of numbers each less than 1
underflows to an unusably tiny number. `log(a×b) = log(a) + log(b)` turns the product into a
sum, which stays numerically manageable.

**Why negative?** `log(x)` for `x` in `(0,1)` is always negative, so the sum of logs is
negative too. Flipping the sign gives a positive number that's easier to reason about as a
"loss" — something to minimize. Lower NLL (closer to 0) = better model; a perfect model
(100% probability on every real transition) would score exactly 0.

```python
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        prob = N[ix1, ix2].float() / N[ix1].float().sum()
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
nll = -log_likelihood
avg_nll = nll / n   # average over all bigrams — normalizes for different word lengths
```

**Word length → bigram count**: an `L`-letter word produces `L+1` bigrams (after adding the
start/end `.`). Verified by hand: `emma` (4 letters) → 5 bigrams, `olivia` (6) → 7, `ava` (3)
→ 4, totaling 16 for the first 3 words — matched the code's `n=16` output exactly.

Result on the full dataset (32,033 words, 228,146 bigrams): **avg nll = 2.4541**. This is
the counting-based model's quality score — the number the neural-net version will later be
compared against.

## 9. From counting to a trainable neural network

The counting approach (`N` matrix) works for bigrams but doesn't scale: trigram needs
27³ ≈ 20K cells, 5-character context needs 27⁵ ≈ 14M cells — real language models with much
longer context need a fundamentally different approach. The fix: replace the fixed count
table with a **trainable weight matrix** learned via gradient descent — same idea as
01-micrograd's `Neuron`, just applied to this problem.

**Input encoding — one-hot, not raw index.** Feeding a raw character index (e.g. `stoi['a']
= 1`) directly into a network would wrongly imply an ordering/magnitude relationship between
letters (as if 'a' < 'b' numerically). Instead, each character is represented as a 27-length
vector with a single `1` at its index and `0` elsewhere — no letter is "bigger" than another.

```python
import torch.nn.functional as F
xs_ex = torch.tensor([1])
xenc = F.one_hot(xs_ex, num_classes=27).float()   # shape (1, 27), 1 at index 1 ('a')
```

**One linear layer + softmax = the whole "network" here:**

```python
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)   # random, trainable

logits = xenc @ W          # (1,27) — matrix multiply; since xenc is one-hot, this is
                             # equivalent to just picking out row W[ix] of W
counts = logits.exp()       # softmax step 1: force everything positive
probs = counts / counts.sum(1, keepdims=True)   # softmax step 2: normalize to sum to 1
```

**Why `exp()` specifically (not some other transform)?** Two reasons: (1) `e^x` is always
positive regardless of sign of `x`, turning arbitrary logits into count-like positive
numbers; (2) `d(e^x)/dx = e^x` — the derivative equals the function itself, the cleanest
possible gradient rule, which is why `e` (not 2, 10, or any other base) is the standard
choice for this kind of exponential transform.

**Clarified: `tanh` and `exp` solve different problems, not competing options for the same
slot.** `tanh` is a bounded (-1 to 1) nonlinearity used *inside* a neuron to prevent
activations from exploding across layers and to allow both positive and negative outputs.
`exp` (inside softmax) is specifically for turning arbitrary real-valued logits into
positive, sum-to-1 probabilities at the *output*. Using `tanh` for softmax would produce
invalid (possibly negative) "probabilities"; using `exp` inside a neuron's activation would
let values explode unboundedly across layers. Each is suited to its own job by construction
— not interchangeable, and not a "which one is simpler" comparison.

**Also clarified: `tan` (trigonometric) vs `tanh` (hyperbolic) are different functions
entirely, despite similar names.** `tan(x) = sin(x)/cos(x)`, unbounded, shoots to infinity
near its asymptotes, derivative `1 + tan(x)^2` is also unbounded. `tanh(x)`, by contrast, is
always strictly between -1 and 1 (approaches but never reaches those bounds — this
"squashing toward but never past ±1" is exactly what saturation looks like), derivative
`1 - tanh(x)^2` stays bounded between 0 and 1. This bounded-ness (both function and
derivative) is why `tanh` is safe inside a neural net and `tan` is never used there.

## 10. Training the bigram network with gradient descent

Built the full training set: every bigram across all 32,033 words as `(input_char_index,
target_char_index)` pairs.

```python
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()   # 228146 — total bigram count across the whole dataset
```

`torch.tensor(...)` converts a plain Python list into PyTorch's native data structure —
required because PyTorch's math operations (matrix multiply, `exp`, `backward`) only work on
`Tensor` objects, not on ordinary Python lists.

Training loop:

```python
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

for k in range(100):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean()

    W.grad = None       # PyTorch idiom for zeroing gradients (equivalent to = 0)
    loss.backward()
    W.data += -50 * W.grad   # note the large learning rate — needed because averaging
                               # over 228K examples makes gradients naturally small
```

`probs[torch.arange(num), ys]` picks out, for every one of the 228,146 examples, the
probability the model assigned to the *actual* next character — the tensor equivalent of
the by-hand `N[ix1,ix2]/N[ix1].sum()` lookup, done for all examples simultaneously.

**Why does `loss.backward()` reach all the way back through the whole computation?**
`loss` is downstream of the full chain `xenc → logits (via W) → counts → probs → loss`.
Every one of `W`'s 729 elements influences `loss` indirectly through this chain (since every
one of the 228K bigrams' probability comes from the relevant row of `W`, and all of them
feed into the single loss value). `backward()` walks back through that whole chain once and
computes, for every element of `W`, exactly how much nudging it would move `loss` — same
mechanism as 01-micrograd's `Value.backward()`, just PyTorch's own (much faster) built-in
autograd doing the work instead of hand-written `_backward` functions.

**Result:** loss went from `3.7590` (iteration 0) to `2.4729` (iteration 99) — and notably,
this is nearly identical to the counting-based model's `avg nll = 2.4541`. Two completely
different approaches (direct counting vs. random init + iterative gradient descent) converge
to essentially the same score, which makes sense: both are solving the same underlying
problem (best-fit bigram probabilities), counting just solves it directly/closed-form while
gradient descent approaches it iteratively.

**Why bother with the gradient descent version if counting gets there just as well (and
faster) for bigrams?** Scalability. A count table for trigram needs 27³ cells, 4-gram 27⁴,
and quickly becomes intractable for the longer contexts real language models use. The neural
network approach doesn't require enumerating every possible context combination — it learns
a function instead — so it scales to arbitrarily long context, which is the direction the
rest of the Karpathy series (MLP → RNN → Transformer) is headed.

## 12. Sampling from the trained network, compared to the counting model

Same sampling loop as the counting model, but each step now does a forward pass through
`W` instead of reading a row of `N` directly:

```python
g = torch.Generator().manual_seed(2147483647)
for i in range(10):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
```

Results, side by side with the counting model's output (same seed):

| Counting model | Neural net model |
|---|---|
| `cexze.` | `cexze.` — identical |
| `momasurailezitynn.` | `momasurailezityha.` — near-identical |
| `konimittain.` | `konimittain.` — identical |
| `llayn.` | `llayn.` — identical |
| `ka.` | `ka.` — identical |
| `da.` | `da.` — identical |
| `staiyaubrtthrigotai.` | `staiyauelalerigotai.` — same start, diverges partway |
| `moliellavo.` | `moliellavo.` — identical |
| `ke.` | `ke.` — identical |
| `teda.` | `teda.` — identical |

8 of 10 generated names came out byte-for-byte identical, the other 2 nearly so. Not a
coincidence — this directly reflects how close the two models' scores were (`2.4541` vs
`2.4729` avg NLL): gradient descent converged `W` to a probability distribution very close
to what direct counting/normalizing produces. Counting and gradient descent are two
different routes to essentially the same answer for this simple problem; the value of the
gradient-descent route is that, unlike counting, it scales to much longer contexts where
enumerating a full count table becomes intractable (trigram, 4-gram, ... eventually
Transformer-scale context).

## 13. Status: `02-makemore-bigram` complete

Built, end to end: bigram counting from 32,033 names → probability matrix → sampling new
names → quantitative scoring via negative log likelihood → reframed as a trainable one-layer
neural network (one-hot input, linear layer, softmax) → trained with real gradient descent
(PyTorch autograd) on all 228,146 bigrams → confirmed it converges to essentially the same
result as direct counting, both in loss (~2.47 vs ~2.45) and in generated samples (8/10
identical). This is the first working example (after 01-micrograd's from-scratch engine) of
using PyTorch's built-in `Tensor`/`autograd` for a real, if small, language model.

Next in the series: extend from 1-character context (bigram) to a learned, multi-character
context using an MLP — the natural next step toward RNN/Transformer-scale models.