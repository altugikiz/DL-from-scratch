# 04-makemore-mlp-internals — Notes

Learning log for diagnosing and fixing MLP training internals, following Andrej Karpathy's
makemore series (video 3): activation/gradient statistics, saturated tanh, weight
initialization, and (upcoming) Batch Normalization.

## 1. Starter code

Rebuilt the same pipeline as `03-makemore-mlp` (character-level dataset, `build_dataset`,
train/dev/test split) as a clean starting point, plus a couple of generalizations:

```python
vocab_size = len(itos)   # instead of hardcoding 27, for more flexible code
```

## 2. Scaling up the model

```python
n_embd = 10      # embedding size per character (was 2 in 03-makemore-mlp)
n_hidden = 200    # hidden layer size (was 100)
```

Named variables instead of hardcoded numbers — easier to tune and reason about. Flattened
input size to the hidden layer is `n_embd * block_size = 10*3 = 30` (was `2*3=6` before) —
each character now gets a richer, higher-dimensional representation.

**Why does a bigger embedding (10 dims vs 2) give more "expressive power"?** With only 2
dimensions, a character's embedding has to compress every distinguishing property (vowel vs
consonant, common word-start vs word-end letter, phonetic neighbors, frequency, special
pairings like `q` almost always preceding `u`) onto just 2 axes — forcing multiple unrelated
properties to share the same few numbers. More dimensions give the model more "room" to
represent multiple independent properties without them interfering with each other.

**Important caveat, corrected during discussion:** it's *not* accurate to say the model
assigns one specific embedding axis to one specific human-interpretable property (like "this
axis = follows u"). What's actually learned is a **distributed representation** — the
information about how a character behaves is spread across the interaction of its embedding
vector with `W1` and `W2` jointly, shaped purely by "does this reduce the loss," with no
explicit, individually-readable meaning per dimension. We can observe that a character like
`q` ends up in a distinctly different part of embedding space than most other letters (since
it behaves unusually), but we generally can't point to "this one number is the q-follows-u
feature." This is the general interpretability problem in deep learning.

Total parameters: `270 (C) + 6000 (W1) + 200 (b1) + 5400 (W2) + 27 (b2) = 11897`.

## 3. Diagnosing a broken initial loss

With naive `torch.randn` init for all parameters, initial loss on a random minibatch was
**27.15** — even worse than the `17.9455` seen in 03-makemore-mlp (larger model compounds the
problem). Theoretical "uniform guessing" loss is `-log(1/27) ≈ 3.296`.

**Diagnosis — inspected raw `logits`:** ranged from about `-27.8` to `+19.6`, a spread of
~47. Explained why this destroys the loss: `exp()` grows *exponentially* with its input, so
even a moderate difference in logits becomes an astronomically large difference in `exp()`
output — e.g. `exp(19.56) ≈ 3.14×10^8` vs `exp(-27.83) ≈ 8.2×10^-13`, a ratio of roughly
`3.8×10^20`. Softmax normalizes these into probabilities, so this huge gap in `exp()` output
means the model assigns ~100% probability to one character and ~0% to the rest — extreme
overconfidence. When that overconfident guess is wrong (which happens often when weights are
still random), `-log(near-zero)` explodes toward a huge loss value.

**exp() and log() — quick recap of where each is used:** `exp()` is step 1 of softmax,
turning arbitrary (possibly negative) logits into strictly positive numbers, which are then
normalized (divided by their sum) into a valid probability distribution. `log()` is used
separately, at the loss stage, to turn a product of many small probabilities (which would
underflow to an unusably tiny number) into a sum instead (`log(a×b) = log(a)+log(b)`),
negated to get a positive "loss to minimize." Different tools solving different problems in
the pipeline, not competing options.

**Fix — scale down the output layer at initialization:**

```python
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0.0   # zero-init bias entirely
```

Zeroing `b2` (rather than just scaling it down) removes any baseline preference for specific
characters at initialization — the most "neutral" starting point. Result: initial loss
dropped to **3.334**, very close to the theoretical `~3.296`.

## 4. Diagnosing saturated tanh

Even with the fixed initial loss, inspected `hpreact` (the pre-activation values going into
`tanh`) and found values like `7.6, 9.18, 11.64` — large magnitudes. Checked what fraction
exceed `|x| > 0.99`:

```python
(hpreact.abs() > 0.99).float().mean()   # 0.8561 — 85.6% of pre-activations!
```

**Why this matters:** `tanh(x)` for large `|x|` is squashed extremely close to ±1 (e.g.
`tanh(9) ≈ 0.99999998`), and its derivative `1 - tanh(x)^2` approaches 0 as `tanh(x)`
approaches ±1. So a neuron whose pre-activation is large has an almost-zero gradient — it's
"saturated" and essentially stops learning during backward pass, no matter how many training
steps run. With 85.6% of neurons in this saturated regime, the vast majority of the
200-neuron hidden layer was effectively frozen/dead at initialization.

**First-pass fix attempt — scale down `W1` (and `b1`) too:**

```python
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * 0.2
b1 = torch.randn(n_hidden, generator=g) * 0.01
```

Result: saturation ratio dropped from `0.8561` to `0.3642` (36.4%) — a real improvement,
but still well above the healthy target (video aims for closer to ~5%). Loss stayed
essentially the same (`3.333`), since fixing saturation is about gradient health / trainability
going forward, not the loss value at this single snapshot.

## 6. Kaiming init — the mathematically principled scale

Instead of guessing a scale factor for `W1` by trial and error, derived it: `hpreact` is a
sum of `fan_in` (30, here) independent weighted terms. Variances of independent random
variables add when summed, so summing `fan_in` terms roughly multiplies variance by
`fan_in`, and standard deviation by `sqrt(fan_in)`. To keep `hpreact`'s spread from growing
with `fan_in`, scale the weights down by the inverse: `W ~ randn * (1/sqrt(fan_in))`.

```python
fan_in = n_embd * block_size   # 30
W1_scale = fan_in ** -0.5       # ≈ 0.1826
```

Notably close to the `0.2` guessed earlier by trial and error — the guess happened to be
good, but now the reasoning behind the right scale is understood rather than assumed.

**Result was underwhelming on its own:** loss `3.3318` (fine, expected), but saturation
ratio only dropped to `0.3237` (32.4%) — still far from the ~5% target. **Why Kaiming init
alone isn't a full fix:** it controls the *average* variance of `hpreact`, but individual
values are still drawn from a random distribution — some will land in the tail (e.g. |x| > 2)
purely by chance, and `tanh` already saturates noticeably past that point. More importantly,
this kind of careful initial-scale tuning only holds at the very start of training — as
weights update over many steps, the distribution of `hpreact` can drift again, and manually
re-balancing this by hand becomes impractical, especially as networks get deeper. This is
the motivation for Batch Normalization.

## 7. Batch Normalization

**Core idea:** instead of hoping initialization keeps `hpreact` well-scaled, force it to
have mean 0 and std 1 on every forward pass, by explicitly normalizing it right before the
nonlinearity:

```python
normalized = (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True)
```

`mean(0, keepdim=True)` / `std(0, keepdim=True)` compute per-neuron statistics across the
batch dimension (dim 0 = examples) — for each of the `n_hidden` neurons, the mean/std over
all examples in the current minibatch.

**But forcing an exact mean-0/std-1 every time removes the network's ability to choose a
different scale/shift if that's actually useful for learning.** Fix: after normalizing, add
two learnable parameters — `bngain` (scale) and `bnbias` (shift) — initialized to have no
effect (`gain=1`, `bias=0`), letting gradient descent adjust them if a different
scale/offset turns out to help:

```python
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
```

Also added `(5/3)` as an extra scaling factor on `W1`'s Kaiming-init scale — a `tanh`-specific
gain constant used alongside Kaiming init in the video.

Total parameters grew from 11,897 to **12,297** (`+200` for `bngain`, `+200` for `bnbias`,
matching `n_hidden=200`).

## 8. Training with Batch Norm + learning rate decay

```python
max_steps = 30000
for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 15000 else 0.01   # learning rate decay: large steps early, fine-tune later
    for p in parameters:
        p.data += -lr * p.grad
```

Loss went from `3.3147` (iter 0) down into the `~2.0-2.4` range by iter 30000, with visibly
less noisy fluctuation after the LR drop at iter 15000.

**Evaluating with Batch Norm requires care:** during training, each minibatch normalizes
using *its own* mean/std. But evaluation runs over a full split at once, not a small
minibatch — so a consistent mean/std needs to be used instead of ad-hoc per-call statistics.
Used the simplest approach here: compute the *entire training set's* `hpreact` mean/std once
(under `torch.no_grad()`, since no gradients are needed for this) and reuse those fixed
statistics for all evaluation:

```python
with torch.no_grad():
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    hpreact_mean = hpreact.mean(0, keepdim=True)
    hpreact_std = hpreact.std(0, keepdim=True)
```

(Note: a running mean/std updated incrementally during training is the more standard
approach in practice, not yet implemented here.)

## 9. Result: Batch Norm's measurable effect

| | 03-makemore-mlp (no BatchNorm) | 04 (with BatchNorm) |
|---|---|---|
| Train loss | 2.3136 | **2.1589** |
| Dev loss | 2.3201 | **2.1724** |

Both dropped by about 0.15, and train/dev remain close (diff 0.0135) — confirms the
improvement is genuine generalization gain, not new overfitting. Batch Norm delivered this
by (1) largely fixing the saturated-tanh problem (forcing `hpreact` into a healthy range
every forward pass, most neurons stay trainable), (2) making the network much less sensitive
to exact initialization scale (the normalization compensates for imperfect init
automatically), and (3) keeping inter-layer signal magnitudes more stable throughout
training, letting gradient descent proceed more efficiently.

## 10. Open questions / next steps

- Replace the "compute train-set stats once at the end" evaluation approach with a proper
  running mean/std tracked incrementally *during* training (the standard practice) —
  current approach works but is a simplification.
- Video covers additional diagnostic visualizations not yet done here: forward-pass
  activation statistics across layers, backward-pass gradient statistics, and "update:data
  ratio" over time — standard tools for monitoring a deep network's training health.
- Video also rewrites this same model using PyTorch's `nn.Linear` / `nn.BatchNorm1d` /
  `nn.Tanh` classes ("PyTorch-ifying") instead of manually managed tensors — worth doing to
  get closer to idiomatic PyTorch code.
- E01 exercise (from video): try initializing all weights and biases to exactly zero and
  observe/explain the partial-training behavior that results.
- E02 exercise: after training a small BatchNorm MLP, "fold" the batchnorm gamma/beta into
  the preceding linear layer's weights and biases, and verify the folded version produces
  identical forward-pass output — showing BatchNorm's training-time-only role.
- Residual connections and the Adam optimizer are flagged in the video as topics for a later
  session, not covered here.