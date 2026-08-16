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

## 5. Open questions / next steps

- `0.2` for `W1`'s scale was a guessed value, not derived — need to work through the
  mathematically principled way to pick this scale ("Kaiming init," per the video), rather
  than trial and error.
- Saturation is still too high (36% vs a healthy ~5%) — Kaiming init should get this much
  closer to healthy.
- Main topic of the video not yet reached: **Batch Normalization** — normalizing each
  layer's pre-activations during training so this kind of careful, fragile initialization
  tuning becomes far less necessary.
- Later in the video: visualizing forward-pass activation statistics, backward-pass gradient
  statistics, and the "update:data ratio" as standard diagnostic tools for monitoring
  network health during training.
- Also later: rewriting the model using PyTorch's `nn.Linear` / `nn.BatchNorm1d` /
  `nn.Tanh` layer classes instead of manually managed tensors — closer to how real PyTorch
  code is typically written.