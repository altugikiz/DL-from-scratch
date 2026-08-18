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

## 11. Batch Norm, explained from first principles (follow-up deep dive)

Revisited Batch Norm after the initial pass felt too abstract — worked through it with a
concrete 4-example numeric walkthrough instead of just formulas.

**The problem, restated simply:** `hpreact = x1*w1 + ... + x30*w30 + b` is a sum of many
terms, so its scale is not naturally controlled — it can easily land far from 0 (e.g. `8.0`,
`9.18`), and `tanh` saturates hard past roughly `|x| > 2`, killing that neuron's gradient.

**Batch Norm's core move: instead of *hoping* the raw scale stays reasonable, force it into
a safe range every single forward pass, by measuring and correcting it on the spot.**

Worked through a toy example — 4 examples in a minibatch, one neuron, raw `hpreact` values
`[8.0, 2.0, 6.0, 4.0]` (all large, saturation risk):

1. Compute the mean across the batch: `(8+2+6+4)/4 = 5.0`
2. Compute the std across the batch: `≈2.24`
3. Standardize each value: `(x - mean) / std` → `[1.34, -1.34, 0.45, -0.45]`

Result: guaranteed mean-0, std-1 output, *regardless of what the raw values were* — the
large, saturation-prone values `[8,2,6,4]` become small, tanh-safe values. This is why the
statistics need to be computed over multiple examples at once (a "batch"): mean/std of a
single value is undefined in a useful sense (std of one number is 0, causing a `0/0` divide
in the normalization formula) — normalization is only meaningful across a group.

**Why not just always force mean-0/std-1 forever, with no escape hatch?** Because sometimes
a wider or shifted range might genuinely help the network for a particular problem — hard-
forcing normalization removes that flexibility. Fix: after normalizing, apply two learnable
parameters:

```
final = gamma * normalized + beta
```

`gamma` (scale) and `beta` (shift), initialized to have zero effect (`gamma=1`, `beta=0`),
so training starts from "just the safe, normalized version" but gradient descent can learn
to stretch/shift it later if that reduces loss.

**Batch Norm's three concrete benefits, restated plainly:**
1. Prevents saturation — `hpreact` is kept in a safe range every forward pass, so neurons
   stay trainable instead of freezing.
2. Makes exact initialization scale far less critical — even an imperfectly-scaled `W1` gets
   auto-corrected by the normalization step each time.
3. Keeps signal magnitudes stable *throughout* training, not just at initialization — as
   weights drift during training, Batch Norm re-corrects the scale every single step, so
   drift doesn't accumulate.

**Batch requirement, and what this means for evaluation:** because computing a meaningful
mean/std needs multiple examples, Batch Norm can't run on a single example in isolation
(mean of 1 number = itself, std = 0, causing a divide-by-zero). During training this is
naturally satisfied (minibatches of 32). During evaluation/inference on a single example
(e.g. generating one name), there's no batch to compute stats from — hence why this project
precomputed fixed mean/std over the *entire* training set (section 8) and reused those as a
stable reference at evaluation time, rather than trying to compute per-example statistics.

## 12. Visualization tools — activation and gradient histograms

**Activation histogram** — plot the distribution of `h` (post-tanh hidden layer output)
across all neurons and examples in a batch:

```python
plt.hist(h.view(-1).tolist(), 50)
```

What to look for: if the histogram piles up almost entirely at the extremes (near -1 and
+1) with an empty middle, that's a strong visual sign of widespread saturation. Result here:
a moderate uptick at both ends (~255 and ~225 count in the outermost bins) but the middle
region (-0.5 to 0.5) is still meaningfully populated (~85-130 count per bin) — a "partially
improved, not perfectly healthy" picture. Matches expectations: Batch Norm greatly reduces
saturation but doesn't eliminate it entirely.

**Gradient histogram** — plot the distribution of `h.grad` (gradient flowing back into the
hidden layer) after a backward pass:

```python
h.retain_grad()   # h is a non-leaf tensor (output of tanh), so its .grad isn't kept by
                    # default — needed this call (inside the forward pass, right after
                    # h = torch.tanh(hpreact)) before .grad becomes accessible
plt.hist(h.grad.view(-1).tolist(), 50)
```

Result: a smooth, symmetric bell-shaped distribution centered at 0, with no sharp "needle"
spike exactly at zero. This is the healthy pattern — a sharp needle at 0 would indicate most
neurons carry essentially no gradient signal (the saturated-neuron failure mode from section
4). The smooth spread here confirms most neurons still carry meaningful, usable gradient.

**Together, these two plots give a before/after-style health check:** activations show mild
residual saturation, but gradients show the network is still learning effectively overall —
consistent with Batch Norm being a large but not total fix.

## 13. Viz #3 — parameter gradient statistics

Instead of looking at neuron *output* statistics (h and h.grad), looked at the *weights
themselves* — specifically, each 2D weight matrix's gradient distribution and its
"grad:data ratio" (gradient's std relative to the parameter's own std):

```python
for i, p in enumerate(parameters):
    t = p.grad
    if p.ndim == 2:   # only look at weight matrices (C, W1, W2), not 1D biases
        print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (
            tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
```

Result (end of training):
```
C  (27,10)  grad:data ratio 0.0135
W1 (30,200) grad:data ratio 0.0352
W2 (200,27) grad:data ratio 0.1354   ← notably higher than the others
bngain (1,200) 0.0612
bnbias (1,200) 0.1098
```

**What this ratio means:** how large the update to a parameter is, relative to the
parameter's own current magnitude. `W2` at 0.1354 means its gradient-driven update at this
step is roughly 13.5% of its own scale — much more aggressive than `C`'s 1.35%. Different
parameter types (embeddings, weight matrices, batchnorm scale/shift) naturally have
different gradient "personalities" — not inherently wrong, but worth monitoring, since a
wildly disproportionate update rate for one layer can destabilize training. This motivates
tracking the ratio *over time*, not just at one snapshot (next section).

## 14. Viz #4 — update:data ratio over the whole training run

Modified the training loop to record, at every iteration, each parameter's
`(lr * grad).std() / p.data.std()` on a log10 scale:

```python
ud = []
for i in range(max_steps):
    ... forward + backward as usual ...
    lr = 0.1 if i < 15000 else 0.01
    with torch.no_grad():
        for p in parameters:
            p.data += -lr * p.grad
        ud.append([((lr * p.grad).std() / p.data.std()).log10().item() for p in parameters])
```

Plotted each 2D parameter's ratio over all 30,000 iterations, with a reference line at
`log10(ratio) = -3` (i.e. ratio ≈ 1/1000), which the video suggests as a healthy target.

**Observations:**
- For the first 15,000 iterations (`lr=0.1`), most parameters sat somewhat above the -3
  reference line (roughly -1.9 to -3.0) — some (like `W2`, embedding `C`) more aggressively
  updated than others.
- At iteration 15000, when `lr` drops from `0.1` to `0.01` (a 10x reduction), every
  parameter's ratio visibly drops by about 1 unit on the log scale (i.e. ~10x smaller
  update) — expected, since the ratio is directly proportional to `lr`.
- After the drop, `C` (embedding) settled around `-4.0`, noticeably *below* the healthy
  reference line — suggesting the embedding layer may have been updating too slowly in the
  second half of training, essentially near-frozen relative to what might be ideal.

**Practical takeaway:** a single global learning rate, even with decay, doesn't treat every
parameter equally — some layers can end up updating too aggressively, others too slowly,
at the same shared `lr`. This is part of the motivation for more advanced optimizers (e.g.
Adam, flagged in the video as a later topic) that adapt the effective learning rate
per-parameter automatically, rather than relying on one global schedule for everything.

## 15. PyTorch-ifying — using built-in layer classes

Rewrote the same architecture using PyTorch's own layer classes instead of manually
managed tensors — closer to how real PyTorch code is typically written.

**Core idea: each `nn.` class is a "box" that stores its own parameters internally and
knows how to apply its own formula when called.**

```python
layer0 = torch.nn.Linear(30, 200, bias=False)
```
creates and stores a weight tensor (`layer0.weight`, shape `(200,30)`) internally — no need
to manually create/track `W1` as a separate variable. Calling `layer0(embcat)` then
automatically runs `embcat @ layer0.weight.T` (+bias if enabled) internally — the same
computation as manually writing `embcat @ W1 + b1`, just packaged inside the layer object.

```python
layers = [
    torch.nn.Linear(n_embd * block_size, n_hidden, bias=False),   # replaces W1, b1
    torch.nn.BatchNorm1d(n_hidden),                                 # replaces bngain, bnbias, mean/std normalize
    torch.nn.Tanh(),                                                 # replaces torch.tanh(...)
    torch.nn.Linear(n_hidden, vocab_size),                           # replaces W2, b2
]

with torch.no_grad():
    layers[-1].weight *= 0.1   # keep the final layer small at init, same reasoning as before

parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True
```

**Why 4 layers specifically — each has a distinct role:**
- First `Linear` (30→200): transforms the raw flattened embedding into a rich, learnable
  hidden representation ("thinking space").
- `BatchNorm1d`: forces that layer's pre-activation output into a healthy range every
  forward pass, preventing saturation.
- `Tanh`: the nonlinearity — without it, stacking two `Linear` layers would collapse
  mathematically into one single linear layer, making depth meaningless.
- Final `Linear` (200→27): compresses the rich hidden representation down to one score per
  possible next character — the "final decision" step.

**Layer parameter signatures reflect what each layer type needs to know:**
- `Linear(in_features, out_features)` needs both, since it changes dimensionality (must
  build a weight matrix of the right shape).
- `BatchNorm1d(num_features)` needs only one number, since it doesn't change dimensionality
  (normalizes in place, same size in and out) — it just needs to know how many per-feature
  gamma/beta pairs to track.
- `Tanh()` needs nothing — it's a fixed mathematical function with no learnable parameters.

**Forward pass collapses from 4 manually-written formula lines into one loop:**

```python
# before (manual):
hpreact = embcat @ W1 + b1
hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
h = torch.tanh(hpreact)
logits = h @ W2 + b2

# after (layer classes):
x = embcat
for layer in layers:
    x = layer(x)
```

Each layer already "knows" its own formula, so the loop just feeds the previous layer's
output into the next layer, layer by layer — mathematically identical result, but the code
no longer grows linearly with the number of layers (adding a 10-layer network wouldn't need
10 hand-written formula lines, just 10 entries in the `layers` list).

Confirmed correctness: initial loss on a random minibatch was `3.2871`, matching the
`~3.296` theoretical baseline (same as the manual version).

**Training loop is otherwise unchanged** — same minibatch sampling, same
zero-grad/backward/update steps, same learning rate decay schedule (`0.1` → `0.01` at
iteration 15000). Only the forward-pass section differs.

```python
max_steps = 30000
for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 15000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
```

Result: loss `3.2662` (iter 0) down to the `~1.9-2.3` range by iter 27000 — consistent with
the manually-written Batch Norm version, as expected (same math, different code
organization).

## 16. The fully-linear case — removing Tanh entirely

Experiment: remove `Tanh()` from the layer stack, keeping only
`Linear → BatchNorm1d → Linear`, and train the same way.

```python
layers = [
    torch.nn.Linear(n_embd * block_size, n_hidden, bias=False),
    torch.nn.BatchNorm1d(n_hidden),
    # Tanh removed
    torch.nn.Linear(n_hidden, vocab_size),
]
```

**Prediction based on earlier theory:** stacking two `Linear` layers with nothing
nonlinear between them should mathematically collapse into one single linear
transformation — depth should add no expressive power.

**Result:** loss settled in the `~2.1-2.5` range over 30,000 iterations — worse than the
`Tanh`-included version (`~1.9-2.3`), but *not* a total collapse into bigram-only
performance either. Notably close to bigram's converged score (`~2.45`), which makes sense:
without a real nonlinearity, the model can only learn something closer to a simple,
near-linear relationship, similar in character to bigram's simple statistical relationship.

**Why doesn't it collapse completely, given the "stacked linears = one linear" theory?**
Because `BatchNorm1d` isn't purely linear itself: its normalization divides by the current
minibatch's standard deviation, which changes from batch to batch — this data-dependent
division introduces a mild nonlinear element into the `Linear → BatchNorm1d → Linear` chain,
so it doesn't fully degenerate into a single linear map the way two bare `Linear` layers
back-to-back would. The network still learns *something*, just far less expressively than
with a real nonlinearity like `Tanh` providing genuine bends/kinks in the function it can
represent.

**Takeaway:** removing the nonlinearity doesn't break training outright (thanks to
BatchNorm's incidental nonlinear behavior), but it clearly caps how much the network can
learn — concretely demonstrating, with real numbers, why nonlinearities are necessary for
depth to actually pay off (not just a theoretical claim from earlier sections).

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