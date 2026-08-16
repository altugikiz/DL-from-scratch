# 03-makemore-mlp — Notes

Learning log for building an MLP-based character-level language model, following Andrej
Karpathy's makemore series (video 2, based on Bengio et al. 2003).

## 1. Why go beyond bigram

Bigram only looks 1 character back — a fundamental ceiling on what it can learn. Extending
context via counting doesn't scale: trigram needs 27³ cells, 4-gram 27⁴, and it explodes
from there. An MLP-based approach instead *learns a function* from a wider context window to
a next-character prediction, without needing to enumerate every possible context combo —
this scales to much longer contexts.

## 2. Building multi-character context windows

Using `block_size = 3` (3 characters of context per prediction). Each word is padded with
leading `.` tokens and slides a window across it:

```python
block_size = 3
X, Y = [], []
for w in words:
    context = [0] * block_size   # start with [0,0,0] = three '.' tokens
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]   # slide the window: drop oldest, append newest
X = torch.tensor(X)   # shape (228146, 3)
Y = torch.tensor(Y)   # shape (228146,)
```

For `emma`: `... -> e`, `..e -> m`, `.em -> m`, `emm -> a`, `mma -> .` — same total example
count (228,146) as the bigram version, since context length doesn't change how many
character transitions exist in the dataset, just how much history each one carries.

## 3. Embeddings — a lookup table instead of one-hot

**Why not reuse one-hot + matrix multiply from the bigram model?** One-hot works but is
wasteful: multiplying a mostly-zero vector against a weight matrix is mathematically
equivalent to just selecting one row of that matrix — so skip the wasted computation and do
the row-selection directly. Also, one-hot forces every character into a 27-dimensional,
mutually-equidistant representation, with no way to express that some characters behave
similarly. A smaller, dense, *learned* embedding lets similar-behaving characters end up
close together in vector space.

```python
C = torch.randn((27, 2))   # 27 rows (one per character), 2-dim embedding per row (learnable)
```

**How do embeddings end up meaningfully placed, if training never sees "meaning"?**
Gradient descent has no concept of linguistic meaning — it only follows "reduce the loss."
But if two characters (e.g. 'a' and 'e') tend to appear in similar contexts across the
dataset, the loss-minimizing solution often involves placing their embeddings close
together, because that lets the rest of the network reuse similar internal representations
for both — reducing loss more than keeping them arbitrarily far apart would. The
"similarity" that emerges is a side effect of optimization, not something explicitly coded.
Confirmed later (per video) by visualizing the learned 2D embeddings and seeing vowels
cluster.

**Fancy indexing — `C[X]`:** given a lookup table `C` and an index tensor `X` of any shape,
`C[X]` replaces every individual index in `X` with its corresponding row from `C`, keeping
`X`'s original shape and adding the embedding dimension on top:

```python
emb = C[X]   # X is (228146, 3) of indices -> emb is (228146, 3, 2) of embeddings
```

Concretely: `X`'s shape is preserved structurally, but each scalar index "expands" into its
2-number embedding vector in place — no explicit loop needed, done as a single fast
operation.

## 4. Flattening embeddings for the hidden layer with `.view()`

A hidden-layer neuron expects a flat vector input, but `emb` is `(228146, 3, 2)` — need each
example's 3 characters' embeddings concatenated into one 6-length vector per example.

```python
emb_flat = emb.view(-1, 6)   # (228146, 3, 2) -> (228146, 6)
```

`-1` tells PyTorch "figure out this dimension yourself" — it computes it from total element
count ÷ the dimensions given explicitly (here, `total / 6`). Safer than hardcoding the exact
row count (`228146`), which would silently break if the dataset size ever changed. Only one
`-1` allowed per `.view()` call, since PyTorch needs all-but-one dimension fixed to solve for
the remaining one.

## 5. Hidden layer and output layer

```python
W1 = torch.randn((6, 100))   # 6 = flattened embedding size, 100 = hidden layer size (a
b1 = torch.randn(100)         # chosen hyperparameter, like MLP's [4,4] layer sizes earlier —
                                # no formula for it, just a "reasonable" starting choice;
                                # too few neurons -> underfitting, too many -> overfitting risk
h = torch.tanh(emb_flat @ W1 + b1)   # (228146, 100)

W2 = torch.randn((100, 27))   # 100 = hidden layer size, 27 = one logit per character
b2 = torch.randn(27)
logits = h @ W2 + b2           # (228146, 27) — raw scores, not yet probabilities
```

Full architecture summary:
```
X (228146,3) --C[X]--> emb (228146,3,2) --view--> emb_flat (228146,6)
--tanh(@W1+b1)--> h (228146,100) --@W2+b2--> logits (228146,27)
```

## 6. `F.cross_entropy` — the built-in version of manual NLL

```python
loss = F.cross_entropy(logits, Y)
```

Does in one call what was hand-written for the bigram model: softmax (exp + normalize) →
pick out the probability assigned to the true target → negative log → average. Takes raw
logits directly (no need to manually softmax first).

## 7. Weight initialization matters — E02 exercise

Naive `torch.randn` initialization of `W2`/`b2` gave an initial loss of **17.9455** — far
worse than the theoretical "uniform guessing" loss of `-log(1/27) ≈ 3.296`. Root cause:
stacking `randn`-scale weights across layers (embedding → hidden → output) let `logits`
drift to extreme values, which softmax turns into overconfident (near-0/near-1)
probabilities — very costly when an overconfident guess is wrong.

Fix: scale down `W2` and `b2` at initialization (`* 0.01`), so initial `logits` start small
and close together, giving a near-uniform initial probability distribution:

```python
W2 = torch.randn((100, 27)) * 0.01
b2 = torch.randn(27) * 0.01
```

Result: initial loss dropped to **3.3056**, very close to the theoretical `~3.296`. This
matters beyond cosmetics — a bad initialization wastes early training iterations just
"undoing" the bad start rather than making real progress.

**Bug hit along the way:** writing `torch.randn(..., requires_grad=True) * 0.01` makes the
`* 0.01` result a non-leaf tensor — its `.grad` is never populated during backward (silent
`None`), causing `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'` at
the update step. Fix: apply `requires_grad_(True)` *after* the scaling operation, e.g.
`(torch.randn(...) * 0.01).requires_grad_(True)`, so the scaled tensor itself becomes the
tracked leaf parameter.

## 8. `keepdims=True`, revisited

When a reduction (like `.sum()`) collapses a dimension, `keepdims=True` keeps that dimension
present as size 1 instead of removing it entirely (`(4,27).sum(1)` → `(4,)` without
keepdims, vs `(4,1)` with it). Keeping it as `(4,1)` lets subsequent division/multiplication
broadcast correctly against the original `(4,27)` shape; dropping it to `(4,)` can cause
shape-mismatch errors or, in some cases, silently incorrect broadcasting.

## 9. First training run

```python
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((27, 2), generator=g, requires_grad=True)
W1 = torch.randn((6, 100), generator=g, requires_grad=True)
b1 = torch.randn(100, generator=g, requires_grad=True)
W2 = (torch.randn((100, 27), generator=g) * 0.01).requires_grad_(True)
b2 = (torch.randn(27, generator=g) * 0.01).requires_grad_(True)
parameters = [C, W1, b1, W2, b2]   # 3481 total learnable scalars
```

Training loop (full-batch, all 228,146 examples every iteration — same shape as
01-micrograd's loop, just PyTorch-native):

```python
for k in range(200):
    emb = C[X]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)

    for p in parameters:
        p.grad = None
    loss.backward()

    for p in parameters:
        p.data += -0.1 * p.grad
```

Result: loss `3.3032` (iter 0) → `2.6387` (iter 180). Approaching but not yet at the
bigram model's converged score (`2.4541`) — expected, since only 200 iterations were run and
this is full-batch (slow per step). The 3-character-context MLP has strictly more
information available than bigram's 1-character context, so it should be able to surpass
bigram's score with more training — that's the next thing to verify.

## 11. Minibatch training — much faster iteration

Full-batch (all 228,146 examples per step) is accurate but slow. Switched to sampling a
small random subset ("minibatch") of 32 examples per step:

```python
ix = torch.randint(0, X.shape[0], (32,))
emb = C[X[ix]]
...
loss = F.cross_entropy(logits, Y[ix])
```

Each step now processes far less data, so many more iterations fit in the same time.
Trade-off: the per-step loss becomes noisy (each minibatch is a different, small, random
sample, so easy vs. hard minibatches produce very different loss values step to step) —
the overall downward trend is what matters, not any single printed value. Ran 10,000
iterations at `lr=0.1`; loss fluctuated between ~2.1 and ~2.8, already surpassing bigram's
converged score (2.4541) at its better points.

## 12. Finding a good learning rate systematically

Rather than guessing a fixed learning rate, swept a wide logarithmic range and plotted loss
against it:

```python
lre = torch.linspace(-3, 0, 1000)   # exponents from -3 to 0
lrs = 10**lre                        # learning rates from 0.001 to 1.0

lossi, lri = [], []
for k in range(1000):
    ix = torch.randint(0, X.shape[0], (32,))
    ... # forward + backward as usual
    lr = lrs[k]
    for p in parameters:
        p.data += -lr * p.grad
    lri.append(lre[k].item())
    lossi.append(loss.item())

plt.plot(lri, lossi)
```

Used a logarithmic sweep (exponents, not raw values) because learning rate's effect isn't
linear — the difference between 0.001 and 0.01 matters much more than between 0.5 and 0.51,
so a log scale gives small values fair representation.

Plot showed a "bowl" shape: loss stays in a reasonable ~2.0-2.5 band from `lr≈0.001` up
through `lr≈0.3`, then shoots up sharply (to 4+ and unstable) as `lr` approaches 1.0 — too
large a step overshoots and destabilizes training. Confirmed `lr≈0.1` (the value already
being used) sits in a reasonable part of that range.

## 13. Train / validation / test split

**Why needed:** looking only at loss on the data the model was trained on can be
misleading — a model can overfit (memorize training examples) and look great on that data
while generalizing poorly to anything new. A held-out validation set the model never trains
on reveals whether it actually learned generalizable patterns.

```python
def build_dataset(words):
    block_size = 3
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
words_shuffled = words.copy()
random.shuffle(words_shuffled)   # shuffle first — avoids a biased split if names.txt has
                                    # any ordering (e.g. alphabetical)

n1 = int(0.8 * len(words_shuffled))   # cutoff index for 80%
n2 = int(0.9 * len(words_shuffled))   # cutoff index for 90%

Xtr,  Ytr  = build_dataset(words_shuffled[:n1])    # 80% — train: updates model weights
Xdev, Ydev = build_dataset(words_shuffled[n1:n2])   # 10% — dev/validation: monitor only,
                                                       # used for tuning hyperparameters
Xte,  Yte  = build_dataset(words_shuffled[n2:])      # 10% — test: touched once, at the end
```

`int(0.8 * len(...))` truncates a fractional index count (`25626.4` → `25626`) down to a
valid integer list index. Verified split sizes: `(182625, 3)`, `(22655, 3)`, `(22866, 3)`
— summing back to 228,146.

**Bug hit along the way:** typo'd `parameteres` (extra/misplaced letters) instead of
`parameters` when building the parameter list — the training loop's `for p in parameters`
then silently referenced a *different, stale* `parameters` variable from an earlier cell
run, producing the same `NoneType` gradient error as before but for a completely different
reason. Lesson: this class of bug (subtly misspelled variable name shadowing/missing the
intended one) can look identical to other bugs in the error message, so check spelling
carefully, especially after a Restart & Run All didn't fix an already-diagnosed issue.

## 14. Why the hidden layer is *wider* than the input, then narrows to the output

Input (6 dims) is just raw data — the 3 characters' embeddings side by side, carrying no
processed meaning yet. The hidden layer (100 dims) is where the network actually
"thinks" — more neurons means more capacity to simultaneously detect different patterns in
that raw input (e.g., one neuron might end up sensitive to "does this end in a vowel",
another to some statistically useful but human-indescribable pattern, etc.). A hidden layer
the same size as the input would only allow a limited transformation of the data; going
wider lets the network build a much richer internal representation before making a
decision. The output layer (27 dims) then compresses that rich internal representation back
down to exactly the number of things being decided between (27 possible next characters) —
the "final verdict" step. Hidden layer width is a design choice / hyperparameter (not
determined by input or output size) that trades off capacity: too narrow risks
underfitting, too wide risks overfitting.

## 15. Training with a proper split

Retrained from scratch, sampling minibatches only from `Xtr`/`Ytr` (never touching
`Xdev`/`Ydev` during weight updates), 30,000 iterations at `lr=0.1`. Then measured real
(non-minibatch) loss on the full train set and the untouched dev set:

```
train loss: 2.3136
dev loss:   2.3201
```

The two are very close (diff ≈ 0.0065) — strong evidence the model is genuinely learning
generalizable structure, not memorizing the training set (memorization would show up as a
much lower train loss than dev loss). Both numbers also beat bigram's `2.4541`, confirming
the 3-character-context MLP does capture more useful structure than bigram's 1-character
context, as hypothesized.

## 17. Sampling from the trained MLP

Same overall sampling loop as bigram (start with an empty context, repeatedly sample the
next character, stop at `.`), but now the context is a sliding 3-character window run
through the trained embedding → hidden layer → softmax pipeline each step:

```python
g = torch.Generator().manual_seed(2147483647)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))
```

**Why `tanh` on `h` but not on `logits`:** `tanh` (a nonlinearity) is needed on the hidden
layer so stacking layers actually adds expressive power — without it, layered linear
operations collapse back into one linear function. The final layer producing `logits`
deliberately skips this: it's about to go through `softmax` (built on `exp`), which does its
own nonlinear transform, and squashing logits into `(-1,1)` with `tanh` beforehand would
dull the score differences softmax relies on to sharply separate character probabilities.
General pattern: nonlinearities go between layers to enable depth; the final output layer is
typically left "raw," feeding whatever transform (softmax, sigmoid, etc.) is appropriate for
the task.

## 18. Comparing generated names: bigram vs. MLP

Bigram (1-character context) samples: `cexze, momasurailezitynn, konimittain, llayn, ka,
da, staiyaubrtthrigotai, moliellavo, ke, teda`

MLP (3-character context, trained) samples: `dex, daidaller, ile, kayde, dinichana, nylla,
kyn, tar, samiyah, jansi, iota, mic, jen, kaugk, treda, kamerle, sadey, niaviyny, fols,
milli`

Clear qualitative jump: MLP outputs are consistently reasonable name-lengths (bigram
produced some absurdly long/malformed ones like `momasurailezitynn`), and several MLP
outputs (`samiyah`, `jansi`, `milli`, `treda`, `iota`) look strikingly close to real names.
Some outputs are still awkward (`kaugk`, `niaviyny`) — expected, since this is still only a
3-character context with a modest 100-unit hidden layer. The improvement directly
demonstrates the thesis from section 1: more context lets the model capture a word's overall
shape (length, syllable structure) rather than just very local, 1-character-back
transitions — the same underlying reason production language models use much longer
context windows.

## 19. Status: `03-makemore-mlp` complete

Built, end to end: multi-character context windows (sliding 3-char window) → learned
character embeddings (replacing one-hot) → hidden layer with `tanh` → output layer →
`F.cross_entropy` loss → fixed a bad weight-initialization bug (E02) → minibatch training →
systematic learning-rate search → proper train/val/test split confirming genuine
generalization (train 2.3136 vs dev 2.3201, both beating bigram's 2.4541) → sampled new
names and confirmed a clear qualitative improvement over the bigram model.

Remaining nice-to-haves (not blocking, can revisit later): visualize the learned 2D
embeddings (check whether similar characters, e.g. vowels, cluster), experiment with larger
hidden layer / embedding sizes (video does both), and only touch the held-out test set once
at the very end of any further tuning.

Next in the series: move beyond a fixed-size context window (currently hardcoded at 3)
toward architectures that handle variable-length context more naturally — RNN, then
Transformer.