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

## 10. Open questions / next steps

- Full-batch training (all 228K examples per step) is slow — the video introduces
  minibatches (random subsets per step) for much faster iteration; try that next.
- Need a systematic way to pick a good learning rate (video: "finding a good initial
  learning rate") rather than guessing a fixed value like `0.1`.
- Haven't split data into train/val/test yet — needed to check the model is actually
  generalizing rather than overfitting the training set.
- Try a larger hidden layer and larger embedding size (video experiments with both) and see
  how loss responds.
- Visualize the learned 2D character embeddings once trained — check whether
  similar-behaving characters (e.g. vowels) cluster together, as theorized in section 3.
- Sample new names from the trained model and compare against bigram's output quality.