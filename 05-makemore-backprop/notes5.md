# 05-makemore-backprop — Notes

Learning log for manually backpropagating through a 2-layer MLP (with BatchNorm), following
Andrej Karpathy's makemore series (video 4 — "Becoming a Backprop Ninja"). Goal: replace
`loss.backward()` with hand-written, tensor-level backward passes for every operation in the
compute graph, verified against PyTorch's real autograd at each step.

## 1. Why this matters — scalar backprop vs. tensor/matrix backprop

01-micrograd did backprop on individual scalars (`Value` objects) — simple rules like
`a.grad += b.data * out.grad`. Real models operate on matrices (e.g. `C = A @ B` where `A` is
`(32,30)` and `B` is `(30,200)`), and naively trying to reuse the scalar rule breaks
immediately: shapes don't match (`B`'s shape `(30,200)` can't multiply directly against
`C.grad`'s shape `(32,200)`).

**The correct matrix chain rule for `C = A @ B`:**
```
A.grad = C.grad @ B.T
B.grad = A.T @ C.grad
```
Verified by shape-checking: `C.grad (32,200) @ B.T (200,30) = (32,30)` — matches `A`'s shape.
`A.T (30,32) @ C.grad (32,200) = (30,200)` — matches `B`'s shape. **Getting the transpose and
multiplication order right, largely by checking that shapes come out consistent, is the core
new skill in matrix-level backprop** — beyond just knowing the calculus rule.

## 2. Starter code

Same dataset pipeline as previous notebooks (build_dataset, train/dev/test split).

Model uses a smaller hidden layer than 04 (`n_hidden=64` instead of 200) — practical choice
for faster iteration while manually deriving/checking each backward step. Biases and
BatchNorm gain/bias are initialized to small random values (not exactly zero or one) so that
later gradient comparisons against PyTorch don't accidentally mask bugs that a
trivial/degenerate initial value might hide.

## 3. Forward pass, fully "unrolled" into atomic operations

Instead of calling `F.cross_entropy` and letting BatchNorm's internals stay hidden, wrote
every intermediate step as its own named variable — necessary so each one can be
individually backpropagated and checked:

```python
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)

hprebn = embcat @ W1 + b1                              # linear 1

bnmeani = 1/n * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True)          # note: n-1, not n — Bessel's correction
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias                        # batchnorm, fully unrolled

h = torch.tanh(hpreact)                                   # nonlinearity

logits = h @ W2 + b2                                       # linear 2

logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes                          # numerical stability trick:
                                                              # subtracting the row max before
                                                              # exp() prevents overflow; doesn't
                                                              # change softmax's result since
                                                              # this shift cancels out during
                                                              # normalization
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum**-1                              # inverse (not direct division) —
                                                              # slightly easier to differentiate
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()
```

Initial loss ≈ 3.43 — close to the theoretical `~3.3` baseline, confirms the architecture is
set up correctly before starting on backward.

## 4. Verification method — `cmp()`

The key tool for this whole exercise: after manually computing a gradient, compare it
against PyTorch's own (via `retain_grad()` on every intermediate tensor + a real
`loss.backward()` call) to get a definitive right/wrong answer, rather than guessing.

```python
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')
```

Note: after calling `loss.backward()` once, the graph is consumed — need to re-run the
forward pass cell before calling backward again if re-checking.

## 5. dlogprobs — first gradient of the chain

`loss = -logprobs[range(n), Yb].mean()` only uses one specific cell per row of `logprobs`
(the true target's log-probability) — every other cell in the `(n, vocab_size)` matrix is
unused, so its gradient must be exactly 0.

Worked through a tiny 2-example, 3-class numeric example by hand: only
`logprobs[0][target_0]` and `logprobs[1][target_1]` appear in the loss formula, each
contributing `-1/n` to the gradient (from the sum-then-divide-by-n mean, plus the negative
sign in front).

```python
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
```

`cmp('logprobs', dlogprobs, logprobs)` → **exact: True, maxdiff: 0.0**.

## 6. dprobs

`logprobs = probs.log()`. Local derivative of `ln(x)` is `1/x`, so by chain rule:

```python
dprobs = (1.0 / probs) * dlogprobs
```

Shapes match exactly (`probs` and `dlogprobs` both `(n, vocab_size)`), so this is a
straightforward element-wise multiply, no broadcasting subtlety here.

`cmp('probs', dprobs, probs)` → **exact: True, maxdiff: 0.0**.

## 7. dcounts_sum_inv — first broadcasting gradient, and why sum is needed

`probs = counts * counts_sum_inv`, where `counts` is `(n, vocab_size)` but
`counts_sum_inv` is `(n, 1)` — broadcasting means each row's single `counts_sum_inv` value
gets implicitly reused across all `vocab_size` columns of that row when computing `probs`.

**Two distinct steps combine here, not a contradiction:**
1. **Multiply** (ordinary chain rule, local derivative × incoming gradient), applied
   per-cell: `contribution[j] = counts[i][j] * dprobs[i][j]` for each column `j`.
2. **Sum** those `vocab_size` per-cell contributions together, because they all trace back
   to the *same single* `counts_sum_inv[i]` value being reused (same accumulation logic as
   01-micrograd's `b = a + a` case — a value used in multiple places gets its gradient
   contributions added together).

Worked a small numeric example (`counts[0]=[2,5,3]`, `counts_sum_inv[0]=[0.1]`,
`dprobs[0]=[0.4,0.1,0.2]`): per-column products `[0.8, 0.5, 0.6]`, summed to `1.9`.

```python
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
```

Shape check: `(counts * dprobs)` is `(n, vocab_size)`; `.sum(1, keepdim=True)` collapses to
`(n, 1)` — matches `counts_sum_inv`'s own shape.

`cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)` → **exact: True, maxdiff: 0.0**.

## 9. dcounts_sum — power rule through the inverse

`counts_sum_inv = counts_sum**-1`. Worked through what `**-1` means concretely: it's just
`1/x` (verified: `5**-1 = 1/5 = 0.2`). Recognized `counts_sum**-1` as an alternative way of
writing `1/counts_sum`, chosen over direct division because differentiating a power/multiply
is more familiar than deriving division's rule separately.

Applied the power rule (`f(x)=x^n → f'(x)=n*x^(n-1)`) with `n=-1`:
```
d(counts_sum_inv)/d(counts_sum) = -1 * counts_sum^(-2) = -1/counts_sum²
```

```python
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
```
Both operands are `(n,1)`, so this is a simple element-wise multiply, no broadcasting
subtlety.

`cmp('counts_sum', dcounts_sum, counts_sum)` → **exact: True, maxdiff: 0.0**.

## 10. dcounts — combining a broadcasting-multiply gradient and a sum's gradient

`counts` is used in *two* places:
```python
counts_sum = counts.sum(1, keepdim=True)    # path 1
probs = counts * counts_sum_inv              # path 2
```
Just like `b = a + a` in 01-micrograd, using a value in two places means its total gradient
is the sum of the contributions from each path.

**Path 2 (multiply, already-familiar pattern):**
```python
contribution_mult = counts_sum_inv * dprobs
```

**Path 1 (sum's backward — "distributing" a gradient) required extra work to build
intuition for.** Key realization, worked through with a concrete tiny example
(`counts[0]=[7,2,1]`, `counts_sum[0]=[10]`): increasing *any one* of `7`, `2`, or `1` by 1
increases `counts_sum` by exactly 1 — all three terms affect the sum identically. So if
`dcounts_sum[0] = 0.5` (nudging `counts_sum` by 1 changes loss by 0.5), then nudging *any*
of the three original values by 1 has that same 0.5 effect on loss — each one independently
"deserves" the full 0.5, not a fraction of it. This is the backward mirror of forward's
"squeeze many values into one via sum": backward "distributes one gradient value out to
many," each original position getting an identical copy of the incoming gradient.

```python
contribution_sum = torch.ones_like(counts) * dcounts_sum
```

**What `torch.ones_like(counts)` is actually doing, and why it's necessary (not just a
readability choice):** `dcounts_sum` has shape `(n,1)` — one number per row — but the
gradient contribution needed for `counts` must have shape `(n, vocab_size)` — one number per
*column* too. A bare scalar `1` multiplied against `dcounts_sum` leaves it at `(n,1)`,
still the wrong shape. `torch.ones_like(counts)` creates a template tensor already shaped
`(n, vocab_size)`, filled with 1s; multiplying it by `dcounts_sum` triggers broadcasting,
which copies each row's single `dcounts_sum` value across all `vocab_size` columns of that
row — this is literally *how* a `(n,1)` gradient gets turned into an `(n,vocab_size)`
gradient with identical values across each row (matching the "every term in the sum gets an
identical copy" reasoning above). Without the `ones_like` template, there's no multiply
operation to trigger the broadcast, and thus no mechanism to expand the shape.

**Combined (sum of both paths' contributions):**
```python
dcounts = counts_sum_inv * dprobs + torch.ones_like(counts) * dcounts_sum
cmp('counts', dcounts, counts)
```
→ **exact: True, maxdiff: 0.0**.

**On the `cmp()` function itself:** takes a manually-computed gradient (`dt`) and the real
PyTorch tensor (`t`, whose `.grad` was populated by an actual `loss.backward()` call via
`retain_grad()`), and reports whether they match exactly, approximately (allowing for tiny
floating-point differences), and the maximum absolute difference. This turns "did I derive
this gradient correctly" from a guess into a definitive, checkable answer at every single
step — essential for building confidence while working through a long manual derivation.

## 8. Open questions / next steps

- Continue the backward chain: `dcounts`, `dnorm_logits`, `dlogit_maxes`, `dlogits`, then
  through the second Linear layer, `tanh`, BatchNorm (unrolled, including the Bessel's
  correction detail flagged by the video), the first Linear layer, and finally back to the
  embedding table `C`.
- Video's "brief digression" on Bessel's correction (the `1/(n-1)` vs `1/n` in `bnvar`) not
  yet explored in depth — worth understanding why this specific correction is used in
  BatchNorm's variance calculation.
- Exercise 2 (video): derive cross-entropy's backward pass as one direct, closed-form
  expression (rather than the fully step-by-step unrolled version done here) and compare.
- Exercise 3 (video): same closed-form treatment for BatchNorm's backward pass.
- Exercise 4 (video): assemble the complete manual backward pass and confirm it matches
  `loss.backward()` end-to-end, then use it in an actual training loop without
  `loss.backward()` at all.