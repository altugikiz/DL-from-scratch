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

## 11. dnorm_logits — exp()'s clean derivative

`counts = norm_logits.exp()`. Since `d(exp(x))/dx = exp(x)` (the function is its own
derivative), `counts` itself already *is* the local derivative:

```python
dnorm_logits = counts * dcounts
```
Shapes match exactly (`(n, vocab_size)` both), simple element-wise multiply.

`cmp('norm_logits', dnorm_logits, norm_logits)` → **exact: True, maxdiff: 0.0**.

## 12. Why `logit_maxes` exists — numerical stability, not a training-related choice

Before diving into `dlogit_maxes`, clarified *why* `logit_maxes` is computed at all:
```python
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
```
This has nothing to do with what the model learns — it's purely a numerical safety trick.
`exp()` of a large logit (e.g. `exp(500)`) can overflow to `inf`. Subtracting each row's max
from every value in that row before calling `exp()` doesn't change softmax's final result
(the shift cancels out during normalization — verified with a concrete example:
`logits=[5,8,3]` gives identical final probabilities whether or not the max is subtracted
first), but guarantees every value going into `exp()` is ≤ 0, so it can never overflow.
Subtracting specifically the *max* (rather than any other value) guarantees this property.

## 13. dlogit_maxes — subtraction's sign + broadcasting's sum, combined

`norm_logits = logits - logit_maxes`. Worked through subtraction's local derivatives with a
tiny numeric check: for `y = x - z`, nudging `x` up by 1 nudges `y` up by 1 (`local deriv =
+1`), but nudging `z` up by 1 nudges `y` *down* by 1 (`local deriv = -1`).

So `logits` gets `+1 * dnorm_logits` (unchanged sign, just copied), while `logit_maxes` gets
`-1 * dnorm_logits` (sign flipped) — but `logit_maxes`'s shape `(n,1)` doesn't match
`dnorm_logits`'s shape `(n, vocab_size)`, because `logit_maxes` was broadcast during the
subtraction (each row's single max value was reused across all `vocab_size` columns when
computing `norm_logits`). Same broadcasting-sum pattern as `dcounts_sum` earlier: since one
value fed into many outputs, its gradient contributions from every one of those outputs must
be summed.

```python
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
```

`cmp('logit_maxes', dlogit_maxes, logit_maxes)` → **exact: True, maxdiff: 0.0**.

## 14. dlogits (complete) — combining a copy-through and a max()-selection gradient

`logits` is used in two places:
```python
norm_logits = logits - logit_maxes           # path 1, local deriv +1, already have this
logit_maxes = logits.max(1, keepdim=True).values   # path 2, needs max()'s backward rule
```

**Path 2 required understanding `max()`'s backward rule from scratch, worked through with
a concrete numeric experiment (not just asserted).** For `logits[0] = [5, 8, 3]`, max is 8
(position 1). Nudging the non-max value `5` up by a tiny amount (`5.0001`) doesn't change
the max at all (still 8) — so `5`'s local gradient contribution to the max is 0. Nudging
the max value `8` up by a tiny amount (`8.0001`) changes the max by that *exact same* amount
— so `8`'s local gradient contribution is 1. Conclusion: **only the position that was
actually selected as the max carries any gradient; every other position gets exactly 0** —
not because the max is special in some abstract sense, but because it's literally the only
value that was used in computing the output; nudging an unused value can't affect a result
that never depended on it.

```python
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
```

- `logits.max(1).indices` gives, per row, which column position held the max.
- `F.one_hot(..., num_classes=logits.shape[1])` turns that index into a one-hot vector
  (e.g. index 1 in a 3-wide row → `[0,1,0]`) — matching exactly the gradient pattern derived
  by hand above.
- Multiplying by `dlogit_maxes` places the real gradient value only at the max position,
  zero elsewhere.
- `+=` (not `=`) accumulates this onto the path-1 contribution already computed
  (`dlogits = dnorm_logits.clone()`), since `logits` is used in two places.

`logits.shape[1]` (rather than hardcoding `27`) reads the actual column count dynamically —
same "don't hardcode a size that might change" reasoning as using `-1` in `.view()` or a
`vocab_size` variable elsewhere.

`cmp('logits', dlogits, logits)` → **exact: True, maxdiff: 0.0**.

## 15. Second Linear layer backward — dh, dW2, db2

`logits = h @ W2 + b2`. Applying the general matrix-multiply backward rule derived earlier
(`C = A @ B` → `A.grad = C.grad @ B.T`, `B.grad = A.T @ C.grad`), with the mapping
`A=h, B=W2, C=logits`:

```python
dh = dlogits @ W2.T
dW2 = h.T @ dlogits
```

Shape check (always worth doing explicitly): `h (32,64)`, `W2 (64,27)`, `dlogits (32,27)`.
`dh = dlogits(32,27) @ W2.T(27,64) = (32,64)` ✓ matches `h`. `dW2 = h.T(64,32) @
dlogits(32,27) = (64,27)` ✓ matches `W2`.

**`.T` in code is just PyTorch's transpose shorthand** — swaps rows and columns (e.g. a
`(2,3)` matrix's `.T` becomes `(3,2)`, same numbers, different row/column positions).
Transposing here is specifically what makes the multiplication's shapes line up — without
it, `dlogits @ W2` wouldn't even be a valid matrix multiply.

**Why compute `dh` at all, given `h` isn't a trainable parameter?** Two different reasons
for gradients in this exercise: `W2`/`b2`'s gradients are needed to actually *train* them
(`W2.data -= lr*W2.grad`); `h`'s gradient is needed purely to *keep walking backward* through
the chain — the next steps (`hpreact`, `bnraw`, etc.) all need `dh` as their starting
"incoming gradient," per the chain rule, even though `h` itself is never directly updated.

**`db2` — bias broadcasting, worked through with a concrete tiny example.** With `b2 =
[10, 20, 30]` (3 characters) added to 2 examples' rows, `b2[0]=10` gets added into *both*
example rows' first column (`logits[0][0]` and `logits[1][0]`) — used twice, once per
example. So `b2`'s gradient must sum the contributions from every row it was broadcast
into — concretely, "sum down each column across all rows":

```python
db2 = dlogits.sum(0)
```

`.sum(0)` collapses the row dimension: `dlogits (32,27) → (27,)`, matching `b2`'s own shape.

```python
cmp('h', dh, h)     → exact: True, maxdiff: 0.0
cmp('W2', dW2, W2)   → exact: True, maxdiff: 0.0
cmp('b2', db2, b2)   → exact: True, maxdiff: 0.0
```

All three verified exactly — completes the backward pass through the second Linear layer.

## 16. tanh backward — the familiar micrograd formula, now on matrices

`h = torch.tanh(hpreact)`. Same derivative rule as 01-micrograd (`d(tanh(x))/dx =
1-tanh(x)^2`), expressed in terms of `h` itself (since `h` already equals `tanh(hpreact)`):

```python
dhpreact = (1.0 - h**2) * dh
```

Shapes match (`h` and `dh` both `(32,64)`), simple element-wise multiply.

`cmp('hpreact', dhpreact, hpreact)` → **exact: True, maxdiff: 0.0**.

## 17. BatchNorm backward, step 1 — dbngain, dbnraw, dbnbias

Entering the multi-step BatchNorm backward chain — forward was written fully "unrolled"
(section 3), so backward has to walk back through each of its 7 sub-steps individually.

**First formula:** `hpreact = bngain * bnraw + bnbias`. Three separate inputs, each with a
different role and shape:
- `bngain` (1,64) — learnable BatchNorm scale (gamma), broadcast across all 32 rows.
- `bnraw` (32,64) — the normalized (mean-0, std-1) values, not itself a trainable parameter
  but needed as a stepping stone to continue backward further (toward `bndiff`, `bnvar_inv`).
- `bnbias` (1,64) — learnable BatchNorm shift (beta), also broadcast across all 32 rows.

**Worked through why `bngain`'s gradient needs `.sum(0)`, with a small concrete example**
(`bngain=[2,5,3]` applied to 2 example rows): each element of `bngain` (e.g. the `2`) gets
reused once per row it's broadcast into (2 uses here, since there are 2 examples) — same
broadcasting-accumulation pattern as `b2` earlier. Since `bngain`'s shape has "1" in the row
dimension, summing over dimension 0 ("down each column, across all rows") collapses back to
that same `(1,64)` shape, matching what a valid gradient for `bngain` must look like.

```python
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)   # multiply (chain rule) + sum (broadcast accumulation)
dbnraw = bngain * dhpreact                             # no broadcast on this side — shapes already match
dbnbias = dhpreact.sum(0, keepdim=True)                # same broadcast-sum pattern as bngain, but for
                                                          # the addition case (like b2 earlier)
```

```python
cmp('bngain', dbngain, bngain)   → exact: True, maxdiff: 0.0
cmp('bnraw', dbnraw, bnraw)       → exact: True, maxdiff: 0.0
cmp('bnbias', dbnbias, bnbias)    → exact: True, maxdiff: 0.0
```

All three verified exactly. `bngain`/`bnbias`'s gradients will later drive their training
updates; `bnraw`'s gradient is needed purely to keep walking backward through the rest of
BatchNorm's unrolled steps (`bndiff`, `bnvar_inv`, `bnvar`, `bndiff2`, `bnmeani`, back to
`hprebn`).

## 18. BatchNorm backward, step 2 — dbnvar_inv, dbndiff (partial), dbnvar

**Formula:** `bnraw = bndiff * bnvar_inv`. Same broadcasting pattern as `counts *
counts_sum_inv` earlier: `bndiff` is `(32,64)`, `bnvar_inv` is `(1,64)` (one value per
neuron, broadcast across all 32 rows).

```python
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)   # broadcast side: multiply + sum
dbndiff = bnvar_inv * dbnraw                             # non-broadcast side: no extra sum needed here
                                                            # (this is only the PARTIAL contribution —
                                                            # bndiff is used in a second place too,
                                                            # to be added later)
```

`cmp('bnvar_inv', dbnvar_inv, bnvar_inv)` → **exact: True, maxdiff: 0.0**.

**Formula:** `bnvar_inv = (bnvar + 1e-5)**-0.5`. Power rule (`f(x)=x^n → f'(x)=n*x^(n-1)`)
with `n=-0.5`:
```
f'(x) = -0.5 * (x + 1e-5)^-1.5
```

```python
dbnvar = (-0.5 * (bnvar + 1e-5)**-1.5) * dbnvar_inv
```

Shapes match (`(1,64)` both), no broadcasting subtlety here.

`cmp('bnvar', dbnvar, bnvar)` → **exact: True, maxdiff: 0.0**.

**On the `1e-5` in the formula:** scientific notation for `0.00001` — an "epsilon," added
to prevent a divide-by-zero/undefined-gradient blowup if `bnvar` (a variance) ever came out
to exactly 0 for some neuron (e.g. if all its outputs across the batch happened to be
identical). `1/sqrt(0)` is undefined; `1/sqrt(0.00001)` is safe and large but well-defined.
Negligible effect on the actual computed value, purely a numerical-safety guard — same
general category of trick as the `logit_maxes` subtraction used earlier for `exp()`
stability.

## 19. BatchNorm backward, step 3 — dbndiff2, dbndiff (complete)

**Formula:** `bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True)`. Two things happening: a
constant multiplier (`1/(n-1)`, fixed since `n` is fixed) and a sum.

**Key insight about constant multipliers in chain rule, worked through with a small
example:** for `y = 3*x`, nudging `x` by 1 changes `y` by exactly 3 (verified:
`x=2→y=6`, `x=3→y=9`) — a constant multiplier's local derivative *is* the constant itself.
Combined with the sum's "distribute the gradient equally to every term" rule from before:

```python
dbndiff2 = (1.0/(n-1)) * torch.ones_like(bndiff2) * dbnvar
```

`cmp('bndiff2', dbndiff2, bndiff2)` → **exact: True, maxdiff: 0.0**.

**`bndiff` is used in two places, needs both contributions summed:**
```python
bnraw = bndiff * bnvar_inv    # path 1 — already computed earlier (partial dbndiff)
bndiff2 = bndiff**2             # path 2 — new
```

Path 2 uses the power rule with `n=2`: `f(x)=x^2 → f'(x)=2x`.

```python
dbndiff += 2 * bndiff * dbndiff2   # accumulate onto the path-1 contribution already held
```

`cmp('bndiff', dbndiff, bndiff)` → **exact: True, maxdiff: 0.0** — confirms both paths'
contributions combined correctly.

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