# 12 - Regularization

**Course:** deeplearning.ai — Course 2 (Improving Deep Neural Networks), Week 1
**Assignment:** Regularization

## What this assignment covers

Same 3-layer network as the Initialization assignment, but the focus here is fighting
**overfitting** (high variance) rather than initialization. Two techniques:

1. **L2 regularization** — penalize large weights in the cost function
2. **Dropout** — randomly deactivate neurons during training

Baseline (no regularization): the model fits the noisy training points too closely
(overfits) — good train accuracy, weaker test accuracy.

## 1. L2 regularization

**Cost function** (`compute_cost_with_regularization`):

```python
L2_regularization_cost = (1. / m) * (lambd / 2) * (
    np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))
)
cost = cross_entropy_cost + L2_regularization_cost
```

Adds a penalty term proportional to the sum of squared weights across all layers.
Larger `lambd` → larger penalty → optimization pushes weights toward smaller values.

**Backward pass** (`backward_propagation_with_regularization`):

Only `dW1`, `dW2`, `dW3` change — each gets an extra `+ (lambd / m) * W` term
(the derivative of the L2 penalty). `db` terms are untouched since the penalty only
involves `W`.

**Why it works:** a model with small weights behaves more simply/smoothly — its output
changes less abruptly as inputs change. Penalizing large weights makes it "expensive"
for the network to fit noise, so it's forced toward a smoother decision boundary.

**Result:** train accuracy drops slightly, test accuracy improves — the classic
regularization trade-off. Too large a `lambd` can "oversmooth" into high bias.

## 2. Dropout

At each training iteration, each neuron in a layer is independently kept with
probability `keep_prob` and zeroed out otherwise. This forces the network not to rely
too heavily on any single neuron (co-adaptation), since that neuron might vanish on any
given pass.

**Forward pass** (`forward_propagation_with_dropout`) — 4 steps per dropout layer:
1. Generate a random matrix `D` the same shape as the activation `A`
2. Threshold it: `D = (D < keep_prob).astype(int)`
3. Apply the mask: `A = A * D`
4. Rescale: `A = A / keep_prob` ("inverted dropout" — keeps the expected value of
   activations the same, so no extra work is needed at test time)

**Backward pass** (`backward_propagation_with_dropout`):
- Reapply the *same* mask `D` (stored in the cache) to the corresponding `dA`
- Divide `dA` by `keep_prob` again (chain rule — the same scaling used in forward
  propagation)

**Critical rule:** dropout is used ONLY during training. At test/inference time, all
neurons stay active (no masking, no scaling) — a very common mistake is applying
dropout at test time too.

## Summary table (reference numbers from the original assignment)

| Model | Train accuracy | Test accuracy |
|---|---|---|
| No regularization | 95% | 91.5% |
| L2 regularization (λ = 0.7) | 94% | 93% |
| Dropout (keep_prob = 0.86) | 93% | 95% |

## Key takeaways

- Regularization reduces overfitting by trading some training accuracy for better
  generalization (test accuracy).
- L2 regularization drives weights toward smaller values ("weight decay").
- Dropout trains a different "thinned" sub-network on every iteration, making neurons
  less co-dependent.
- Dropout: training only, never at test time. Always apply the mask in both forward and
  backward passes, and always rescale by `keep_prob` (inverted dropout).

## Note on the dataset

The original assignment uses a proprietary `data.mat` file (French football pitch
positions) loaded via `scipy.io`, only available inside the Coursera lab. Running this
notebook locally uses a synthetic stand-in 2D dataset (`load_2D_dataset()` in
`reg_utils.py`, built from `sklearn.datasets.make_moons` + a rotation) with a similar
two-region, noisy-boundary shape — so exact accuracy numbers will differ from the
92–95% figures quoted above, but the overfitting → regularization pattern should still
be visible.