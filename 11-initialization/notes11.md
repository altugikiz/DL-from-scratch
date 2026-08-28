# 11 - Initialization

**Course:** deeplearning.ai — Course 2 (Improving Deep Neural Networks), Week 1
**Assignment:** Initialization

## What this assignment covers

How you initialize the weights of a neural network has a big effect on how well
(and how fast) it trains. This assignment compares three strategies on the same
3-layer network (LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID):

1. **Zeros initialization**
2. **Random initialization** (large values, scaled by 10)
3. **He initialization** (scaled by `sqrt(2 / n[l-1])`)

## 1. Zero initialization

```python
parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
```

**Result: 50% train accuracy — pure random guessing.**

Why it fails: if every weight starts at zero, every neuron in a layer computes the
exact same function of the input, gets the exact same gradient, and updates the
exact same way. The network never "breaks symmetry" — it behaves like a single
neuron (logistic regression) no matter how many units you add.

**Takeaway:** weights must be initialized randomly to break symmetry. Biases can
safely stay at zero — the randomness in W is enough.

## 2. Random initialization (large values)

```python
parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
```

**Result: 83% train accuracy — much better, but not great.**

Why it's still not ideal:
- Cost starts very high because sigmoid outputs land very close to 0 or 1 for
  some examples, and `log(0)` blows up the loss.
- Large weights push activations into the flat regions of sigmoid/tanh, where
  gradients are tiny → **vanishing/exploding gradients**, slow learning.

**Takeaway:** initializing to very large random values doesn't work well — but
initializing to *small* random values (scaled properly) should.

## 3. He initialization

```python
parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2. / layers_dims[l - 1])
parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
```

**Result: 99% train accuracy — separates the classes cleanly, fast.**

Why it works: the scaling factor `sqrt(2 / n[l-1])` keeps the variance of
activations roughly stable as they pass through layers, which is what you want
for **ReLU** activations (ReLU zeroes out negative inputs, so the factor of 2
compensates for that).

- **Xavier initialization** is the equivalent idea for sigmoid/tanh activations,
  using `sqrt(1 / n[l-1])` instead of `sqrt(2 / n[l-1])`.

## Summary table

| Model | Train accuracy | Problem/Comment |
|---|---|---|
| Zeros initialization | 50% | fails to break symmetry |
| Large random initialization (×10) | 83% | weights too large → slow/unstable training |
| He initialization | 99% | recommended for ReLU networks |

## Key takeaways

- Different initializations lead to very different results, even with identical
  architecture and training loop.
- Random initialization breaks symmetry so different hidden units can learn
  different features.
- Don't initialize to values that are too large — it causes slow convergence
  and vanishing/exploding gradients.
- He initialization is the go-to for ReLU-based networks; Xavier for
  sigmoid/tanh-based ones.