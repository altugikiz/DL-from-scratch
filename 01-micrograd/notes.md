# 01-micrograd — Notes

Learning log for building a tiny autograd engine from scratch, following Andrej Karpathy's
"Neural Networks: Zero to Hero" series.

## 1. Derivatives and gradients

- A **derivative** answers: "if I nudge the input a tiny bit, how much does the output change?"
  For a single-variable function `f(x)`, this is `f'(x)`.
- A **gradient** is the generalization of this idea to functions with multiple inputs.
  For `f(a, b, c)`, the gradient is the vector of partial derivatives:
  `[∂f/∂a, ∂f/∂b, ∂f/∂c]` — one number per input, telling you how much that specific
  input affects the output (holding the others constant).

## 2. Numerical gradient

Approximate the derivative using the limit definition, without symbolic math:

```
f'(x) ≈ (f(x + h) - f(x)) / h
```

- `h` should be small (e.g. `0.0001`) — as `h → 0`, the approximation approaches the true derivative.
- The approximation error shrinks roughly proportionally to `h` (error is `O(h)`).
- **Problem:** this requires re-running `f` for every single parameter you want a gradient for.
  For a neural net with millions of parameters, this is far too slow. This is why we need
  something better: backpropagation.

## 3. Two core derivative rules

For a function built purely from `+` and `*`:

- **Addition (`a + b`):** the local gradient of each term is `1`. Nudging `a` by 1 unit
  changes the sum by exactly 1 unit.
- **Multiplication (`a * b`):** the local gradient of one term is the *value of the other term*.
  `∂(a*b)/∂a = b`, `∂(a*b)/∂b = a`.

Any more complex expression is a chain of these two operations, so these two rules are
enough to build everything else on top (subtraction, powers, etc. can all be expressed via
`+` and `*`).

## 4. The `Value` class — building a computational graph

Instead of using plain Python floats, wrap every number in a `Value` object that remembers
its own history:

- `data` — the actual number
- `grad` — how much this value affects the final output (starts at `0`, meaning "not yet computed")
- `_prev` — the set of `Value` objects this one was created from (its "parents")
- `_op` — which operation created it (`'+'` or `'*'`)

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
```

- `_children=()` is a default argument — a leaf value created directly by the user
  (e.g. `Value(2.0)`) has no parents, so `_prev` ends up empty.
- `_prev` is a `set`, not a list/tuple, so that if the same value is used twice as a direct
  input to one operation (e.g. `a + a`), it's only stored once — this keeps the backward
  traversal from visiting/processing the same node redundantly at that step.
- Every `Value` starts with `_backward = lambda: None` — a no-op placeholder. Operations
  like `__add__` and `__mul__` overwrite this with a real function that knows how to push
  gradient to its parents.

## 5. Backpropagation, one step at a time

Given `a=2.0, b=-3.0, c=10.0`, `e = a*b`, `d = e + c`:

```
d.grad = 1.0        # a value's derivative with respect to itself is always 1
d._backward()        # pushes grad to e and c using the '+' rule
e._backward()        # pushes grad to a and b using the '*' rule
```

Result: `a.grad = -3.0`, `b.grad = 2.0`, `c.grad = 1.0` — matches the numerical gradient
computed earlier. This confirms the analytical (chain rule) approach agrees with the
numerical approximation.

Key insight: `self.grad = 0` (in `__init__`) and `d.grad = 1.0` (set manually before
calling backward) are two different things — the first is just an uninitialized default,
the second is the deliberate starting point of the backward pass, because the derivative
of the final output with respect to itself is always `1`.

## 6. Topological sort — automating the backward pass

Calling `._backward()` manually, one node at a time, in the right order, does not scale to
large graphs. The fix is a `backward()` method that automatically visits every node in the
correct order:

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
        node._backward()

Value.backward = backward
```

- `build_topo` is recursive: it processes all of a node's children *before* adding the node
  itself to `topo`. This guarantees children come before parents in the list.
- `reversed(topo)` flips this to parent-before-children order, which is exactly the order
  backward propagation needs (start at the root, push gradient down to the leaves).
- `self.grad = 1.0` automates what was previously done by hand — the root's derivative
  with respect to itself is always `1`.

Tested on the same `a, b, c, e, d` graph as before: calling just `d.backward()` (no manual
`._backward()` calls) reproduced the exact same gradients (`a.grad=-3.0`, `b.grad=2.0`,
`c.grad=1.0`). Confirms the automation is correct.

## 7. Gradient accumulation — why `+=` matters

Tested what happens when a value is reused in multiple places, e.g. `b = a + a` (so `a` is
used twice as input to the same operation).

Key idea: **total derivative**. If a value contributes to the output through multiple paths,
its total effect is the *sum* of the effects through each path — not the max, not just one
of them. Nudging `a` up by 1 unit changes `b` by 1 unit through the first `+ a`, and by
another 1 unit through the second `+ a`, for a total of 2. In neural net terms: if a weight
feeds into 5 different neurons, its effect on the loss is the sum of its effect through
each of those 5 paths.

This is exactly why `_backward` uses `self.grad += ...` and `other.grad += ...` instead of
`=`. With `+=`, each contribution accumulates on top of what's already there. With `=`, a
later contribution would silently overwrite an earlier one, silently producing a wrong
(too small) gradient.

Verified with `a = Value(3.0)`, `b = a + a` (i.e. `b = 2a`, so `db/da = 2`
analytically): `b.backward()` gives `a.grad = 2.0`. Matches.

## 8. Why a nonlinearity is needed (tanh)

Stacking only `+`/`*` layers never actually creates "depth" — it always collapses back into
one big linear function. Example: `z = w2*(w1*x + b1) + b2 = (w2*w1)*x + (w2*b1 + b2)`,
which is still just `z = W*x + B`. No matter how many linear layers you stack, the whole
thing simplifies to a single linear equation, so extra layers add zero expressive power.

Real-world relationships (image recognition, price curves, etc.) are usually not linear, so
a network needs a nonlinear "squashing" function between layers to actually benefit from
depth. Chose `tanh` (as in the Karpathy series) because:
- it squashes any input into a bounded range, `(-1, 1)`
- it has a clean derivative expressible in terms of its own output, which keeps the
  backward rule simple (see below)
- modern networks often use `ReLU` instead, but `tanh` is easier to derive by hand for
  learning purposes

## 9. Implementing `tanh` on `Value`

```
tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)     # algebraically equivalent to (e^x - e^-x)/(e^x + e^-x),
                                            # just numerically simpler (one exponential term)
d(tanh(x))/dx = 1 - tanh(x)^2
```

```python
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
```

- `tanh` is unary (only `self`, no `other`), so only `self.grad` gets a contribution.
- The local derivative `1 - t**2` reuses `t` (the already-computed forward value) instead
  of recomputing any exponentials — cheap to evaluate in the backward pass.
- `(self,)` — note the trailing comma. Needed to make this a one-element tuple; without it,
  Python would just treat it as `self` in parentheses, not a tuple, and `_prev` construction
  would break.
- **Bug hit along the way:** forgot `return out` in `__mul__` at one point, which silently
  made every multiplication return `None`. Lesson: every method that builds a new `Value`
  must explicitly `return out`, or the whole chain downstream breaks with confusing
  `NoneType` errors.

Verified `Value(0.0).tanh()` gives `data=0.0` (matches `tanh(0) = 0` analytically).

## 10. First full neuron: forward + backward

Built a single neuron with 2 inputs, weights, and a bias, matching the Karpathy reference
example:

```
x1=2.0, x2=0.0, w1=-3.0, w2=1.0, b=6.8813735870195432
n = x1*w1 + x2*w2 + b     # "pre-activation"
o = tanh(n)                # "activation" / output
```

Forward result: `n = 0.8814`, `o = 0.7071` — matches the reference video's numbers exactly
(the odd decimal value of `b` was chosen deliberately to produce a "nice" output).

Ran `o.backward()`:

```
x1.grad: -1.5
x2.grad: 0.5
w1.grad: 1.0
w2.grad: 0.0
b.grad: 0.5
```

- `w2.grad = 0.0` is notable: since `x2 = 0`, and a multiplication's local gradient is
  "the value of the other factor," `w2`'s gradient is exactly `x2 = 0`. Intuition: right now,
  changing `w2` has zero effect on the output, because it's being multiplied by zero anyway.
- `x1.grad = -1.5` is the largest in magnitude, because `x1` is multiplied by `w1 = -3.0`
  (a relatively large weight) — small changes to `x1` get amplified.
- In a real network, `w1.grad`/`w2.grad`/`b.grad` (the *parameters*) are what actually
  matter for learning — `x1.grad`/`x2.grad` (the *inputs*) are computed but not used, since
  inputs are fixed data, not something we update.

## 11. Open questions / next steps

- Next: use these weight gradients to actually update the weights (gradient descent):
  `w1.data += -learning_rate * w1.grad`, and similar for other parameters. This is the
  actual "learning" step.
- Then: wrap this single-neuron logic into a reusable `Neuron` class, then a `Layer`
  (multiple neurons sharing the same inputs), then an `MLP` (multiple layers chained
  together) — building up from this `Value` engine as the foundation.