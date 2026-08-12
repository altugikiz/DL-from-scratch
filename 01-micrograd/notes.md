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

## 6. Open questions / next steps

- Calling `._backward()` manually, one node at a time, in the right order, does not scale
  to large graphs. Need an automatic way to visit every node in the correct order
  (topological sort) so a single `d.backward()` call handles everything.
- Haven't yet tested what happens when a value is reused in multiple places
  (e.g. `a` used in both `e = a*b` and `f = a*c`) — this is where the `+=` in `_backward`
  (instead of `=`) should matter, since gradients from multiple paths need to accumulate.
- Still only have `+` and `*`. Will need `tanh`, `exp`, `**` (power) and similar nonlinear
  operations to actually build a neural network (linear operations alone can't create
  depth that matters).