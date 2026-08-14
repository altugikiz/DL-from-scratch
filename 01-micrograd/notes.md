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

## 12. Extending `Value`: negation and subtraction

Only had `+` and `*`. Needed `-` to compute a loss like `(o - target)^2`. Rather than
writing a brand new backward rule, expressed subtraction in terms of what already exists:

```python
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
```

Also had to make `__mul__` tolerant of plain Python numbers (not just `Value` objects),
since `self * -1` passes a raw `int`, not a `Value`:

```python
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        ...
```

Without this, `self * -1` would crash trying to access `.data` on an `int`. (Note:
`__add__` doesn't have this same guard yet — works for now since we haven't needed
`value + plain_number`, but would need the same fix if that comes up.)

## 13. Gradient descent — the actual "learning" step

Earlier experiment (nudging `o` up by following `o`'s own gradient) was a mistake in
framing — there was no real target, just an arbitrary direction to demonstrate the update
formula. Real training needs a **target/label** and a **loss function** that measures
distance from that target; backward must be run on the *loss*, not on the raw output.

Correct setup:

```python
target = Value(1.0)
loss = (o - target) * (o - target)   # squared error
loss.backward()
```

Update rule — always subtract the gradient (move *against* it), since the goal is always
to *decrease* the loss:

```python
learning_rate = 0.05
w1.data -= learning_rate * w1.grad
w2.data -= learning_rate * w2.grad
b.data -= learning_rate * b.grad
```

Ran one full cycle on the reference neuron (`x1=2.0, x2=0.0, w1=-3.0, w2=1.0,
b=6.8813735870195432`, target `1.0`):

- Before update: `o = 0.7071`, `loss = 0.0858`
- Gradients (via `loss.backward()`, not `o.backward()`): `w1.grad = -0.5858`,
  `w2.grad = 0.0`, `b.grad = -0.2929`
- After one update step (`lr=0.05`): `o = 0.7419`, `loss = 0.0666`

Loss went down, output moved closer to the target. This is one iteration of what will
become the training loop: forward → compute loss → backward → update → repeat.

## 15. `Neuron` class — generalizing the single-neuron example

Instead of manually declaring `x1, x2, w1, w2, b` every time, wrapped the pattern into a
reusable class. Number of weights always equals number of inputs (`nin`) — each input gets
its own weight because the network needs to learn how much each input should matter.
Bias adds an offset/shift so the neuron isn't forced through zero when all inputs are zero
(same role as the `+n` in `y = mx + n`). Both weights and bias start **random**, and are
learned from data via gradient descent — never hand-picked (the earlier `b = 6.881...`
example was a special, deliberately-chosen value just for matching a reference output).

```python
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
```

- `__call__` makes the object callable like a function: `n = Neuron(2); n([x1, x2])`.
- `x` here is the list of input `Value`s the neuron receives — e.g. `[Value(2.0), Value(3.0)]`.
- `zip(self.w, x)` pairs each weight with its matching input so `wi*xi` computes `w1*x1`,
  `w2*x2`, etc.; `sum(..., self.b)` adds them all up starting from the bias.

## 16. `Layer` class — multiple neurons, same inputs

A layer is several neurons side by side, all receiving the *same* input list, each with its
own independent (random) weights, each producing its own single output.

```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs
```

Important distinction that was initially confused: `nin × nout` gives the total number of
**weights** in the layer (e.g. `Layer(2, 3)` → 3 neurons × 2 weights each = 6 weights total).
But the layer's **output** is a list of length `nout` only — one number per neuron,
regardless of how many inputs each neuron takes in. Each neuron "summarizes" its inputs down
to a single number.

Tested `Layer(2, 3)` on `x = [Value(2.0), Value(3.0)]` → got a 3-element list of `Value`s, as
expected.

## 17. `MLP` class — chaining layers

An MLP (multi-layer perceptron) is layers chained so one layer's output list becomes the
next layer's input list.

```python
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

- `nouts` is a list of layer sizes, e.g. `[4, 4, 1]` means: hidden layer of 4 neurons →
  hidden layer of 4 neurons → output layer of 1 neuron.
- `sz = [nin] + nouts` builds the full chain of sizes (e.g. `[2, 4, 4, 1]`) so each `Layer`
  knows how many inputs it receives and how many neurons (outputs) it has.

Tested `MLP(2, [4, 4, 1])` on `x = [Value(2.0), Value(3.0)]` → got a single-element list
(since the final layer has only 1 neuron), confirming the chain works end to end: 2 inputs →
4 → 4 → 1 output. This is a full from-scratch neural network — no frameworks.

## 19. `parameters()` — collecting every weight and bias into one list

Manually referencing `mlp.layers[0].neurons[0].w[0]` for 41 separate parameters doesn't
scale. Added a `parameters()` method to each class that recursively collects everything
from its sub-components into one flat list:

```python
# in Neuron
    def parameters(self):
        return self.w + [self.b]

# in Layer
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

# in MLP
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

Verified parameter count by hand for `MLP(3, [4, 4, 1])`:
- layer 1 (`Layer(3,4)`): 4 neurons × (3 weights + 1 bias) = 16
- layer 2 (`Layer(4,4)`): 4 neurons × (4 weights + 1 bias) = 20
- layer 3 (`Layer(4,1)`): 1 neuron × (4 weights + 1 bias) = 5
- total = 41 — matched `len(mlp.parameters())` exactly.

**Bug hit along the way:** pasted all three `parameters()` methods into the same class
(`MLP`) instead of distributing one to each class. Symptom: `AttributeError: 'MLP' object
has no attribute 'w'` — because the `Neuron`-specific body (`self.w + [self.b]`) ended up
inside `MLP`, which has no `.w`. Lesson: when adding the "same shaped" method to multiple
related classes, double check each one landed inside the right class body, not just that
the method exists somewhere.

## 20. Full training loop on a toy dataset

Dataset (from the reference example):

```python
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
mlp = MLP(3, [4, 4, 1])
```

Also needed:
- `__pow__` on `Value` (for `(pred - target)**2`), using the power rule
  `d(x^n)/dx = n * x^(n-1)`.
- `isinstance` guard in `__add__` (mirroring the one already in `__mul__`) — without it,
  `Value - float` crashed with `AttributeError: 'float' object has no attribute 'data'`,
  because subtraction is implemented as `self + (-other)` and `other` here is a plain
  Python number, not a `Value`.
- `__radd__` on `Value` — needed because `sum(generator)` starts from plain `0` internally
  (`0 + Value(...)`), and Python falls back to `__radd__` on the right-hand operand when the
  left-hand type's own `__add__` fails.

Loop:

```python
for k in range(20):
    ypred = [mlp(x)[0] for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    for p in mlp.parameters():
        p.grad = 0          # must zero every iteration — _backward uses += (accumulates),
                              # so without this, gradients from the previous iteration would
                              # add onto the new ones instead of being replaced
    loss.backward()

    for p in mlp.parameters():
        p.data -= 0.05 * p.grad
```

Key clarification worked through: the gradient computed each iteration is *not* saved up
and used later — it's computed and consumed within the same iteration (steps: forward →
zero_grad → backward → update, in that order, every loop pass). What persists across
iterations is `p.data` (the parameter values), which each iteration nudges a little further
in the loss-reducing direction. The next iteration's forward pass then runs on the
just-updated parameters.

Result: loss went from `4.78` (iteration 0) down to `0.05` (iteration 19) — roughly a
100x reduction over 20 iterations. Final predictions vs targets:

```
target:  1.0   →  0.8712
target: -1.0   → -0.9530
target: -1.0   → -0.8687
target:  1.0   →  0.8975
```

All four predictions moved substantially toward their targets. This confirms the full
pipeline works end to end: a from-scratch `Value` autograd engine, chained into
`Neuron`/`Layer`/`MLP`, actually learns from data via gradient descent — no PyTorch,
no TensorFlow, just the two core rules (sum → gradient passes through as 1, product →
gradient is the other factor) applied recursively across a real (if tiny) network.

## 21. What's covered vs what's still open (vs. the Karpathy video)

Core mechanics from the video are done: `Value`, manual + automatic backward, topological
sort, `tanh`, building up `Neuron` → `Layer` → `MLP`, and a full training loop with
falling loss.

Not yet done, still on the list:
- **Graphviz visualization** of the computational graph (the `trace`/`Digraph` code seen in
  the original lecture screenshot) — would help visually confirm the graph structure
  instead of only reasoning about it in text.
- **Comparison against real PyTorch** — running the same forward/backward computation in
  actual PyTorch and checking the numbers match, as a final correctness/confidence check.
- A few more operator overloads (`__rmul__`, `__truediv__`, etc.) for full ergonomic parity
  with the video's version of `Value`.
- The video's end-of-lecture exercises (e.g. adding `exp`, `log`, alternative loss
  functions).