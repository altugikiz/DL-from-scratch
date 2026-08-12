# DL-from-scratch

A learning log for implementing deep learning concepts from scratch — no frameworks, just Python.
The goal: internalize the math and mechanics behind deep learning by building it with code before relying on libraries like PyTorch or TensorFlow.

## Contents

- [01-micrograd](./01-micrograd) — A tiny autograd engine and a simple MLP (multi-layer perceptron), built from scratch in Python. Inspired by Andrej Karpathy's "Neural Networks: Zero to Hero" series.

## Why this repo exists

I'm taking a Deep Learning course this semester and wanted to build a solid foundation beforehand — not just to pass the course, but to actually understand what's happening under the hood when a model calls `.backward()`.

Each folder in this repo covers one core concept, implemented from first principles, with notes on what I learned along the way.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```