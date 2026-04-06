# 🚀 MicroTorch

**MicroTorch** is a lightweight, hardware-accelerated deep learning framework built from scratch. It features a custom autograd engine, supporting both **CPU (NumPy)** and **GPU (CuPy)** backends with seamless device migration.

---

## 🌟 Key Features

* **Hardware Agnostic:** A custom `Array` class handles the heavy lifting between `numpy` and `cupy`, providing a unified `xp` interface for tensors.
* **Unified Device Management:** Automatic handling of data migration between CPU and GPU.
* **Dynamic Autograd Engine:** Supports backpropagation through binary and unary operations, building the computational graph on the fly.
* **Modular API:** Mirrors the PyTorch experience with `nn.Module`, `nn.functional`, and `optim` structures.
* **Efficient Data Loading:** Built-in `DataLoader` for batching, shuffling, and preprocessing.

---

## 🏗 Project Architecture

### 1. Core: Array & Tensor
The `Array` class abstracts the compute backend. By using an `xp` dispatcher, the `Tensor` class remains device-agnostic, allowing you to write code once and run it on either a CPU or an NVIDIA GPU.
* **Autograd:** Implements the chain rule to compute gradients for all operations in the graph.

### 2. Neural Networks (`nn`)
The `nn` directory is split into two logical parts:
* **`functional/`**: Contains the core mathematical logic for activations (ReLU, Sigmoid), losses (MSE, CrossEntropy), and layers.
* **`modules/`**: Contains stateful classes like `Linear` and `Module` that manage parameter registration and the forward pass lifecycle.

### 3. Optimization (`optim`)
This folder contains optimization algorithms (like SGD). These algorithms use the `.grad` attributes populated by the autograd engine to update the model parameters.

### 4. Data Handling (`data`)
The `data` folder includes the `DataLoader`, which is responsible for:
* Batching input samples.
* Shuffling data to ensure stochasticity during training.
* Managing memory efficiency during data streaming.

---
