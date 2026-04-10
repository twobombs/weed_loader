# Weed Loader

The `weed_loader` directory contains the core Python package for loading and interacting with compiled Weed C++ models. It provides a set of classes that wrap the low-level C++ library, exposing a clean API for inference, training steps, and tensor manipulation.

## Files and Features

- **`__init__.py`**: Exposes the main classes to be used by consumers of the package, including `DType`, `WeedModule`, `WeedTensor`, `WeedSystem`, and a global initialized `Weed` instance.
- **`dtype.py`**: Defines the `DType` integer enumeration (`NONE_DTYPE`, `REAL`, `COMPLEX`, `INT`), representing the data types that Weed supports.
- **`weed_module.py`**: Contains the `WeedModule` class, which serves as the primary interface for an instantiated Weed model.
  - **Loading & Saving:** Loads a binary `.weed` module from disk (`load_module`) and supports saving the module (`save_module`).
  - **Inference (`forward`):** Passes a `WeedTensor` to the C++ backend for forward propagation, allocating memory for the result and returning a new `WeedTensor`.
  - **Training (`train_step`):** Performs a single stochastic gradient descent (or Adam) step on the loaded model given input tokens and target tokens.
  - **KV Cache Management:** Exposes methods to configure the maximum sequence length (`set_max_kv_seq_len`) and reset the key-value cache (`reset_kv_cache`).
- **`weed_tensor.py`**: Defines `WeedTensor`, a minimal Python representation of multi-dimensional data arrays. It does not perform mathematical operations natively but rather stores:
  - Multidimensional structure information: `shape` and `stride` (using column-major ordering).
  - Storage details: `data` array, `offset`, and `dtype`.
  - Validates contiguousness and length against `shape` and `stride` on instantiation.

These classes abstract away the ctypes interaction found in `weed_system/`, providing a familiar object-oriented interface for machine learning practitioners.
