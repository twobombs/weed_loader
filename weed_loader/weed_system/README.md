# Weed System

The `weed_loader/weed_system` directory implements the low-level Foreign Function Interface (FFI) bridging Python to the compiled Weed C++ shared library. It abstracts away the dynamic library loading and sets up the strict memory interfaces expected by the C++ engine.

## Files and Features

- **`__init__.py`**: Initializes a global instance of `WeedSystem` mapped to the variable `Weed`. This singleton allows the rest of `weed_loader` to access the C++ functions without redundantly loading the shared library.
- **`weed_system.py`**: Defines the `WeedSystem` class.
  - **Dynamic Library Loading:** Uses `ctypes.CDLL` to load the compiled C++ Weed engine. It handles cross-platform path resolution, falling back to typical installation directories (`C:/Program Files (x86)/Weed/...` on Windows, `/usr/local/lib/weed/...` or `/usr/lib/weed/...` on Linux/macOS) if a relative or local path is not found. It respects the `WEED_SHARED_LIB_PATH` environment variable as a priority override.
  - **Function Signature Declarations:** Explicitly declares the `argtypes` (argument types) and `restype` (return type) for numerous exported C functions:
    - **Module management:** `load_module`, `save_module`, `free_module`.
    - **Inference/Execution:** `forward`, `forward_int`, `train_step`.
    - **Memory/Result retrieval:** `get_result`, `get_result_index_count`, `get_result_dims`, `get_result_size`, `get_result_offset`, `get_result_type`.
    - **Error handling & caching:** `get_error`, `reset_kv_cache`, `set_max_kv_seq_len`.

This directory acts as the crucial foundation for `weed_module.py`, translating Python objects and primitive types into explicit low-level structures compatible with the Weed ABI.
