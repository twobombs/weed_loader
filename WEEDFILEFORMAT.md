# Weed File Format Specification

The Weed file format is a binary format used to serialize machine learning and quantum neural network models created in the Weed framework. It heavily uses C++ binary serialization to store modules, their nested parameters, and tensors sequentially into an output stream.

## Primitive Types

Serialization primitives are written directly as their binary representations using `std::ostream::write`.

*   **`bool`**: 1 byte boolean value.
*   **`size_t`**: Unsigned integer (typically 64-bit).
*   **`int64`** / **`int64_t`**: 64-bit signed integer.
*   **`symint`**: Platform-dependent signed integer (usually 32-bit or 64-bit depending on `WEED_TCAPPOW` macro, defaults to 64-bit in most systems).
*   **`tcapint`**: Platform-dependent unsigned integer (usually 32-bit or 64-bit depending on `WEED_TCAPPOW` macro, defaults to 64-bit in most systems).
*   **`bitLenInt`**: Integer size type used for bits.
*   **`real1`**: Floating-point scalar type. Depends on `WEED_FPPOW` macro (could be half, float, double, or float128. Default is `float` or `double`).
*   **`real1_f`**: Single precision float (32-bit `float`).
*   **`complex`**: Two consecutive `real1` values representing the real and imaginary parts of a complex number.

## Enumeration Types

Enumerations are serialized by casting to their underlying integer type (usually 4 bytes / `int`).

*   **`ModuleType`**: Represents the type of module (e.g., Linear, Embedding, Dropout).
*   **`StorageType`**: Represents the type of tensor storage (e.g., CPU Dense Real, GPU Sparse Complex).
*   **`QuantumFunctionType`**: Enum for Qrack-specific quantum functions.
*   **`Qrack::QNeuronActivationFn`**: Enum for activation function types within a quantum neuron.

## File Structure

A Weed model file consists of a single root **Module**. Serializing a module inherently serializes all of its sub-modules, parameters, and tensors recursively.

### 1. Storage Format (`Storage::save`)

Storage contains the actual data payload.

1.  `StorageType` (Enum, 4 bytes): The type of storage.
2.  `device_id` (`int64`): ID of the device (e.g., -1 for CPU, >= 0 for GPU).
3.  `size` (`tcapint`): Number of elements.
4.  **Payload**: Depends on the `StorageType`:
    *   **CPU/GPU Dense Real**: `size` elements of `real1`.
    *   **CPU/GPU Dense Complex**: `size` elements of `complex`.
    *   **CPU/GPU Dense Int**: `size` elements of `symint`.
    *   **CPU Sparse Real**:
        *   `ksize` (`tcapint`): Number of non-zero elements.
        *   Followed by `ksize` pairs of `(key, value)` where `key` is `tcapint` and `value` is `real1`.
    *   **CPU Sparse Complex**:
        *   `ksize` (`tcapint`): Number of non-zero elements.
        *   Followed by `ksize` pairs of `(key, value)` where `key` is `tcapint` and `value` is `complex`.

### 2. Parameter Format (`Parameter::save`)

Parameters wrap Storage with shape and stride metadata.

1.  `device_id` (`symint`)
2.  `offset` (`tcapint`)
3.  `shape_size` (`tcapint`): Number of dimensions.
4.  **Dimensions**: For `i = 0` to `shape_size - 1`:
    *   `shape[i]` (`tcapint`)
    *   `stride[i]` (`tcapint`)
5.  `Storage` Object: Serialized `Storage` block.

### 3. Module Format (`Module::save`)

Every module begins with its type identifier.

1.  `ModuleType` (Enum, 4 bytes)
2.  **Module Payload**: Depends on the `ModuleType` enum value:

#### SEQUENTIAL_T
1.  `size` (`tcapint`): Number of nested modules.
2.  Followed by `size` serialized `Module` objects.

#### LINEAR_T
1.  `in_features` (`tcapint`)
2.  `out_features` (`tcapint`)
3.  `weight` (`Parameter` object)
4.  `has_bias` (`bool`): True if bias exists.
5.  `bias` (`Parameter` object, only if `has_bias` is true)

#### SWIGLU_T
1.  `hidden_size` (`tcapint`)
2.  `intermediate_size` (`tcapint`)
3.  `gate_proj` (`Module` object / Linear)
4.  `up_proj` (`Module` object / Linear)
5.  `down_proj` (`Module` object / Linear)

#### DROPOUT_T
1.  `p` (`real1`): Dropout probability.
2.  `training` (`bool`): Training mode flag.

#### EMBEDDING_T
1.  `num_embeddings` (`tcapint`)
2.  `embedding_dim` (`tcapint`)
3.  `weight` (`Parameter` object)

#### LAYERNORM_T
1.  `features` (`tcapint`)
2.  `eps` (`real1`)
3.  `gamma` (`Parameter` object)
4.  `beta` (`Parameter` object)

#### GRU_T / LSTM_T
1.  `input_dim` (`tcapint`)
2.  `hidden_dim` (`tcapint`)
3.  `W_x` (`Module` object / Linear)
4.  `W_h` (`Module` object / Linear)

#### RMS_NORM_T
1.  `axis` (`symint`)
2.  `hidden_size` (`tcapint`)
3.  `weight` (`Parameter` object)

#### RESHAPE_T
1.  `size` (`tcapint`): Number of dimensions.
2.  `shape[i]` (`symint`) for each dimension.

#### ROPE_T (Rotary Positional Embedding)
1.  `head_dim` (`tcapint`)
2.  `max_seq_len` (`tcapint`)
3.  `base` (`real1_f`)

#### MULTIHEAD_ATTENTION_T
1.  `mask_val` (`real1_f`)
2.  `d_model` (`symint`)
3.  `num_heads` (`symint`)
4.  `num_kv_heads` (`symint`)
5.  `head_dim` (`symint`)
6.  `use_kv_cache` (`bool`)
7.  `kv_quant_bits` (`symint`)
8.  `W_q` (`Module` object)
9.  `W_k` (`Module` object)
10. `W_v` (`Module` object)
11. `W_o` (`Module` object)
12. `has_rope` (`bool`)
13. `rope` (`Module` object, only if `has_rope` is true)

#### TRANSFORMER_ENCODER_LAYER_T
1.  `d_model` (`tcapint`)
2.  `d_ff` (`tcapint`)
3.  `num_heads` (`tcapint`)
4.  `self_attn` (`Module` object)
5.  `ff1` (`Module` object)
6.  `ff2` (`Module` object)
7.  `norm1` (`Module` object)
8.  `norm2` (`Module` object)
9.  `activation` (`Module` object)

#### POSITIONAL_ENCODING_T
1.  `max_seq_len` (`tcapint`)
2.  `d_model` (`tcapint`)
3.  `pos_val` (`real1_f`)

#### LEARNED_POSITIONAL_ENCODING_T
1.  `max_len` (`tcapint`)
2.  `d_model` (`tcapint`)
3.  `pos_encoding` (`Parameter` object)

#### QWEN_DECODER_LAYER_T
1.  `d_model` (`tcapint`)
2.  `num_heads` (`tcapint`)
3.  `num_kv_heads` (`tcapint`)
4.  `self_attn` (`Module` object)
5.  `mlp` (`Module` object)
6.  `input_layernorm` (`Module` object)
7.  `post_attention_layernorm` (`Module` object)

#### QRACK_NEURON_LAYER_T
1.  `has_pre_init_circ` (`bool`)
2.  `pre_init_circ` (Binary Qrack circuit data, if true)
3.  `has_post_init_circ` (`bool`)
4.  `post_init_circ` (Binary Qrack circuit data, if true)
5.  `sdrp` (`real1_f`)
6.  `ncrp` (`real1_f`)
7.  `ace_qb` (`bitLenInt`)
8.  `sparse_mb` (`size_t`)
9.  `sparse_fl` (`real1_f`)
10. `num_input_indices` (`tcapint`)
11. `num_output_indices` (`tcapint`)
12. `num_hidden_indices` (`tcapint`)
13. `lowest_cmb` (`tcapint`)
14. `highest_cmb` (`tcapint`)
15. `pre_qfn` (`QuantumFunctionType` Enum)
16. `post_qfn` (`QuantumFunctionType` Enum)
17. `activation_fn` (`Qrack::QNeuronActivationFn` Enum)
18. `qrack_config_mask` (`tcapint`)
19. `neurons_angles` (`Parameter` objects, one for each neuron in the layer)

#### Stat/Aggregation Modules (MEAN_CENTER_T, SOFTMAX_T, LOGSOFTMAX_T, FLATTEN_T, MEAN_T, MAX_T, MIN_T, VARIANCE_T, STDDEV_T)
1.  `axis` (`symint`)

#### Activation and Utility Modules (GELU_T, RELU_T, SIGMOID_T, TANH_T, MIGRATE_CPU_T, MIGRATE_GPU_T)
*   No additional payload beyond the `ModuleType` enum.

## Backward Compatibility
The format is raw binary and directly serializes struct fields without versioning headers or alignment padding metadata, meaning it is tightly coupled to the C++ structure, platform endianness, and `#define` macros (e.g. `WEED_TCAPPOW` and `WEED_FPPOW`) active during compilation. Endianness is not strictly defined, assuming native byte order.