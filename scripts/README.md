# Weed Scripts

The `scripts` directory provides high-level command-line tools to interact with, adapt, and train models via the Weed inference engine. It bridges standard ML formats (like HuggingFace `safetensors`) to the minimalist, high-performance C++ Weed environment.

## Files and Features

- **`hf_to_weed.py`**: A robust conversion script that transforms HuggingFace `.safetensors` models into the Weed binary module format (`.weed`).
  - Implements conversion mappings for supported architectures: **GPT-2**, **BERT**, and **Qwen**.
  - Maps common transformer primitives (Linear, LayerNorm, Embeddings, Multi-head attention, Reshape, RMSNorm, RoPE, SwiGLU) to their low-level representations using `ModuleType` enumeration.
  - Formats parameter blocks iteratively with offset/shape/stride metadata matching `StorageType` definitions.
- **`fine_tuning.py`**: A lightweight script to perform fine-tuning loops on a converted `.weed` model.
  - Accepts an input prompt, completion string, learning rate, and HF tokenizer path.
  - Automatically loads a `Tokenizer` using `tokenizers` package, encodes the combined target text, and calls the `WeedModule.train_step` method using sequence shift.
  - Backs up the `.weed` file automatically before writing the fine-tuned parameters back to disk.
- **`weed_gpt2_chat.py` & `weed_qwen_chat.py`**: Experimental chat and inference interfaces designed for specific model architectures.
  - These scripts serve as proof-of-concept wrappers showcasing how a `WeedModule` object might accept token input natively and stream generated outputs.
  - They handle tokenization, looping inference (using `WeedModule.forward`), appending newly generated tokens to the context, and terminating on EOS logic.
