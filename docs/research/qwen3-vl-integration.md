# Qwen3-VL integration notes

Scope: `Qwen/Qwen3-VL-8B-Instruct`, compared with this repository's current local
`Qwen2.5-VL` client. Sources are limited to Qwen and Hugging Face upstream material.

## Required runtime

- Use `transformers>=4.57.0`. Qwen's current upstream README states this as the minimum
  for Qwen3-VL. The 8B checkpoint itself records `Qwen3VLForConditionalGeneration`,
  `model_type: qwen3_vl`, and a Transformers 4.57 development version in `config.json`.
- Pin `qwen-vl-utils==0.0.14` when retaining this repository's explicit
  `process_vision_info` pipeline. Qwen documents 0.0.14 for Qwen3-VL and says it remains
  backward-compatible with Qwen2.5-VL.
- `torchvision` remains a runtime requirement for this repository. The 0.0.14 utility
  imports `torchvision` unconditionally, even when the caller supplies images rather
  than a video. It is also the utility's fallback video decoder. For HTTP(S) video
  input Qwen requires `torchvision>=0.19.0`; that version constraint is not needed for
  the repository's already-extracted local frame images, but a Torch-compatible
  torchvision build is still needed.
- `accelerate` is required for `device_map="auto"`. Qwen recommends Flash Attention 2
  for multi-image/video speed and memory savings, but it is optional and therefore
  should not be required for the initial compatibility change.
- A 24 GB RTX 4090 is plausible but not guaranteed to keep every request entirely on
  GPU: the official BF16 checkpoint shards total about 17.53 GB before CUDA runtime,
  activations, visual tokens, and KV cache. `device_map="auto"` may offload layers when
  memory is insufficient, which can make evaluation much slower. Keep VLM concurrency
  at one, retain the current frame/pixel caps, and validate representative worst-case
  episodes before deployment. Qwen publishes no official 4090 VRAM guarantee for this
  checkpoint; Flash Attention 2 is the official memory-saving recommendation.

Sources:

- [Qwen3-VL upstream installation and inference README](https://github.com/QwenLM/Qwen3-VL#quickstart)
- [Qwen3-VL-8B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Qwen3-VL-8B-Instruct config.json](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)
- [Qwen3-VL-8B-Instruct weight index](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/model.safetensors.index.json)
- [Bundled qwen-vl-utils 0.0.14 package metadata](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/pyproject.toml)
- [Bundled qwen-vl-utils vision imports and video backends](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)

## Model and processor loading

Both of these official approaches are valid for the dense 8B checkpoint:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",
    local_files_only=True,
)
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
```

The 8B model card instead imports the concrete
`Qwen3VLForConditionalGeneration` class. For this repository, the auto class is the
better compatibility seam: Transformers 4.57 maps both `qwen2_5_vl` and `qwen3_vl`
to their correct conditional-generation classes, so one loader can retain Qwen2.5-VL
and add Qwen3-VL without guessing from the directory name.

Qwen's current examples use `dtype=`, not `torch_dtype=`. Use `dtype="auto"` with
`device_map="auto"` on CUDA; the checkpoint declares BF16. Preserve the existing
explicit FP32 CPU fallback if CPU execution is still supported. Move processed inputs
to `model.device` as in the official examples rather than assuming the device string is
always `"cuda"`.

Sources:

- [Qwen3-VL upstream Transformers example](https://github.com/QwenLM/Qwen3-VL#using--transformers-to-chat)
- [Qwen3-VL-8B-Instruct concrete-class example](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct#quickstart)
- [Transformers 4.57 auto-model mappings](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/auto/modeling_auto.py)
- [Accelerate big-model inference and device maps](https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling)

## Messages and vision preprocessing

The existing message structure is compatible: a user message may contain multiple
`{"type": "image", "image": <local path>}` entries followed by text. The same chat
template and generated-token trimming pattern remain valid.

The important preprocessing difference is the visual patch size:

- Qwen2.5-VL: `image_patch_size=14` (the utility default).
- Qwen3-VL: `image_patch_size=16`.

Therefore the current bare `process_vision_info(messages)` call is not correct for
Qwen3-VL. Obtain the patch size from `processor.image_processor.patch_size` (or pass 16)
and call:

```python
image_inputs, video_inputs = process_vision_info(
    messages,
    image_patch_size=processor.image_processor.patch_size,
)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    do_resize=False,
    padding=True,
    return_tensors="pt",
)
```

`do_resize=False` matters because `qwen-vl-utils` has already resized the visual
inputs; Qwen warns that omitting it performs duplicate resizing. Qwen3-VL rounds image
dimensions to a multiple of 32, versus 28 for Qwen2.5-VL. The repository only sends
extracted images, so Qwen3's `return_video_metadata=True` and `video_metadata` processor
argument are not needed. They become required if the client later sends actual video
objects instead of frame images.

An alternative upstream path is to call `processor.apply_chat_template(...,
tokenize=True, return_dict=True, return_tensors="pt")`, which performs multimodal
preparation directly and avoids `qwen-vl-utils` in this call. Retaining the explicit
utility path is the smaller change for the current client.

Sources:

- [Qwen3-VL new qwen-vl-utils usage](https://github.com/QwenLM/Qwen3-VL#new-qwen-vl-utils-usage)
- [Qwen3-VL multi-image example](https://github.com/QwenLM/Qwen3-VL#using--transformers-to-chat)
- [qwen-vl-utils process_vision_info implementation](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)

## Generation and output behavior

Token generation, prompt-token trimming, and `batch_decode` work the same way. Follow
the Qwen3 examples and set `clean_up_tokenization_spaces=False` when decoding.

There is one behavior change relevant to strict JSON parsing: Qwen3-VL-8B-Instruct's
shipped `generation_config.json` enables sampling (`do_sample: true`, `temperature:
0.7`, `top_p: 0.8`, `top_k: 20`), whereas the Qwen2.5-VL-7B checkpoint is configured
with an effectively greedy temperature. Calling `generate` with only
`max_new_tokens`, as the current client does, inherits those checkpoint defaults and
makes Qwen3 responses stochastic. The compatibility layer should explicitly set
`do_sample=False` for repeatable evaluator JSON unless sampling is intentionally added
to the profile schema.

Use the **Instruct** checkpoint requested here. Qwen separately publishes a Thinking
checkpoint with different generation settings and reasoning-oriented output; it is not
a drop-in choice for a parser expecting only one compact JSON object.

Sources:

- [Qwen3-VL-8B-Instruct generation_config.json](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/generation_config.json)
- [Qwen2.5-VL-7B-Instruct generation_config.json](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/generation_config.json)
- [Qwen3-VL official Instruct and Thinking generation settings](https://github.com/QwenLM/Qwen3-VL#generation-hyperparameters)

## Repository impact summary

The minimal safe implementation is:

1. Raise/pin the GPU extra to `transformers>=4.57.0`, `qwen-vl-utils==0.0.14`, and add
   a Torch-compatible `torchvision` dependency.
2. Replace the hard-coded Qwen2.5 model class with
   `AutoModelForImageTextToText`, while keeping `AutoProcessor`.
3. Pass the processor's image patch size to `process_vision_info` and pass
   `do_resize=False` to the processor.
4. Move inputs to `model.device`, decode with cleanup disabled, and explicitly disable
   sampling for deterministic JSON.
5. Add a separate Qwen3 profile/model path; do not replace the existing Qwen2.5 profile.

No prompt-schema change is required solely to load Qwen3-VL, but the existing labeled
validation set should be run before treating its attempt counts as equivalent to the
Qwen2.5 results.
