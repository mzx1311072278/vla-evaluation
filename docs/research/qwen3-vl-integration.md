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

## Multi-image context budget

### Native limit and exact image-token formula

The 8B checkpoint's native text context is `262144` tokens (the “256K” in Qwen's
documentation): both `text_config.max_position_embeddings` and the tokenizer's
`model_max_length` are 262144. Qwen documents YaRN configuration for inputs beyond
256K, up to 1M, but that requires changing the checkpoint configuration and is not
part of the normal local loader. Reserve output tokens and all prompt/chat-template
tokens inside the same 262144-token budget.

For an image resized to `(H', W')`, the checkpoint has:

- vision patch size `P = 16`;
- spatial merge size `M = 2`;
- temporal patch size `T = 2`.

The image processor first requires both spatial dimensions to be multiples of
`P * M = 32`. Its spatial patch grid is `grid_h = H' / 16` and
`grid_w = W' / 16`. A still image is duplicated along time to fill one temporal
patch, so `grid_t = 1`; `T=2` changes the content of each vision patch vector but does
not double or halve the number of LLM tokens for a still image. The processor source
then expands the image placeholder to exactly:

```text
image_pad_tokens = grid_t * grid_h * grid_w / M²
                 = H' * W' / (16² * 2²)
                 = H' * W' / 1024
```

The chat template also emits one `<|vision_start|>` and one `<|vision_end|>` per
image. Thus the exact per-image context contribution with the repository's default
`add_vision_id=False` is `image_pad_tokens + 2`. Prompt text, user/assistant control
tokens, and generated output are additional. DeepStack injects three ViT feature
levels into early language layers, but it does not create extra input token IDs and
therefore does not change this context count.

For an actual video with `F` frames (not this repository's list of independent still
images), temporal patching instead gives `grid_t = ceil(F / 2)` after padding, before
the spatial `/ M²` merge. Qwen3's video processor additionally inserts timestamp and
visual boundary tokens per temporal group, so do not reuse the still-image total for
video inputs.

Sources:

- [Qwen3-VL-8B-Instruct text and vision config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)
- [Qwen3-VL-8B-Instruct tokenizer context limit](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/tokenizer_config.json)
- [Qwen3-VL-8B-Instruct processor defaults](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/preprocessor_config.json)
- [Transformers 4.57 Qwen3-VL placeholder expansion](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)
- [Transformers 4.57 patch-grid construction inherited by Qwen3-VL](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py)
- [Transformers 4.57 Qwen3-VL DeepStack implementation](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- [Qwen native 256K and YaRN guidance](https://github.com/QwenLM/Qwen3-VL#processing-long-texts)

### What the repository's 336-pixel cap produces

`qwen-vl-utils` 0.0.14 uses the same `smart_resize` rule: round each dimension to the
nearest multiple of `image_patch_size * spatial_merge_size = 32`, then enforce pixel
bounds while preserving aspect ratio. Its Qwen3 call must pass
`image_patch_size=16`; `336 / 32 = 10.5`, and Python's `round` is ties-to-even, so a
336-pixel dimension becomes 320 rather than 352. The repository already resizes the
long edge to at most 336 before this step.

Representative exact counts are:

| Frame before smart resize | Model size `(H', W')` | Image-pad tokens | Total context per image |
| --- | ---: | ---: | ---: |
| 336 × 336 (1:1) | 320 × 320 | 100 | 102 |
| 252 × 336 (4:3 landscape) | 256 × 320 | 80 | 82 |
| 189 × 336 (16:9 landscape) | 192 × 320 | 60 | 62 |

Portrait images have the same totals with height and width exchanged. These examples
are above qwen-vl-utils' Qwen3 minimum (4096 pixels). The 16:9 result is below the
checkpoint processor's own 65536-pixel default, but the supported integration passes
the utility-resized image with `do_resize=False`, so the processor does not enlarge it
a second time. Explicit `resized_height` / `resized_width` values are also rounded to
multiples of 32.

Source:

- [Qwen qwen-vl-utils 0.0.14 smart-resize implementation](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)

### 48, 64, and more still images

The table below counts only each image's expanded image-pad tokens plus its two visual
boundary tokens. Add the tokenized attempt prompt/chat controls and reserve
`max_new_tokens` (currently 256) to obtain the actual sequence budget.

| Images | 1:1 at 102 tokens/image | 4:3 at 82 tokens/image | 16:9 at 62 tokens/image |
| ---: | ---: | ---: | ---: |
| 48 | 4,896 | 3,936 | 2,976 |
| 64 | 6,528 | 5,248 | 3,968 |
| 96 | 9,792 | 7,872 | 5,952 |
| 128 | 13,056 | 10,496 | 7,936 |
| 256 | 26,112 | 20,992 | 15,872 |
| 512 | 52,224 | 41,984 | 31,744 |

Consequently, 48 or 64 of the repository's capped still frames are nowhere near the
262144-token architectural context limit. Even in the square worst case, 512 images
use only 52224 image-related tokens. The purely arithmetic ceiling is 2570 square
images (`floor(262144 / 102)`) only if prompt and output consumed zero tokens, which
they never do; it is not a practical deployment target. The current profile's 8
global + 8 dense frame caps are far below these context limits.

### Context capacity versus RTX 4090 memory

“Fits within 262144 tokens” does **not** imply “fits in 24 GB VRAM.” The native BF16
weight shards already total 17,534,247,392 bytes (16.33 GiB). In addition, CUDA needs
vision-encoder intermediates, language-model prefill activations, allocator/workspace
memory, and the generation KV cache. Multi-image prefill also processes four unmerged
16×16 spatial patches for every final image-pad token, so its transient vision cost is
not represented by the final context count alone.

From the official 8B config (36 layers, 8 KV heads, head dimension 128), the standard
BF16 K/V cache payload is approximately:

```text
KV bytes/token = 2 (K and V) * 36 * 8 * 128 * 2 bytes = 147456 bytes (144 KiB)
```

Ignoring prompt text and implementation overhead, square 336-capped frames therefore
add about 688.5 MiB of KV payload at 48 images (4896 tokens), 918 MiB at 64 images
(6528 tokens), 1.79 GiB at 128 images, and 3.59 GiB at 256 images. Prefill activations
and vision tensors can make the real peak materially higher; non-Flash attention can
also use substantially more temporary memory. Exact peak VRAM cannot be inferred from
the context formula and must be measured with the installed Torch/Transformers/CUDA
stack.

Operationally, retain concurrency one and the image-size cap, test 48 and 64 images
separately under `torch.cuda.max_memory_allocated()`, and leave headroom rather than
targeting the 24 GB nameplate. `device_map="auto"` may keep the request alive by CPU
offload, but that changes latency drastically. Qwen's official recommendation for
multi-image/video memory saving is Flash Attention 2; it does not reduce model weights
or the persistent KV payload. For this 4090 deployment, VRAM—not the 256K context
limit—is expected to become the binding constraint first.

Sources:

- [Qwen3-VL-8B-Instruct official weight index](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/model.safetensors.index.json)
- [Qwen3-VL-8B-Instruct attention dimensions and layer count](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)
- [Qwen official Flash Attention 2 recommendation](https://github.com/QwenLM/Qwen3-VL#flash-attention-2-to-speed-up-generation)
