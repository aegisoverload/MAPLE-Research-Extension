# Model state and upgrade paths

## Current text MAPLE (verified on local Windows environment)

- Uses original MAPLE `API` and `Voting_histogram` in `biorxiv/MAPLE/MAPLE_windows.py`.
- Uses `biorxiv/MAPLE/mini_experiment.py` to run text MAPLE and a simplified AugPE baseline.
- With `--model_id Qwen/Qwen2.5-1.5B-Instruct`, Local Transformers backend in `biorxiv/MAPLE/hf_backend.py` generates text.
- `evaluate_mauve.py`, `evaluate_metadata.py`, `evaluate_faithfulness.py`, `privacy_accounting_calibrated.py` are diagnostic tools. Latest conditional accounting is NOT a certified end-to-end DP proof.
- Original experiments at 30 and 100 samples are smoke and scale tests; they are NOT identical to author's 7B / exact prompts / full-data runs.

## Switching to a larger TEXT model (same output type)

1. Keep `mini_experiment.py`, `MAPLE_windows.py` and metadata/voting logic unchanged.
2. On a capable computer, install a compatible PyTorch/Transformers GPU stack, then try `python biorxiv/MAPLE/mini_experiment.py --method maple --model_id Qwen/Qwen2.5-7B-Instruct --PE_T 1 --PE_num_syn 4 --noise_multiplier 0 --output_dir results/7b_smoke`.
3. The 7.61B weights in 16-bit precision consume about 15.2GB (decimal) plus substantial model overhead/KV cache; a 24GB GPU is the practical starting point. Windows 8GB RTX 3070 won't comfortably run the full 16-bit model.
4. Optionally evaluate 4-bit quantization/offload on that hardware. This **requires changes in `hf_backend.py`** (e.g. quant config); test exact OS/CUDA support separately, don't promise the same performance.
5. Keep same evaluation data, prompt, seed, temperature, max_tokens, and version logging for fair 1.5B vs 7B comparison.

## Switching to a hosted TEXT LLM

The current `MAPLE_windows.API` uses `AutoTokenizer.apply_chat_template` and `self.llm.generate` tied to the HF backend. To use Gemini/OpenAI as text provider, introduce a message-oriented backend that takes `list[{'role','content'}]` and returns text. Avoid sending Qwen-formatted raw prompt strings to another vendor. Preserve `RANDOM_API`/`VARIATION_API` at the pipeline boundary.

## Switching to IMAGE generation

**NOT a model-id-only swap.** Use the included `image_extension/` `ImageBackend.generate(GenerationRequest)->Path` contract. Real images replace text rows; voting uses local image encoder, outputs are image paths; metadata schema/DP AIM needs image-specific preprocessing; image metrics replace MAUVE. Image extension's Mock backend is runnable now, but real API and all scientific validation are still TODO.

## Common confidentiality model

Local-only: private original text/images, private embeddings, raw NN matches, any non-DP statistics.
May reach third-party provider if permitted: DP-derived metadata, donated PUBLIC examples and prior synthetic outputs. API keys must remain in OS env or secrets manager.
