# Teammate API work — precise integration contract

Target location: `image_extension/src/image_maple/providers/teammate_backend.py`.
No new repository and no rewrite of `mini_experiment.py` necessary to start.

## Input

`TeammateBackend.generate(request)` receives:
- `request.mode`: `random` or `variation`
- `request.metadata`: DP-generated categorical metadata (provenance checked by separate workflow)
- `request.examples`: public/donated `(image_path, metadata)` pairs, NEVER private images
- `request.source_image`: a **previous synthetic image** for variation, not raw private input
- `request.output_path`: absolute/local target path
- `request.seed`: optional generation seed, unrelated to DP noise
- `request.variation_strength`: optional API-facing edit guidance

## Return contract

Generate exactly one image, save **PNG at request.output_path**, return that exact `Path`.
If the provider refuses a request / rate limits / times out, raise a descriptive exception (do not silently return blank images). Make API model identifier, request limits and retry policy configurable. Keep an explicit call budget and log provider refusal counts without logging private data or keys.

## First three steps for the teammate

1. Clone full shared repo, create `feature/image-api` branch; install `pip install -e './image_extension[gemini]'` and set token as an environment variable locally. Test `--backend mock` first.
2. Fill `TeammateBackend.generate`, run one **PUBLIC-only** API request. Start with only text metadata and 0 reference images to confirm provider; then 1 public donor; then a synthetic variation.
3. `python -m image_maple.cli --backend custom --custom-backend image_maple.providers.teammate_backend:TeammateBackend --embedding pixel-smoke --allow-paid-api --max-paid-calls 6 --metadata image_extension/data/demo/metadata.csv --private-manifest image_extension/data/demo/private.csv --donated-manifest image_extension/data/demo/donated.csv --num-syn 2 --lookahead 1 --iterations 1 --sigma 0 --out image_extension/runs/custom_probe`

`--embedding pixel-smoke` and demo inputs only validate integration; real study must use local CLIP/Inception, authentic (licensed) data, and proper DP metadata provenance and accountant.

## Next scientific implementation stages

1. Agree on CelebA-style categorical attribute schema and privacy unit (image vs person).
2. Generate or verify DP-AIM metadata, never feed raw private images to hosted API.
3. Add local image embedding and Gaussian NN-voting with an audited budget for all rounds.
4. Add PE and SPTI baselines, image metrics FID and metadata-faithfulness/JSD.
5. Repeat across epsilon, random seeds and dataset types.

Note that API generation **is not model training**. MAPLE evolves synthetic data; downstream classifier training is a separate stage.
