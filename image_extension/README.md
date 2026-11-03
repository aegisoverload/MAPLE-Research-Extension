# Image extension inside the FULL MAPLE repository

This is **part of the existing MAPLE repository** (`MAPLE/image_extension/`), not a replacement repository.
The verified text experiments remain in `MAPLE/biorxiv/MAPLE/`.

## Install (from MAPLE repository root)

```bash
python -m pip install -e ./image_extension
python image_extension/scripts/make_demo_data.py
python -m image_maple.cli \
  --metadata image_extension/data/demo/metadata.csv \
  --private-manifest image_extension/data/demo/private.csv \
  --donated-manifest image_extension/data/demo/donated.csv \
  --backend mock --embedding pixel-smoke \
  --num-syn 4 --lookahead 1 --iterations 1 --sigma 0 \
  --out image_extension/runs/smoke
```

## Hand off to the API teammate

Edit `image_extension/src/image_maple/providers/teammate_backend.py`.
The required method is `generate(request) -> Path`. See `docs/TEAM_API_CONTRACT.md`.

For a real image generation endpoint:

```bash
python -m pip install -e './image_extension[gemini,clip]'
# Set GOOGLE_API_KEY locally, NOT in source control
python -m image_maple.cli --backend gemini --embedding clip \
  --metadata PATH_TO_DP_DERIVED_METADATA.csv \
  --private-manifest PATH_TO_LOCAL_PRIVATE_MANIFEST.csv \
  --donated-manifest PATH_TO_PERMITTED_DONOR_MANIFEST.csv \
  --num-syn 4 --iterations 1 --lookahead 1 --sigma 0 \
  --allow-paid-api --max-paid-calls 12 --out image_extension/runs/gemini_probe
```

**Warning:** The mock/pixel smoke demo is functional plumbing only. Neither the mock pipeline nor any fixed sigma constitutes a certified end-to-end private image generator. No raw private images may be given to a third-party API. Obtain permission before API use with donor images, including celebrity or medical images.
