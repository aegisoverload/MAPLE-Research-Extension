# MAPLE full-repository GitHub sharing — team handoff

## What is done vs not done

- **WORKING**: native Windows text prototype with Qwen2.5-1.5B-Instruct via Transformers; MAPLE RANDOM, private nearest-neighbor voting, rank selection, variation, repeated runs, AugPE prototype, evaluation, conditional zCDP diagnostics.
- **NOT CLAIMED**: full paper-matched 7B reproduction; certified (epsilon, delta) end-to-end DP; production-safe image generation; image AIM or image FID completed.
- The new `image_extension/` adds an independent image-space backend interface **inside** the same repository. No existing files are overwritten.

## Before first push (Git Bash, from current Windows MAPLE root)

1. Create a NEW **private**, empty GitHub repo under your own account/org, without README. Invite teammates as collaborators.
2. Back up your original working directory before making Git operations.
3. Copy **the contents of this overlay** into the ROOT of the existing `MAPLE/` directory (not a sibling repo). The existing `biorxiv`, `aim`, `README.md` remain.
4. Append `.gitignore.additions` to an existing `.gitignore` (do NOT overwrite existing contents):
   `cat .gitignore.additions >> .gitignore`.
5. Run `git status --short`, `git remote -v`, and inspect ALL paths to be committed.
6. If `DPSDA/` is a second Git clone, either register it as a real Git submodule (best), or ignore it and document the URL. Do not rely on a broken embedded-git-pointer.
7. Run `python tools/check_staged_git.py` **after** `git add` to check staged source. Manually inspect Git history and licences too.

## Push while preserving upstream source and full project history

The local MAPLE repo was originally cloned from the authors. Git remote `origin` probably points to the AUTHORS. Do **not** `git push` to their repository.

```bash
cd ~/Desktop/MAPLE
git remote -v
git branch --show-current
# Only if origin currently points to the AUTHORS' MAPLE repository:
git remote rename origin upstream
# Your NEW private empty repository URL:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_PRIVATE_REPO.git
# Git identity only if not already configured:
# git config user.name 'YOUR_GITHUB_NAME'
# git config user.email 'YOUR_EMAIL'
git add .gitignore .gitignore.additions docs image_extension tools .github
# Plus explicitly add all the other LOCAL script modifications you want shared,
# e.g. text backend, original patches, accounting and evaluation scripts.
git add biorxiv/MAPLE/MAPLE_windows.py biorxiv/MAPLE/mini_experiment.py
git status --short
python tools/check_staged_git.py
git commit -m 'Share working Windows MAPLE and image API extension'
git push -u origin HEAD:main
```

If your origin is NOT upstream, do not blindly rename it. Recheck remotes before commands. `git add .` is convenient only after careful review of all untracked or large files. `.gitignore` doesn't untrack already committed files; use `git rm --cached ...` if necessary (do not delete local data) and review history leakage.

### DPSDA nested dependency

A fully cloned second repo under `DPSDA/` should ideally be a Git submodule, with `.gitmodules` tracked. Inspect first:

```bash
git -C DPSDA remote -v
git -C DPSDA status --short
git submodule status
```

If it is not registered and its origin is the Microsoft DPSDA project, `git submodule add https://github.com/microsoft/DPSDA.git DPSDA` may register an existing clean checkout. If Git says the path is already tracked, stop and reconcile rather than using `git rm -rf` blindly.

Teammate then clones with:

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/YOUR_PRIVATE_REPO.git
```

## First run on teammate's machine

```bash
cd YOUR_PRIVATE_REPO
python -m venv .venv
# Git Bash on Windows:
source .venv/Scripts/activate
python -m pip install --upgrade pip
# Original text prototype needs PyTorch, Transformers, sentence-transformers, DPSDA & other Windows dependencies;
# use your established environment/setup guide; no hosted API is needed for the local Qwen backend.
# For the independent image extension's mock mode:
python -m pip install -e ./image_extension
python image_extension/scripts/make_demo_data.py
python -m unittest discover -s image_extension/tests -v
```

## Collaboration process

- `main`: verified baseline; both teammates can read all project code.
- `feature/image-api`: API teammate edits `image_extension/src/image_maple/providers/teammate_backend.py`, API demos and tests, commits and opens a PR.
- `feature/image-metadata-dp`: you prepare image schema and AIM/DP provenance, local voting model, evaluations, opens a PR.
- Rebase/merge through PR; do not share token via repository/Issue/screenshot.
- Shared outputs of real experiments are tracked as small **sanitized summaries** only. Private or restricted images, weights, checkpoints and raw data are obtained independently on each machine or via permitted storage.

## Model upgrades

Text: `python biorxiv/MAPLE/mini_experiment.py --model_id Qwen/Qwen2.5-7B-Instruct ...` on a machine with enough VRAM (usually ~16GB just for FP16 weights plus runtime overhead; 24GB or more is preferable). `hf_backend.py` must support the specified tokenizer/model and target device. The 8GB RTX 3070 cannot comfortably host 7B FP16 without memory-saving strategies.

Image: change `--backend gemini --image-model MODEL_ID`, or implement another image provider in `teammate_backend.py`. The 1.5B text LLM weights cannot be swapped directly for an image generator: output schema, embedding, evaluation and variation differ.

### Copyright

The source MAPLE repository is public but currently does not visibly display a license in GitHub's repository sidebar. Do not assume public visibility grants blanket redistribution rights; check the authors' LICENSE and your institution's policy before making any copied upstream code public. Keeping team repo PRIVATE reduces exposure but does not replace a license check. Maintain author attribution and original history.
