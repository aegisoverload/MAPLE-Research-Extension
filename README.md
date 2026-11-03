# MAPLE Research Extension

**共同研究專案｜文字版 MAPLE 重建 + 影像版 MAPLE 擴展**  
**最後更新：2026-11-03｜專案狀態：文字核心可執行，影像 API 整合中**

> **一句話說明：** 我們已在 Windows 原生環境完成文字版 MAPLE 的主要流程，並建立 AugPE 與初步評估、DP 會計工具。接下來在**同一個 Repository** 開發影像版：組員負責影像生成／編輯 API，另一端負責影像 Metadata、AIM、本地 Embedding／Voting 及實驗評估。  
> **注意：** 目前尚未完成「原論文數值的嚴格復現」或「正式端到端 DP 認證」；影像版目前是可執行的 Mock 骨架，尚非完成的研究模型。

**Repository：** https://github.com/aegisoverload/MAPLE-Research-Extension  
**原始 MAPLE：** https://github.com/elichien-google/MAPLE  
**DPSDA（Git Submodule）：** https://github.com/microsoft/DPSDA

## 目錄

1. [專案目標與技術概念](#1-專案目標與技術概念)
2. [目前進度與驗證紀錄](#2-目前進度與驗證紀錄)
3. [Repository 結構](#3-repository-結構)
4. [新組員：下載、安裝與第一個測試](#4-新組員下載安裝與第一個測試)
5. [兩位組員的工作分配](#5-兩位組員的工作分配)
6. [影像生成 API：介面與整合流程](#6-影像生成-api介面與整合流程)
7. [文字版 MAPLE 如何重新執行](#7-文字版-maple-如何重新執行)
8. [未來更換大型模型](#8-未來更換大型模型)
9. [DP 保證與資料安全邊界](#9-dp-保證與資料安全邊界)
10. [接下來的里程碑與驗收條件](#10-接下來的里程碑與驗收條件)
11. [GitHub 合作規則](#11-github-合作規則)
12. [已知問題與排除方法](#12-已知問題與排除方法)

---

## 1. 專案目標與技術概念

### 1.1 原始 MAPLE 在做什麼？

MAPLE 的主要想法是以 **DP Metadata** 輔助合成資料的初始化，再使用 Private Evolution（PE）的最近鄰投票、篩選與改寫，使生成分布逐漸靠近目標資料。它不是對基礎 LLM 進行參數微調，也不是「呼叫生成 API 就完成模型訓練」。

文字版主要流程：

```text
[Private text] --(本地 Embedding)--------------┐
                                               │
[DP AIM Metadata] + [公開/捐贈 examples]        │
                |                              │
          RANDOM_API (LLM)                     │
                |                              │
         Synthetic candidates                  │
                |                              │
                +--> NN Voting + DP Noise <----┘
                           |
                       Selection
                           |
                     VARIATION_API
                           |
                   (重複 Voting/Selection)
                           |
                    Synthetic dataset
```

### 1.2 影像版研究方向

根據內部《MAPLE 影像擴展：Dataset 與 Model API Survey》（2026-11-02），先開發 **影像空間 Evolution**，並保留 Caption／SPTI 路線作為後續對照。

```text
[Private images] --(本地 CLIP/Inception Embedding)--┐
                                                   │
[Image Metadata / DP AIM] + [授權的 donated images] │
                       |                           │
         IMAGE RANDOM_API (由組員接 API)             │
                       |                           │
               Synthetic PNG candidates             │
                       |                           │
                       +--> local NN Voting --------┘
                                  + DP Noise
                                     |
                                  Selection
                                     |
          IMAGE VARIATION_API (上一輪合成圖，不是私有圖)
                                     |
                             循環 / 最終 PNG
```

**保護邊界：** 私有原始影像及其 Embedding 留在本地。第三方 API 只接受符合授權的公開／donated 參考圖、前一輪合成圖，以及**已查核來源的 DP Metadata**。在完成此查核前，API 測試一律使用 Demo／公開資料。

---

## 2. 目前進度與驗證紀錄

### 2.1 已驗證：文字版核心流程

| 模組／項目 | 狀態 | 證據或備註 |
|---|---|---|
| Windows 原生執行 | ✅ 已跑通 | Windows + VS Code + Git Bash |
| 本地文字生成 | ✅ 已跑通 | `Qwen/Qwen2.5-1.5B-Instruct`，Transformers Backend |
| MAPLE Metadata RANDOM API | ✅ 已跑通 | 官方 bioRxiv AIM Metadata |
| VARIATION、NN Voting、Selection | ✅ 已跑通 | 多輪 CSV 輸出 |
| AugPE Baseline 原型 | ✅ 已跑通 | 同一批測試資料及本地模型；Prompt 為簡化版 |
| 文字版 Gaussian-noised Voting | ✅ 已跑通 | 包含安全亂數來源的 Smoke Test |
| 30 筆／組、1 輪 MAPLE vs AugPE | ✅ 已跑通 | 初步 MAUVE 計算完成 |
| 100 筆合成／500 筆 Private／2 輪 | ✅ 已跑通 | 約 1,696.57 秒，3 次加噪 Voting（較早版本） |
| 基本評估（重複、格式、字數） | ✅ 已跑通 | `evaluate_experiments.py` |
| MAUVE 與 Metadata 診斷 | ✅ 已跑通 | 小樣本診斷，不作論文優劣結論 |
| Metadata Faithfulness 初步分類 | ⚠️ 初步完成 | 部分類別 Macro-F1 偏低，需人工驗證 |
| 自動條件式 DP Noise 校準 | ✅ 計算與整合工具已建立 | **不是端到端 DP 認證** |

### 2.2 已測數據（僅供工程回歸比對）

- 30 筆初步 MAUVE：MAPLE **0.867315**；AugPE **0.873411**；Reference 實際等量抽樣 30 筆，且 FAISS 有分群樣本不足警告。**不可據此比較兩種方法優劣。**
- 100 筆合成／500 筆 Private／`PE_T=2`、σ=2.578758 的舊實驗：3 輪 Voting 正常；字數誤差率從 **0.4810** 下降到 **0.3552**。這是工程／品質觀察，**不可重新標記成使用新版安全亂數的 ε=4 實驗**。
- Replace-One、3 次 Voting、假設 AIM 使用總 ρ 的 10%、δ=1/28,846：保守計算 `minimum_sigma_conditional ≈ 3.186910`，加入 5% 數值餘裕後 `sigma ≈ 3.346256`，條件式 ε 上界約 **3.81557**。**AIM 實際成本未驗證。**

### 2.3 尚未完成（請勿誤寫成已完成）

- ❌ 與原始論文完全一致的 Qwen2.5-7B／完整資料量／原 Prompt／正式實驗重現。
- ❌ 經可核實 AIM 預算、抽樣／鄰接定義及完整發布稽核的端到端 `(ε,δ)`-DP 證明。
- ❌ 影像生成商用 API 的真實整合、拒絕／重試／計費測試。
- ❌ 影像資料集授權確認、真實 Image AIM、CLIP／Inception 基準驗證、FID／影像 JSD。
- ❌ PE／SPTI／Image MAPLE 的正式公平比較。

---

## 3. Repository 結構

以下列出**關鍵路徑**；實際以本 Repo 為準，檔案可能依後續 PR 擴充。

```text
MAPLE-Research-Extension/
├── README.md                         # 這份團隊總覽
├── README_UPSTREAM.md                # [建議保留] 原始 MAPLE README 的備份
├── aim/                              # AIM 與原始 DP metadata 工具
├── biorxiv/
│   └── MAPLE/
│       ├── MAPLE_windows.py          # 文字版 MAPLE / Voting / Selection
│       ├── mini_experiment.py        # 文字版 MAPLE / AugPE 小型實驗 CLI
│       ├── hf_backend.py             # 本地 Hugging Face Transformers LLM
│       └── util_windows.py           # Windows 相容工具
├── DPSDA/                            # Git Submodule，不要自行刪除 .git 資訊
├── image_extension/
│   ├── pyproject.toml                # 可安裝的影像模組
│   ├── README.md                     # 影像模組的細部操作
│   ├── src/image_maple/
│   │   ├── cli.py                    # Image MAPLE CLI 入口
│   │   ├── engine.py                 # 影像版迭代流程
│   │   ├── backends.py               # Mock Backend
│   │   ├── embeddings.py             # Pixel Smoke / CLIP 封裝
│   │   ├── contracts.py              # API request 資料結構
│   │   └── providers/
│   │       └── teammate_backend.py   # ★ 組員的 API 整合入口
│   ├── scripts/make_demo_data.py
│   └── tests/test_smoke.py
├── docs/
│   ├── TEAM_API_CONTRACT.md
│   ├── STATUS_AND_UPGRADE.md
│   └── FULL_REPO_GITHUB_GUIDE.md
├── evaluate_experiments.py           # 若已納入 Git
├── evaluate_mauve.py                 # 若已納入 Git
├── evaluate_metadata.py              # 若已納入 Git
├── evaluate_faithfulness.py          # 若已納入 Git
├── privacy_accounting_calibrated.py  # 若已納入 Git
├── run_maple_calibrated.py           # 若已納入 Git
└── tools/check_staged_git.py
```

> 原始 GitHub README 建議另存成 `README_UPSTREAM.md`，避免覆蓋後失去作者原本的安裝與論文資料說明。

---

## 4. 新組員：下載、安裝與第一個測試

### 4.1 先 Clone（包含 DPSDA）

需要 Git、Python 3.10+；Windows 建議使用 VS Code + Git Bash。

```bash
git clone --recurse-submodules https://github.com/aegisoverload/MAPLE-Research-Extension.git
cd MAPLE-Research-Extension
git submodule status
```

`DPSDA` 應顯示指定 Commit，而不是 `fatal`。若已經 Clone 但漏掉 Submodule：

```bash
git submodule update --init --recursive
```

### 4.2 建立環境與安裝影像擴展

Windows Git Bash：

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ./image_extension
```

> **環境區分：** 安裝 `image_extension` 的最低依賴，只能運作 Mock／Pixel Smoke，不會自動安裝文字版 Qwen、SentenceTransformers、完整 DPSDA 或 GPU 相容套件。文字版相依套件需要依 GPU／作業系統另行安裝與驗證，不能假設 `pip install -e ./image_extension` 就能復現所有實驗。

### 4.3 不需 API Token 的共同 Smoke Test

```bash
python image_extension/scripts/make_demo_data.py
python -m unittest discover -s image_extension/tests -v
```

再測試一次端到端 Mock Pipeline：

```bash
python -m image_maple.cli \
  --metadata image_extension/data/demo/metadata.csv \
  --private-manifest image_extension/data/demo/private.csv \
  --donated-manifest image_extension/data/demo/donated.csv \
  --backend mock \
  --embedding pixel-smoke \
  --num-syn 4 \
  --lookahead 1 \
  --iterations 1 \
  --sigma 0 \
  --out image_extension/runs/smoke
```

**驗收標準：** 指令完成、沒有 Exception、產生 PNG、`PE_iter_0.csv`、`PE_iter_1.csv`。這只測試程式接線；**不是 FID／DP／正式影像實驗**。

### 4.4 先建立自己的分支

API 組員：

```bash
git switch -c feature/image-api
```

Image Metadata／Voting 組員：

```bash
git switch -c feature/image-metadata-voting
```

不要在 `main` 直接放未完成的 API Key、私人資料或未測過的改動。

---

## 5. 兩位組員的工作分配

| 模組 | 主要負責 | 明確交付物／完成標準 |
|---|---|---|
| **文字版 MAPLE 維護** | 現有核心負責者 | 保留可執行的 `mini_experiment.py` 與結果設定 |
| **影像 API** | API 組員 | `TeammateBackend.generate()` 可完成 Random 與 Variation |
| API Key／請求管理 | API 組員 | 本機環境變數；timeout／retry／拒絕與呼叫量紀錄 |
| **Image Metadata Schema** | 核心負責者 | 固定欄位與合法選項，來源／授權文件 |
| **Image AIM / DP Prior** | 核心負責者 | 具來源的 Metadata CSV、schema 與會計紀錄 |
| **本地 Image Embedding／Voting** | 核心負責者 | 本地 CLIP／Inception 接入並測試 |
| FID、Metadata JSD、DP Audit | 共同 | 統一的實驗命令與可重算報告 |
| Baselines（PE、SPTI） | 共同 | 相同資料／預算／前處理下的對照實驗 |

**每個人都可以看整個 Repo；只有未完成的工作以 Branch 區隔。**

---

## 6. 影像生成 API：介面與整合流程

**組員主要修改位置：**

```text
image_extension/src/image_maple/providers/teammate_backend.py
```

不要重寫 `MAPLE_windows.py` 的文字版 Voting，也不要為 API 另開一個獨立 MAPLE Repository。

### 6.1 已定義好的介面

```python
class TeammateBackend:
    def generate(self, request):
        # request.mode: "random" 或 "variation"
        # request.metadata: 已核對來源的 DP metadata（Demo 時使用假資料）
        # request.examples: donated / 公開授權參考圖
        # request.source_image: 上一輪「合成」影像；random 模式為 None
        # request.output_path: 指定輸出 PNG 路徑

        # 呼叫外部 API -> 儲存 PNG -> 回傳 output_path
        return request.output_path
```

| Mode | 必須支援的輸入 | 預期輸出 |
|---|---|---|
| `random` | Metadata + 0～數張允許傳送的參考影像 | 1 張新 PNG |
| `variation` | 前一輪合成 PNG + Metadata + 可選參考圖 | 1 張新 PNG |

無論使用 Gemini、其他付費 API 或自架影像模型，都應保留相同的 `generate(request) -> Path` 回傳格式。**生成 API 不能讀取 Private Manifest，也不能將私有原始圖片送出去。**

### 6.2 API 組員的最小工作順序

1. 先跑 `--backend mock`；確保整條流程可執行。
2. 在 `TeammateBackend` 加入 Key 讀取與真實生成請求。**先只用文字 Metadata，0 張參考圖**。
3. 增加 1 張可授權傳送的 donated 圖；確認輸出 PNG。
4. 測試 `variation`：輸入上一輪合成 PNG，生成新的 PNG。
5. 加入超時、429／rate-limit、重試上限、模型拒絕、輸出影像檢查與實際呼叫數紀錄；不要無限重試。
6. 跑 Mock Test + 少量真實 API 測試；最後提交 PR。

API Key 僅儲存在組員本機環境變數或 Secret Manager，例如 `GOOGLE_API_KEY`；**絕不 commit `.env`、Key、憑證或帶 Token 的日誌**。

### 6.3 API 整合的 CLI 小型測試

先在本機安裝 image extension：

```bash
python -m pip install -e './image_extension[gemini]'
```

完成 `TeammateBackend` 後，使用 Demo（非私有）資料：

```bash
python -m image_maple.cli \
  --backend custom \
  --custom-backend image_maple.providers.teammate_backend:TeammateBackend \
  --metadata image_extension/data/demo/metadata.csv \
  --private-manifest image_extension/data/demo/private.csv \
  --donated-manifest image_extension/data/demo/donated.csv \
  --embedding pixel-smoke \
  --num-syn 2 \
  --lookahead 1 \
  --iterations 1 \
  --sigma 0 \
  --allow-paid-api \
  --max-paid-calls 6 \
  --out image_extension/runs/custom_probe
```

這個實驗最多**預先允許** 6 次生成請求，不代表能預測實際帳單。用戶端還需檢查真正的計費、失敗重試及模型 ID。**任何真實資料／私有 DP 主張都不能從這個 Demo 推出。**

### 6.4 API 的「完成定義」（Definition of Done）

- [ ] RANDOM 能生成合法 PNG，且正確保存。
- [ ] VARIATION 能根據前一輪**合成圖**生成新 PNG。
- [ ] 不讀取或上傳私有影像／私有 Embedding。
- [ ] 失敗時丟出可診斷錯誤，不默默產生空白圖片。
- [ ] 可選模型 ID、參考圖數量、重試與成本上限。
- [ ] 保持介面相容；Mock 測試仍全部通過。
- [ ] 使用開發分支 PR，附測試指令與小型非敏感輸出。

> **API 呼叫 ≠ 訓練影像模型。** MAPLE 的 Image Backend 用於產生及編輯 Synthetic Data；之後若要用合成影像訓練分類器，那是獨立的 Downstream Training 工作。

---

## 7. 文字版 MAPLE 如何重新執行

文字版由 `biorxiv/MAPLE/mini_experiment.py` 呼叫 `MAPLE_windows.py`、`hf_backend.py`，使用 Qwen 文字生成、MiniLM 與 bioRxiv CSV。

### 7.1 可重現的非 DP 小型測試

**前提：** 已安裝 PyTorch、Transformers、SentenceTransformers 與 Windows 相容的 DPSDA 依賴，官方 bioRxiv CSV 在其原本路徑。

```bash
python biorxiv/MAPLE/mini_experiment.py \
  --method maple \
  --PE_T 1 \
  --PE_num_syn 4 \
  --PE_L_random 1 \
  --num_private 100 \
  --noise_multiplier 0 \
  --seed 42 \
  --output_dir results/text_maple_smoke
```

AugPE：

```bash
python biorxiv/MAPLE/mini_experiment.py \
  --method augpe \
  --PE_T 1 \
  --PE_num_syn 4 \
  --PE_L_random 1 \
  --num_private 100 \
  --noise_multiplier 0 \
  --seed 42 \
  --output_dir results/text_augpe_smoke
```

### 7.2 執行結果

預期每個資料夾至少有：

- `config.json`：本次命令參數
- `PE_iter_0.csv`：初始化 Voting/Selection 後輸出
- `PE_iter_1.csv`：第一輪 Evolution 輸出
- `metrics.csv`：每輪執行時間、字數與 Voting 診斷

`noise_multiplier=0` **不是 DP 實驗**。不能將這些結果當成含隱私保證的合成資料公開。

### 7.3 目前隱私會計的位置

- `privacy_accounting.py` / `privacy_accounting_calibrated.py`：依假設計算 zCDP、Noise 門檻。
- `audit_maple_dp.py`：檢查程式與輸出設定。
- `run_maple_calibrated.py`：可讀取條件式校準 JSON 的執行 Wrapper。

**注意：** 正式 AIM ρ、對完整資料或固定抽樣的鄰接定義、發布項目及 Gaussian 實作安全性仍需完整查核。`within_target_conditional=True` **不等於 ε=4 已認證**。

---

## 8. 未來更換大型模型

### 8.1 文字模型：1.5B → 7B

文字版可透過 `--model_id` 切換：

```bash
python biorxiv/MAPLE/mini_experiment.py \
  --method maple \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --PE_T 1 \
  --PE_num_syn 4 \
  --num_private 100 \
  --noise_multiplier 0 \
  --output_dir results/qwen7b_probe
```

**前提：** GPU 顯存足夠、PyTorch/CUDA 與 Transformers 相容。

| 模型 | 權重 FP16 理論記憶體（近似） | 備註 |
|---|---|---|
| Qwen2.5-1.5B | 約 3 GB | 已在 RTX 3070 8GB 跑通 |
| Qwen2.5-7B | 約 15 GB | 加上 KV Cache／Runtime，24GB GPU 較實用 |
| 更大模型 | 更高 | 需要更大 GPU、量化、多卡或服務化 |

8GB RTX 3070 **不適合直接跑 7B FP16**。如選擇 4-bit 量化，需修改 `hf_backend.py` 的模型載入設定，並個別測試 Windows、CUDA、量化工具支援。不能只改 `--model_id` 就假設一定能跑。

### 8.2 Hosted 文字 API

要將本地 Qwen 換成雲端文字模型，應把 Prompt 的**角色訊息** (`system/user/assistant`) 傳給相容的 Backend；不要直接把 Qwen `apply_chat_template` 後的文字硬塞給另一家 API。

### 8.3 影像模型

**不能只把 Qwen 模型 ID 換成影像模型名稱。** 影像的輸入／輸出是 PNG、參考影像與影像 Embedder；請使用 `image_extension` 的 Backend 介面，另設 `--image-model` 或 provider 對應設定。

---

## 9. DP 保證與資料安全邊界

**最重要的共同規則：**

- Private Data、Private Embeddings、未加噪 NN 投票不能送給第三方生成 API。
- Donated／公開參考圖也必須確認授權**允許轉交第三方服務**；不能因為「研究用途」就假定可任意上傳。
- CelebA 等人臉資料、醫療影像需特別處理授權、隱私單位（image-level vs person-level）及可能的多圖同身分問題。
- AIM metadata 的檔案名稱含 `eps4` 並不自動證明它的真實 ρ；必須有產生設定與來源紀錄。
- 固定大小 500 筆子集與完整 28,846 筆資料的隱私宣稱不同；Replace-One／Add-Remove、δ 必須明確指定。
- 安全亂數來源可以避免公開 Seed 重建 Noise，但不等於已完成浮點 Gaussian 的安全認證。
- 已加噪 Voting 分數及選樣結果同時輸出，必須納入發布稽核；外部分享前須核對會計保護邊界。
- 不要上傳 API Token、`.env`、私有照片、私有原始文字、模型權重或未經授權的大型資料集。

**專案現在的 DP 標示應為：**

> *Conditional privacy accounting and experimental Gaussian-noised voting. End-to-end `(ε,δ)`-DP has not yet been certified.*

參見 `docs/STATUS_AND_UPGRADE.md` 與 DP Audit 程式。

---

## 10. 接下來的里程碑與驗收條件

| 優先度 | 工作 | 負責方 | 驗收標準 |
|---|---|---|---|
| **P0** | 共同 Repo 安裝／Clone 驗證 | 兩人 | `--recurse-submodules`、Unit Test 與 Mock Pipeline 通過 |
| **P0** | 影像 API Random／Variation | API 組員 | 少量公開圖生成 PNG，保留相容介面與錯誤處理 |
| **P0** | 確定影像 Dataset／授權與 Metadata Schema | 核心組員 | Schema CSV + 使用範圍／授權說明 |
| **P1** | CLIP／Inception 本地 Embedding | 核心組員 | 真實影像向量長度、NN 分配與投票測試通過 |
| **P1** | AIM／Metadata 分布與來源 | 核心組員 | DP 合成 Metadata + 可查核會計參數 |
| **P1** | 真正 Image MAPLE 小型整合 | 共同 | 4–30 張生成、至少一輪 Evolution、無資料外洩 |
| **P2** | PE／SPTI／MAPLE Baseline | 共同 | 相同資料／前處理、API 呼叫量、Seed、ε |
| **P2** | FID／影像 Metadata JSD、Ablation | 共同 | 可重算評估表及完整實驗設定 |
| **P2** | 文字版 7B 及正式論文復現 | 視算力 | 7B Smoke、對齊作者設定與可核對比較 |

### 下一次小組會議前，雙方要交付

**API 組員**：一個可運作的 `generate(request)`、Random 與 Variation 各 1 張示例、使用的模型 ID、每次輸出成本／限制、PR。  
**核心組員**：一份影像 Metadata Schema 草案、授權可使用的 Demo 圖片、CLIP／Inception 的執行方案、PR。  
**共同**：同一個 Mock Test 在至少一台非原開發電腦上通過，並將錯誤／版本列在 Issue。

---

## 11. GitHub 合作規則

```text
main                      = 已通過基本測試的共同版本
feature/image-api         = 組員 A：生成 API
feature/image-metadata-voting = 組員 B：Metadata / 本地 Voting
feature/evaluation        = FID / JSD / 實驗比較
```

### 每天開工與提交

```bash
git switch main
git pull --ff-only origin main
git submodule update --init --recursive
git switch -c feature/your-task
```

完成後：

```bash
git add <你修改過的檔案或目錄>
git diff --cached --stat
python tools/check_staged_git.py
git commit -m "Describe completed work"
git push -u origin feature/your-task
```

在 GitHub 提 Pull Request，至少描述：改動目的、測試指令、實際結果、尚未解決的問題。確認相容後再合併至 `main`。

> `git add -A` 前先檢查 `.gitignore` 和 `git status`。`DPSDA/` 是 Submodule，不能將它當普通資料夾整包重新提交。`.venv/`、`results/`、`image_extension/data/`、API Token 等不應進 Git。

---

## 12. 已知問題與排除方法

| 情況 | 原因／處理 |
|---|---|
| `DPSDA` Clone 後為空 | `git submodule update --init --recursive` |
| 影像版無法 import `image_maple` | 在 Repo 根目錄執行 `python -m pip install -e ./image_extension` |
| `TeammateBackend.generate` 提示 `NotImplementedError` | 正常：組員尚未接上真實 API；先用 `--backend mock` |
| 收到 `--allow-paid-api` 檢查 | 真實 API 呼叫必須明確啟用並設定數量上限 |
| GPU OOM | 模型或 Batch 太大；改用較小模型、較短輸出、GPU 加大或單獨檢查量化 |
| Hugging Face Windows symlink warning | 常見快取警告，一般不妨礙模型生成 |
| DP Accountant 顯示 `CONDITIONAL_ONLY...` | 正常：AIM 成本與端到端證明尚未完成，**不能標為 DP-certified** |
| MAUVE 有 FAISS 小樣本 Warning | 小樣本評估不穩；不得從 30 筆結論推廣至正式實驗 |
| API 產生空白／被拒絕圖片 | Provider 應回傳明確錯誤，記錄拒絕率並停止／重試有限次 |
| `.gitignore` 看似正確但資料已上傳 | `.gitignore` 不會移除已追蹤檔案，需檢查 `git ls-files` 與 Git 歷史；必要時輪替外洩金鑰 |

---

## 參考與詳細文件

- [API 組員工作契約](docs/TEAM_API_CONTRACT.md)
- [模型與版本升級](docs/STATUS_AND_UPGRADE.md)
- [完整 Repository GitHub 指南](docs/FULL_REPO_GITHUB_GUIDE.md)
- [影像擴展模組 README](image_extension/README.md)
- [原始 MAPLE Repository](https://github.com/elichien-google/MAPLE)
- [DPSDA Repository](https://github.com/microsoft/DPSDA)

**文件維護原則：** 每完成一個 PR，就更新第 2 節「目前進度」、第 10 節「下一步與驗收」。不要只更新程式卻讓 README 停留在過期狀態。
