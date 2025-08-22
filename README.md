# Multi-Family LLM QA Evaluation & Analysis

This project evaluates **question–answer tasks** (author-name QA) using multiple families of large language models (LLMs).  
It compares **Llama**, **Gemma**, and **DeepSeek** models at three different size tiers:

- **Small** (~1–2B parameters)  
- **Medium** (~7–9B parameters)  
- **Large** (~65–70B parameters)

Each model is tested with **two prompt modes**:
- **Zero-shot** (`zero`) → model sees only the question  
- **Few-shot** (`few`) → model sees 2 worked examples before the question  

The pipeline:
1. **Generates answers** with each model.  
2. **Computes similarity metrics** (Exact match, Edit similarity, Jaccard, ROUGE-L, BLEU, BERT cosine).  
3. **Saves per-model results** (`llama_small_zero.csv`, `gemma_large_few.csv`, etc.).  
4. **Builds comparison CSVs** (small/medium/large × zero/few).  
5. Provides an **analysis script** to aggregate results and produce comparison **charts**.

---

## Installation

### 1. Clone repo
```bash
git clone git@github.com:marcomaccarini/ISW_repo.git
cd ISW_repo
```

### 2. Create a virtual environment (recommended)
```bash
conda create -n reasoning 
conda activate reasoning
conda install pip
```

### 3. Install dependencies
Install PyTorch (adjust CUDA version as needed):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install project dependencies:
```bash
pip install   transformers   sentence-transformers   bitsandbytes   accelerate   scikit-learn   rouge-score   nltk   pandas   numpy   tqdm   seaborn   matplotlib   huggingface-hub
```

---

## Usage


### Step 1: Configure `eval_multimodel.py`
Edit the variables in `main()`:
- `INPUT_CSV = Path("./book_writers_QA.csv")`
- `OUTDIR = Path("./results_multimodel")`
- `HF_TOKEN = "your_huggingface_token_here"` (or set env var `HF_TOKEN`)

Check the `MODELS` dictionary for model IDs you have access to:
- **Llama**: e.g. `meta-llama/Llama-3.2-1B-Instruct`, `Llama-3.1-8B-Instruct`,  (removed `Llama-3.3-70B-Instruct`)
- **Gemma**: `google/gemma-2-2b-it`, `gemma-2-9b-it`, (removed `gemma-2-27b-it`)
- **DeepSeek**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `DeepSeek-R1-0528-Qwen3-8B`

### Step 3: Run evaluation
```bash
python eval_multimodel.py
```

This will generate:
- Per-model results: `llama_small_zero.csv`, `gemma_medium_few.csv`, etc.  
- Comparison files:
  - `comparison_small_zero.csv`
  - `comparison_small_few.csv`
  - `comparison_medium_zero.csv`
  - `comparison_medium_few.csv`
  - `comparison_large_zero.csv`
  - `comparison_large_few.csv`

### Step 4: Run analysis & generate charts
```bash
python analyze_results.py
```

This creates:
- `summary_metrics.csv` → average metrics per model.  
- Charts in `./analysis_charts/`, including:
  - `bar_exact_match.png` → accuracy per family & size  
  - `mode_bleu.png` → effect of zero vs few prompts  
  - `scatter_bleu_vs_bert.png` → correlation between BLEU and BERT cosine  

---
