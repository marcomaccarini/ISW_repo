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
git clone https://github.com/your-username/multi-llm-eval.git
cd multi-llm-eval
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
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

### Step 1: Prepare input CSV
The input file must have at least two columns:
- `question` → e.g. `Who wrote The Old Man and the Sea?`
- `answer`   → e.g. `Ernest Hemingway`

Your file (`book_writers_QA.csv`) also includes useful metadata (`pattern_id`, `predicate`, `sup`, `card_class`, `pr_quartile`) which the script uses for additional breakdowns.

### Step 2: Configure `eval_multimodel.py`
Edit the variables in `main()`:
- `INPUT_CSV = Path("./book_writers_QA.csv")`
- `OUTDIR = Path("./results_multimodel")`
- `HF_TOKEN = "your_huggingface_token_here"` (or set env var `HF_TOKEN`)

Check the `MODELS` dictionary for model IDs you have access to:
- **Llama**: e.g. `meta-llama/Llama-3.2-1B-Instruct`, `Llama-3.1-8B-Instruct`, `Llama-3.3-70B-Instruct`  
- **Gemma**: `google/gemma-2-2b-it`, `gemma-2-9b-it`, `gemma-2-27b-it`  
- **DeepSeek**: `deepseek-ai/deepseek-llm-1.3b-instruct`, `deepseek-llm-7b-instruct`, `deepseek-llm-67b-instruct`  

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

## Notes

- Large models (~70B) require GPUs with high VRAM. The script uses **8-bit quantization** (`bitsandbytes`) with CPU offload to reduce memory needs.  
- For smaller models or CPU-only runs, set `USE_8BIT = False` in `eval_multimodel.py`.  
- Hugging Face Hub authentication is required to download gated models (Meta LLaMA, Gemma). Generate a token at [Hugging Face settings](https://huggingface.co/settings/tokens).

---

## Project Structure

```
├── book_writers_QA.csv        # Input data (your QA dataset)
├── eval_multimodel.py         # Main evaluation script
├── analyze_results.py         # Analysis & chart generation
├── results_multimodel/        # Output CSVs & Parquet files
└── analysis_charts/           # Plots saved by analyze_results.py
```

---

## License
MIT (or any license you choose)
