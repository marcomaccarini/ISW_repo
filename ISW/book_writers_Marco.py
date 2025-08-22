#!/usr/bin/env python3
"""
Multi-family, multi-size LLM evaluation (no argparse), now testing BOTH prompt modes:
- "zero" (zero-shot)
- "few"  (few-shot with 2 demos)

For each (family × size × mode) the script:
1) Generates answers
2) Computes metrics
3) Saves per-model CSV + Parquet

Then it builds SIX comparison CSVs:
- comparison_small_zero.csv
- comparison_small_few.csv
- comparison_medium_zero.csv
- comparison_medium_few.csv
- comparison_large_zero.csv
- comparison_large_few.csv

Expected input CSV columns: 'question', 'answer'
Optional: 'pattern_id','predicate','sup','card_class','pr_quartile'
"""

from __future__ import annotations
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

# HF / Transformers
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# GPU_use = 7
# st = "cuda:" + str(GPU_use)
# torch.cuda.set_device(GPU_use)

from huggingface_hub import login

# ──────────────────────────────────────────────────────────
# Utilities & global lazy state
# ──────────────────────────────────────────────────────────

_smooth = SmoothingFunction().method1
_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
_bert_model = None  # lazy init

THRESHOLDS: Dict[str, float] = {
    'exact_match': 1.00,
    'edit_sim': 0.80,
    'jaccard_sim': 0.50,
    'rouge_l': 0.70,
    'bleu': 0.50,
    'bert_cosine': 0.85
}

ZERO_TPL = (
    "Answer the following question. Respond with just the author name(s), "
    "or 'unsure' if unknown. Do not include any other text.\n\n"
    "Question: {q}\nAnswer:"
)

FEW_TPL = (
    "Answer the following question with just the author name(s), or 'unsure' if unknown.\n"
    "Do not include any other text.\n\n"
    "Question: Who is the author of the book \"The Lost Throne\"?\nAnswer: Chris Kuzneski\n\n"
    "Question: Who are the authors of the book \"Dragons of Summer Flame\"?\nAnswer: Margaret Weis, Tracy Hickman\n\n"
    "Question: {q}\nAnswer:"
)


def setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=(logging.DEBUG if verbose else logging.INFO),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_nltk() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def detect_cuda_version() -> str | None:
    try:
        out = os.popen("nvidia-smi").read()
        m = re.search(r"CUDA Version: (\d+\.\d+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must contain 'question' and 'answer' columns.")
    return df


# ──────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────

def bleu_sim(pred: str, ref: str) -> float:
    return sentence_bleu([ref.split()], pred.split(), smoothing_function=_smooth)


def bert_sim(pred: str, ref: str) -> float:
    global _bert_model
    if _bert_model is None:
        _bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    a = _bert_model.encode(pred, convert_to_tensor=True)
    b = _bert_model.encode(ref, convert_to_tensor=True)
    return float(util.cos_sim(a, b).item())


def edit_sim(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if max(la, lb) == 0:
        return 1.0
    dp = np.zeros((la + 1, lb + 1), dtype=int)
    dp[:, 0] = np.arange(la + 1)
    dp[0, :] = np.arange(lb + 1)
    for i in range(1, la + 1):
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return 1 - dp[la, lb] / max(la, lb)


def jaccard_sim(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return 1.0 if not (sa or sb) else len(sa & sb) / len(sa | sb)


def exact_match(p: str, g: str) -> int:
    return int(p.strip().lower() == g.strip().lower())


def score_row(pred: str, gold: str) -> Dict[str, float | str]:
    em = exact_match(pred, gold)
    ed = edit_sim(pred, gold)
    ja = jaccard_sim(pred, gold)
    rl = _rouge.score(gold, pred)['rougeL'].fmeasure
    b4 = bleu_sim(pred, gold)
    bS = bert_sim(pred, gold)

    verdicts = {}
    for name, val in [
        ('exact_match', em),
        ('edit_sim', ed),
        ('jaccard_sim', ja),
        ('rouge_l', rl),
        ('bleu', b4),
        ('bert_cosine', bS)
    ]:
        if pred.strip().lower() == 'unsure':
            verdicts[f"verdict_{name}"] = 'unsure'
        else:
            verdicts[f"verdict_{name}"] = 'correct' if val >= THRESHOLDS[name] else 'hallucination'

    return {
        **verdicts,
        "exact_match": em,
        "edit_sim": ed,
        "jaccard_sim": ja,
        "rouge_l": rl,
        "bleu": b4,
        "bert_cosine": bS
    }


# ──────────────────────────────────────────────────────────
# LLM pipeline builder
# ──────────────────────────────────────────────────────────

def build_pipeline(model_id: str,
                   offload_dir: Path,
                   max_new_tokens: int,
                   temperature: float,
                   use_8bit: bool = True):
    if use_8bit and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested 8-bit loading for {model_id} but CUDA GPU is not available."
        )

    if use_8bit:
        cuda_ver = detect_cuda_version() or "?"
        logging.info("CUDA %s detected; loading %s in 8-bit with CPU offload", cuda_ver, model_id)
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        quant_cfg = bnb_cfg
    else:
        logging.info("Loading %s without 8-bit quantization", model_id)
        quant_cfg = None

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    safe_mkdir(offload_dir)

    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        device_map="auto",
        offload_folder=str(offload_dir),
    )

    gen = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tok,
        pad_token_id=tok.pad_token_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False if temperature == 0.0 else True,
        return_full_text=False,
    )

    # Sanity check
    try:
        test = gen("Who wrote The Old Man and the Sea?\nAnswer:", batch_size=1)[0]["generated_text"].strip()
        logging.info("Model smoke test → %s", test)
    except Exception as e:
        logging.warning("Smoke test failed for %s: %s", model_id, e)

    return gen


# ──────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────

def generate_answers(df_in: pd.DataFrame, tpl: str, gen_pipe, batch_size: int) -> List[str]:
    prompts = [tpl.format(q=q) for q in df_in["question"].astype(str).tolist()]
    answers: List[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i: i + batch_size]
        outs = gen_pipe(batch, batch_size=len(batch))
        answers.extend([o[0]["generated_text"].split("\n", 1)[0].strip() for o in outs])
    return answers


def evaluate_single_model(df: pd.DataFrame,
                          model_id: str,
                          size_tier: str,
                          family: str,
                          outdir: Path,
                          gen_pipe,
                          batch_size: int,
                          prompt_mode: str) -> Path:
    """
    Runs generation + scoring for one model in one prompt mode and saves a per-model-per-mode CSV.
    Returns path to saved CSV.
    """
    tpl = FEW_TPL if prompt_mode == "few" else ZERO_TPL
    logging.info("Evaluating %s [%s / %s] with %s prompt...", model_id, family, size_tier, prompt_mode)

    answers = generate_answers(df, tpl, gen_pipe, batch_size=batch_size)

    recs = []
    df_loc = df.copy()
    df_loc["model_answer"] = answers
    for row in tqdm(df_loc.itertuples(index=False), total=len(df_loc), desc="Scoring"):
        p = str(getattr(row, "model_answer")).strip()
        g = str(getattr(row, "answer")).strip()
        s = score_row(p, g)
        recs.append({**row._asdict(), **s})

    scored = pd.DataFrame(recs)
    model_tag = sanitize_tag(family + "_" + size_tier + "_" + prompt_mode)
    save_csv = outdir / f"{model_tag}.csv"
    save_parq = outdir / f"{model_tag}.parquet"
    scored.to_csv(save_csv, index=False)
    scored.to_parquet(save_parq, index=False)
    logging.info("Saved → %s (and Parquet)", save_csv)
    return save_csv


def sanitize_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def build_size_comparison(df_gold: pd.DataFrame,
                          per_model_csv_by_mode: Dict[Tuple[str, str, str], Path],
                          outdir: Path) -> None:
    """
    Creates 6 CSVs: compare size tiers across families for BOTH modes.
    For each tier in [small, medium, large] and mode in [zero, few],
    produces a CSV with: question, gold answer, answers from Llama/Gemma/DeepSeek, and exact_match flags.
    """
    for mode in ["zero", "few"]:
        for tier in ["small", "medium", "large"]:
            frames = []
            for family in ["llama", "gemma", "deepseek"]:
                key = (family, tier, mode)
                if key not in per_model_csv_by_mode:
                    logging.warning("Missing results for %s %s (%s); it will be blank in comparison.", family, tier,
                                    mode)
                    continue
                dfm = pd.read_csv(per_model_csv_by_mode[key])
                keep_cols = ["question", "answer", "model_answer", "exact_match"]
                missing = [c for c in keep_cols if c not in dfm.columns]
                if missing:
                    logging.warning("Model CSV %s missing columns %s; skipping.", per_model_csv_by_mode[key], missing)
                    continue
                sub = dfm[keep_cols].copy()
                sub = sub.rename(columns={
                    "model_answer": f"{family}_answer",
                    "exact_match": f"{family}_exact_match"
                })
                frames.append(sub)

            if not frames:
                logging.warning("No frames for tier=%s mode=%s; skipping comparison build.", tier, mode)
                continue

            merged = df_gold[["question", "answer"]].copy()
            for f in frames:
                merged = merged.merge(f, on=["question", "answer"], how="left")

            out_csv = outdir / f"comparison_{tier}_{mode}.csv"
            merged.to_csv(out_csv, index=False)
            logging.info("Comparison for %s / %s saved → %s", tier.upper(), mode.upper(), out_csv)


# ──────────────────────────────────────────────────────────
# MAIN (you edit variables here)
# ──────────────────────────────────────────────────────────

def main():
    setup_logging(verbose=True)
    ensure_nltk()

    # ==== YOU SET THESE ====
    INPUT_CSV = Path("/home/mmaccarini/Blerina/ISW/book_writers_QA_ALL.csv")  # your input QA file
    OUTDIR = Path("./results_multimodel")  # output directory
    HF_TOKEN = "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"

    # Generation params
    MAX_NEW_TOKENS = 32
    TEMPERATURE = 0.0
    BATCH_SIZE = 8
    USE_8BIT = True  # set False for CPU/smaller models if needed

    # Model matrix: fill with model IDs you can access.
    MODELS: Dict[str, Dict[str, str]] = {
        "llama": {
            "small": "meta-llama/Llama-3.2-1B-Instruct",
            "medium": "meta-llama/Llama-3.1-8B-Instruct",
            #            "large": "meta-llama/Llama-3.3-70B-Instruct",
        },
        "gemma": {
            "small": "google/gemma-2-2b-it",
            "medium": "google/gemma-2-9b-it",
            #            "large": "google/gemma-2-27b-it",
        },
        "deepseek": {
            "small": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "medium": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            #            "large": "deepseek-ai/deepseek-llm-67b-instruct",
        }
    }
    # ========================

    safe_mkdir(OUTDIR)
    df_gold = load_csv(INPUT_CSV)
    print("Loaded %d questions from %s", len(df_gold), INPUT_CSV)

    if HF_TOKEN:
        try:
            login(token=HF_TOKEN)
            print("Logged in to Hugging Face Hub.")
        except Exception as e:
            logging.warning("HF login failed: %s", e)

    # Track per-model-per-mode CSV paths
    per_model_csv_by_mode: Dict[Tuple[str, str, str], Path] = {}

    # Evaluate each family × size × mode
    for family, size_map in MODELS.items():
        for size_tier, model_id in size_map.items():
            tag = f"{family}/{size_tier}"

            # Build one pipeline per (family,size) and reuse for both modes
            offload_dir = OUTDIR / "offload" / sanitize_tag(f"{family}_{size_tier}")
            gen_pipe = build_pipeline(
                model_id=model_id,
                offload_dir=offload_dir,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                use_8bit=USE_8BIT
            )

            for prompt_mode in ["zero", "few"]:
                print("************************************************")
                print("----- " + tag + " -----")
                print("----- " + prompt_mode + " -----")
                print("************************************************")

                csv_path = evaluate_single_model(
                    df=df_gold,
                    model_id=model_id,
                    size_tier=size_tier,
                    family=family,
                    outdir=OUTDIR,
                    gen_pipe=gen_pipe,
                    batch_size=BATCH_SIZE,
                    prompt_mode=prompt_mode,
                )
                per_model_csv_by_mode[(family, size_tier, prompt_mode)] = csv_path

    # Build comparison CSVs for BOTH modes
    build_size_comparison(df_gold=df_gold,
                          per_model_csv_by_mode=per_model_csv_by_mode,
                          outdir=OUTDIR)

    print("All done — outputs are in %s", OUTDIR.resolve())


from datetime import datetime

if __name__ == "__main__":
    start = datetime.now()
    print(start)
    main()
    print(datetime.now() - start)
