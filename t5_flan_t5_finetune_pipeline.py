r"""
T5-base fine-tuning pipeline (with LaTeX/math masking) for the abstract-summarization dataset.

EXPERIMENT OPTIONS (choose at start via menu / CLI / config below)
  A : Blend.   Train on all rows, stratified split by source, evaluate overall.
  B : Cross-generator. Train on ONE source, test on the OTHER source (needs a source_model column).
  C : Train on both, evaluate overall AND separately on the Gemini / Claude test subsets (needs source_model).

NEW: MATH HANDLING (see math_utils.py)
  Formulas like $C^{k+1}$ are replaced by placeholder tokens <M0>, <M1>, ... before training/generation and
  restored afterwards, because T5's vocabulary cannot represent characters such as { } \ ^ < ~.
  Results are scored on the RESTORED text and also split into "__math" / "__no_math" rows in the Summary sheet.

Expected data: CSV with columns  abstract, GENERATED (the summary), CATEGORIES   (+ optional source_model).
If there is no source_model column the data is treated as ONE source and option A is used automatically.
Column names are matched case-insensitively (aliases below).

Usage:
  python t5_finetune_pipeline.py                       # interactive menu
  python t5_finetune_pipeline.py --option A
"""
import argparse
import gc
import math
import os
import random
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

import math_utils
from math_utils import mask_math, restore_math

# =====================================================================
# 1) CONFIG
# =====================================================================
EXPERIMENT_OPTION = "A"            # "A" | "B" | "C"   (B/C need a source column; otherwise A is used)
B_DIRECTION = "both"
INTERACTIVE_MENU = True

# ---- data ----
COMBINED_FILE = r"D:/mini project/t5_arxiv_summarization_dataset.csv"
GEMINI_FILE = "t5_arxiv_summarization_dataset.csv"
CLAUDE_FILE = "t5_arxiv_summarization_claude.csv"
# column aliases (case-insensitive; first match wins)
ABSTRACT_COLS = ["abstract"]
SUMMARY_COLS = ["GENERATED", "generated_summary", "generated", "summary"]
SOURCE_COLS = ["source_model", "source", "model"]
DOMAIN_COLS = ["domain"]
CATEGORY_COLS = ["CATEGORIES", "categories"]     # used to derive domain when there is no domain column

# ---- math handling ----
MASK_MATH = True                   # replace formulas by <M#> placeholders (strongly recommended)
DROP_TRAIN_ROWS_WITH_NEW_FORMULAS = True   # skip TRAINING rows whose summary has a formula that is not in its abstract
EVAL_MATH_SPLIT = True             # also report scores separately for abstracts with / without math
math_utils.INLINE_SIMPLE = True    # $t$ -> t (plain text) instead of a placeholder

# ---- model / lengths ----
MODEL_NAME = "google/flan-t5-base"
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 128

# ---- training ----
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.15
MAX_GRAD_NORM = 1.0
MAX_EPOCHS = 10
PATIENCE = 3
MIN_DELTA = 1e-3
EARLY_STOP_METRIC = "val_loss"     # "val_loss" or "val_rougeL"
TRAIN_BATCH = 4
GRAD_ACCUM = 4
EVAL_BATCH = 16
GRADIENT_CHECKPOINTING = False

# ---- splits ----
VAL_FRAC = 0.10
TEST_FRAC = 0.10
SEED = 42
VAL_GEN_MAX = 300

# ---- final evaluation ----
NUM_BEAMS = 4
NO_REPEAT_NGRAM = 3
BERTSCORE_MODEL = "roberta-large"
BERTSCORE_BATCH = 16

# ---- output ----
RESULTS_XLSX = "flan_t5_results(increasedLR).xlsx"
OUT_DIR = "flan_t5_runs(increasedLR)"
SAVE_MODEL = True
SAVE_PREDICTIONS = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = USE_BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
PREFIX = "Summarize the following research abstract: "


# =====================================================================
# 2) UTILITIESs
# =====================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16)


def normalize_source(v):
    v = str(v).lower()
    if "gemini" in v:
        return "gemini"
    if "claude" in v or "haiku" in v:
        return "claude"
    return None


def make_key(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def find_col(df, names):
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


PHYSICS_PREFIXES = ("physics.", "hep-", "astro-ph", "cond-mat", "quant-ph", "nucl-", "nlin.")
PHYSICS_EXACT = {"gr-qc", "math-ph"}
EXCLUDED_CATEGORIES = {"physics.hist-ph", "physics.ed-ph", "physics.pop-ph", "math.ho", "cs.gl", "cs.cy"}


def classify_domain(categories):
    """Domain from the PRIMARY (first) arXiv category, e.g. 'cs.LG stat.ML' -> CS."""
    if not isinstance(categories, str) or not categories.strip():
        return "Unknown"
    primary = categories.split()[0].strip().lower()
    if primary in EXCLUDED_CATEGORIES:
        return "Other"
    if primary == "cs.sy" or primary.startswith("eess."):
        return "Engineering"
    if primary == "math.it" or primary.startswith("cs."):
        return "CS"
    if primary == "math.st" or primary.startswith("stat.") or primary.startswith("q-fin."):
        return "Stats"
    if primary.startswith("math."):
        return "Math"
    if primary in PHYSICS_EXACT or primary.startswith(PHYSICS_PREFIXES):
        return "Physics"
    return "Other"


def choose_settings():
    option, direction = EXPERIMENT_OPTION, B_DIRECTION
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["A", "B", "C"])
    parser.add_argument("--direction", choices=["gemini_to_claude", "claude_to_gemini", "both"])
    args, _ = parser.parse_known_args()

    if args.option:
        option = args.option
        direction = args.direction or direction
    elif INTERACTIVE_MENU:
        print("\nExperiment options:")
        print("  A) Blend: train on all, stratified split, overall metrics")
        print("  B) Cross-generator: train on one source, test on the other (needs source column)")
        print("  C) Train on both, evaluate overall + per source (needs source column)")
        ans = input(f"Choose option [A/B/C] (default {option}): ").strip().upper()
        option = ans if ans in ("A", "B", "C") else option
        if option == "B":
            print("  1) gemini_to_claude   2) claude_to_gemini   3) both")
            ans = input(f"Direction [1/2/3] (default {direction}): ").strip()
            direction = {"1": "gemini_to_claude", "2": "claude_to_gemini", "3": "both"}.get(ans, direction)
    return option, direction


# =====================================================================
# 3) DATA + MATH MASKING + SPLITS (grouped by abstract so none crosses splits)
# =====================================================================
def apply_math_masking(df):
    """Adds: abstract_raw / summary_raw (original LaTeX), abstract / summary (masked, model input/target),
    formulas (list per row), has_math, skip_train."""
    df = df.copy()
    df["abstract_raw"], df["summary_raw"] = df["abstract"], df["summary"]
    if not MASK_MATH:
        df["formulas"] = [[] for _ in range(len(df))]
        df["has_math"] = df["abstract_raw"].str.contains("$", regex=False)
        df["skip_train"] = False
        return df

    m_abs, m_sum, fmaps, skip, n_new, n_over = [], [], [], [], 0, 0
    for a, s in zip(df["abstract_raw"], df["summary_raw"]):
        ma, formulas, st_a = mask_math(a)
        ms, _, st_s = mask_math(s, formulas, allow_new=False)
        bad = st_a["overflow"] > 0 or (DROP_TRAIN_ROWS_WITH_NEW_FORMULAS and st_s["new"] > 0)
        n_new += st_s["new"] > 0
        n_over += st_a["overflow"] > 0
        m_abs.append(ma); m_sum.append(ms); fmaps.append(formulas); skip.append(bad)
    df["abstract"], df["summary"], df["formulas"], df["skip_train"] = m_abs, m_sum, fmaps, skip
    df["has_math"] = df["abstract_raw"].str.contains("$", regex=False)

    n_formulas = df["formulas"].map(len)
    print(f"[math] abstracts containing $: {df['has_math'].mean():.0%} | "
          f"rows with >=1 placeholder: {(n_formulas > 0).mean():.0%} | mean placeholders/abstract: {n_formulas.mean():.1f} | max: {n_formulas.max()}")
    print(f"[math] summaries with a formula NOT in their abstract: {n_new} | abstracts over {math_utils.MAX_PLACEHOLDERS} formulas: {n_over} "
          f"| rows excluded from TRAINING: {int(df['skip_train'].sum())}")
    return df


def load_data():
    if os.path.exists(COMBINED_FILE):
        df = pd.read_csv(COMBINED_FILE)
    elif os.path.exists(GEMINI_FILE) and os.path.exists(CLAUDE_FILE):
        g = pd.read_csv(GEMINI_FILE); g["source_model"] = "gemini"
        c = pd.read_csv(CLAUDE_FILE); c["source_model"] = "claude"
        df = pd.concat([g, c], ignore_index=True)
    else:
        raise SystemExit(f"Data not found. Put your CSV at '{COMBINED_FILE}'.")

    a_col, s_col = find_col(df, ABSTRACT_COLS), find_col(df, SUMMARY_COLS)
    if a_col is None or s_col is None:
        raise SystemExit(f"Need an abstract column {ABSTRACT_COLS} and a summary column {SUMMARY_COLS}. Found: {list(df.columns)}")
    src_col, dom_col, cat_col = find_col(df, SOURCE_COLS), find_col(df, DOMAIN_COLS), find_col(df, CATEGORY_COLS)

    out = pd.DataFrame({"abstract": df[a_col], "summary": df[s_col]})
    out["source"] = df[src_col].map(normalize_source) if src_col else "single"
    if dom_col:
        out["domain"] = df[dom_col]
    elif cat_col:
        out["domain"] = df[cat_col].map(classify_domain)
    else:
        out["domain"] = "Unknown"
    out = out.dropna(subset=["abstract", "summary", "source"]).copy()
    out["abstract"] = out["abstract"].astype(str).str.strip()
    out["summary"] = out["summary"].astype(str).str.strip()
    out = out[(out["abstract"] != "") & (out["summary"] != "")]
    out["key"] = out["abstract"].map(make_key)            # key from the ORIGINAL abstract
    out = out.drop_duplicates(subset=["key", "source"]).reset_index(drop=True)

    print(f"Loaded {len(out)} rows: {out['source'].value_counts().to_dict()} | domains: {out['domain'].value_counts().to_dict()}")
    return apply_math_masking(out)


def split_groups(df, val_frac, test_frac, seed):
    groups = df.groupby("key")["source"].agg(lambda s: "+".join(sorted(set(s)))).reset_index(name="strat")

    def _split(frame, frac):
        strat = frame["strat"] if frame["strat"].value_counts().min() >= 2 else None
        return train_test_split(frame, test_size=frac, random_state=seed, stratify=strat)

    rest, test = _split(groups, test_frac)
    train, val = _split(rest, val_frac / (1 - test_frac))
    pick = lambda g: df[df["key"].isin(set(g["key"]))].reset_index(drop=True)
    return pick(train), pick(val), pick(test)


def build_splits(df, option, direction):
    slices = {}
    if option in ("A", "C"):
        train, val, test = split_groups(df, VAL_FRAC, TEST_FRAC, SEED)
        eval_sets = {"test_overall": test}
        if option == "C":
            slices = {"test_gemini": ("test_overall", "gemini"), "test_claude": ("test_overall", "claude")}
        train_source = "+".join(sorted(df["source"].unique()))
    else:
        src, tgt = direction.split("_to_")
        src_df, tgt_df = df[df.source == src], df[df.source == tgt]
        train, val, test_in = split_groups(src_df, VAL_FRAC, TEST_FRAC, SEED)
        overlap = set(src_df["key"])
        tgt_clean = tgt_df[~tgt_df["key"].isin(overlap)].reset_index(drop=True)
        print(f"[B] Removed {len(tgt_df) - len(tgt_clean)} {tgt} rows whose abstract also exists in {src} data (leakage guard).")
        if len(tgt_clean) == 0:
            raise SystemExit("No cross-source test rows left after removing overlapping abstracts.")
        eval_sets = {f"in_source_test_{src}": test_in, f"cross_test_{tgt}": tgt_clean}
        train_source = src

    n_before = len(train)
    train = train[~train["skip_train"]].reset_index(drop=True)      # only TRAINING rows are filtered
    if n_before != len(train):
        print(f"[math] excluded {n_before - len(train)} training rows (formula not in abstract / too many formulas).")
    print(f"Split sizes -> train: {len(train)} | val: {len(val)} | "
          + " | ".join(f"{k}: {len(v)}" for k, v in eval_sets.items()))
    return train, val, eval_sets, slices, train_source


# =====================================================================
# 4) MODEL HELPERS
# =====================================================================
def make_collate(tok):
    def collate(batch):
        enc = tok([PREFIX + a for a, _ in batch], max_length=MAX_SOURCE_LEN,
                  truncation=True, padding=True, return_tensors="pt")
        lab = tok(text_target=[s for _, s in batch], max_length=MAX_TARGET_LEN,
                  truncation=True, padding=True, return_tensors="pt")["input_ids"]
        lab[lab == tok.pad_token_id] = -100
        enc["labels"] = lab
        return enc
    return collate


def pairs(df):
    return list(zip(df["abstract"], df["summary"]))


def shift_right(labels, start_id, pad_id):
    dec = labels.new_zeros(labels.shape)
    dec[:, 1:] = labels[:, :-1].clone()
    dec[:, 0] = start_id
    dec[dec == -100] = pad_id
    return dec


def ce_loss(logits, labels, smoothing=0.0):
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1),
                           ignore_index=-100, label_smoothing=smoothing)


@torch.no_grad()
def eval_loss(model, tok, df):
    model.eval()
    loader = DataLoader(pairs(df), batch_size=EVAL_BATCH, shuffle=False, collate_fn=make_collate(tok))
    start_id, pad_id = model.config.decoder_start_token_id, tok.pad_token_id
    total, ntok = 0.0, 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch.pop("labels")
        with autocast():
            out = model(**batch, decoder_input_ids=shift_right(labels, start_id, pad_id))
        n = (labels != -100).sum().item()
        total += ce_loss(out.logits, labels).item() * n
        ntok += n
    return total / max(ntok, 1)


@torch.no_grad()
def generate(model, tok, texts, num_beams, desc=""):
    """texts = MASKED abstracts. Returns MASKED summaries (call restore_math afterwards)."""
    model.eval()
    order = np.argsort([len(t) for t in texts])
    preds = [None] * len(texts)
    kwargs = dict(max_new_tokens=MAX_TARGET_LEN, num_beams=num_beams,
                  no_repeat_ngram_size=NO_REPEAT_NGRAM, use_cache=True)
    if num_beams > 1:
        kwargs["early_stopping"] = True
    for i in tqdm(range(0, len(texts), EVAL_BATCH), desc=desc, leave=False):
        idx = order[i:i + EVAL_BATCH]
        enc = tok([PREFIX + texts[j] for j in idx], max_length=MAX_SOURCE_LEN,
                  truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        with autocast():
            out = model.generate(**enc, **kwargs)
        # skip_special_tokens=True would delete <extra_id_N> placeholders, so strip only pad/eos manually
        decoded = tok.batch_decode(out, skip_special_tokens=False)
        for j, d in zip(idx, decoded):
            preds[j] = d.replace("<pad>", "").replace("</s>", "").strip()
    return preds


def report_unk(tok, df, n=500):
    """Shows how many <unk> tokens T5 produces for the raw LaTeX abstracts vs the masked ones."""
    sub = df[df["has_math"]].head(n)
    if len(sub) == 0:
        return
    unk, f = tok.unk_token_id, lambda texts: sum(ids.count(unk) for ids in tok(list(texts))["input_ids"])
    raw, masked = f(sub["abstract_raw"]), f(sub["abstract"])
    ln = lambda texts: np.mean([len(x) for x in tok(list(texts))["input_ids"]])
    print(f"[math] <unk> tokens in {len(sub)} math abstracts: raw LaTeX = {raw} -> masked = {masked} | "
          f"mean input length: {ln(sub['abstract_raw']):.0f} -> {ln(sub['abstract']):.0f} tokens")


# =====================================================================
# 5) METRICS
# =====================================================================
_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_bert = None


def rouge_scores(preds, refs):
    acc = {f"{k}_{m}": [] for k in ("rouge1", "rouge2", "rougeL") for m in ("f", "p", "r")}
    for p, r in zip(preds, refs):
        s = _rouge.score(r, p)
        for k in ("rouge1", "rouge2", "rougeL"):
            acc[f"{k}_f"].append(s[k].fmeasure)
            acc[f"{k}_p"].append(s[k].precision)
            acc[f"{k}_r"].append(s[k].recall)
    return {k: float(np.mean(v) * 100) for k, v in acc.items()}


def bert_scores(preds, refs):
    global _bert
    if _bert is None:
        from bert_score import BERTScorer
        _bert = BERTScorer(model_type=BERTSCORE_MODEL, lang="en", rescale_with_baseline=False,
                           device=DEVICE, batch_size=BERTSCORE_BATCH)
    P, R, F1 = _bert.score(preds, refs)
    return {"bertscore_p": float(P.mean() * 100), "bertscore_r": float(R.mean() * 100),
            "bertscore_f1": float(F1.mean() * 100)}


# =====================================================================
# 6) TRAINING
# =====================================================================
def train_model(tok, train_df, val_df):
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, dropout_rate=DROPOUT_RATE).to(DEVICE)

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    start_id, pad_id = model.config.decoder_start_token_id, tok.pad_token_id

    g = torch.Generator()
    g.manual_seed(SEED)
    loader = DataLoader(pairs(train_df), batch_size=TRAIN_BATCH, shuffle=True,
                        collate_fn=make_collate(tok), generator=g)

    # sanity check: loss of the untouched pretrained model on one batch (expect ~2-6, not hundreds)
    with torch.no_grad():
        model.eval()
        b = {k: v.to(DEVICE) for k, v in next(iter(loader)).items()}
        lab = b.pop("labels")
        init = ce_loss(model(**b, decoder_input_ids=shift_right(lab, start_id, pad_id)).logits, lab).item()
        print(f"[check] initial loss on one batch: {init:.3f}")
        model.train()

    no_decay = ("bias", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=LEARNING_RATE)
    total_updates = math.ceil(len(loader) / GRAD_ACCUM) * MAX_EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO * total_updates), total_updates)

    val_sub = val_df.sample(n=min(VAL_GEN_MAX, len(val_df)), random_state=SEED)
    best_score, best_state, best_epoch, bad, prev_val = -float("inf"), None, 0, 0, None
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        tot, ntok = 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}", leave=False)
        for step, batch in enumerate(pbar, 1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            with autocast():
                out = model(**batch, decoder_input_ids=shift_right(labels, start_id, pad_id))
            loss = ce_loss(out.logits, labels, LABEL_SMOOTHING)
            (loss / GRAD_ACCUM).backward()
            with torch.no_grad():
                ce = loss.item() if LABEL_SMOOTHING == 0 else ce_loss(out.logits.detach(), labels).item()
            n = (labels != -100).sum().item()
            tot += ce * n
            ntok += n
            if step % GRAD_ACCUM == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=round(tot / ntok, 4))

        train_loss = tot / ntok
        val_loss = eval_loss(model, tok, val_df)
        vpreds = generate(model, tok, val_sub["abstract"].tolist(), num_beams=1, desc="val-gen")
        vr = rouge_scores(vpreds, val_sub["summary"].tolist())      # masked vs masked (placeholders count as words)

        score = -val_loss if EARLY_STOP_METRIC == "val_loss" else vr["rougeL_f"]
        improved = score > best_score + MIN_DELTA
        if improved:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_minus_train_loss": round(val_loss - train_loss, 5),
            "val_loss_rising": bool(prev_val is not None and val_loss > prev_val),
            "val_rouge1": round(vr["rouge1_f"], 3),
            "val_rouge2": round(vr["rouge2_f"], 3),
            "val_rougeL": round(vr["rougeL_f"], 3),
            "learning_rate": sched.get_last_lr()[0],
            "is_best_epoch": improved,
            "epoch_time_sec": round(time.time() - t0, 1),
        })
        prev_val = val_loss
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_R1={vr['rouge1_f']:.2f} val_R2={vr['rouge2_f']:.2f} val_RL={vr['rougeL_f']:.2f}"
              f"{'  <- best' if improved else ''}")

        if bad >= PATIENCE:
            print(f"Early stopping: no improvement for {PATIENCE} epochs. Best epoch = {best_epoch}.")
            break

    model.load_state_dict(best_state)
    del opt, sched, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return model, history, best_epoch


# =====================================================================
# 7) FINAL EVALUATION (scored on RESTORED LaTeX text)
# =====================================================================
def evaluate_all(model, tok, eval_sets, slices, meta):
    done, pred_rows = {}, []
    for name, df in eval_sets.items():
        df = df.copy()
        masked_preds = generate(model, tok, df["abstract"].tolist(), NUM_BEAMS, desc=f"gen {name}")
        restored = [restore_math(p, f) for p, f in zip(masked_preds, df["formulas"])]
        df["prediction"] = [r[0] for r in restored]
        df["bad_placeholders"] = [r[1] for r in restored]
        done[name] = df
        for r in df.itertuples():
            pred_rows.append({"run_id": meta["run_id"], "eval_set": name, "source": r.source,
                              "domain": r.domain, "has_math": r.has_math, "abstract": r.abstract_raw,
                              "reference": r.summary_raw, "prediction": r.prediction})
    for name, (parent, src) in slices.items():
        done[name] = done[parent][done[parent]["source"] == src].copy()

    if EVAL_MATH_SPLIT:
        for name in list(done):
            if name in eval_sets:
                for tag, part in (("math", done[name][done[name]["has_math"]]),
                                  ("no_math", done[name][~done[name]["has_math"]])):
                    if len(part) >= 5:
                        done[f"{name}__{tag}"] = part.copy()

    rows = []
    for name, df in done.items():
        print(f"Scoring {name} ({len(df)} rows)...")
        preds, refs = df["prediction"].tolist(), df["summary_raw"].tolist()
        rows.append({**meta, "eval_set": name, "n_samples": len(df),
                     "pct_with_math": round(100 * df["has_math"].mean(), 1),
                     "invalid_placeholder_pct": round(100 * (df["bad_placeholders"] > 0).mean(), 2),
                     "test_loss": round(eval_loss(model, tok, df), 5),
                     **{k: round(v, 3) for k, v in {**rouge_scores(preds, refs), **bert_scores(preds, refs)}.items()}})
    return rows, pred_rows


def gap_table(summary_rows, run_id):
    in_row = next(r for r in summary_rows if r["eval_set"].startswith("in_source") and "__" not in r["eval_set"])
    x_row = next(r for r in summary_rows if r["eval_set"].startswith("cross_test") and "__" not in r["eval_set"])
    out = []
    for m in ("rouge1_f", "rouge2_f", "rougeL_f", "bertscore_f1", "test_loss"):
        a, b = in_row[m], x_row[m]
        out.append({"run_id": run_id, "metric": m, "in_source": a, "cross_source": b,
                    "difference_in_minus_cross": round(a - b, 4),
                    "relative_change_pct": round((a - b) / a * 100, 2) if a else None})
    return pd.DataFrame(out)


# =====================================================================
# 8) EXCEL OUTPUT (appends across runs)
# =====================================================================
SHEET_ORDER = ["Summary", "Training_Log", "Generalization_Gap", "Config", "Predictions"]


def append_to_excel(path, new_sheets):
    existing = pd.read_excel(path, sheet_name=None) if os.path.exists(path) else {}
    target = path
    for attempt in range(2):
        try:
            with pd.ExcelWriter(target, engine="openpyxl") as w:
                for name in SHEET_ORDER:
                    parts = [x for x in (existing.get(name), new_sheets.get(name)) if x is not None and len(x)]
                    if not parts:
                        continue
                    merged = pd.concat(parts, ignore_index=True)
                    merged.to_excel(w, sheet_name=name, index=False)
                    ws = w.sheets[name]
                    if name != "Predictions":
                        for col in ws.columns:
                            width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                            ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 45)
                    ws.freeze_panes = "A2"
            print(f"Results written to {os.path.abspath(target)}")
            return
        except PermissionError:
            target = path.replace(".xlsx", f"_{datetime.now():%H%M%S}.xlsx")
            print(f"'{path}' is open in Excel; saving to '{target}' instead.")


# =====================================================================
# 9) RUN ONE EXPERIMENT
# =====================================================================
def run_experiment(df, option, direction):
    set_seed(SEED)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}_opt{option}" + (f"_{direction}" if option == "B" else "")
    print(f"\n{'=' * 70}\nRUN {run_id}\n{'=' * 70}")

    train, val, eval_sets, slices, train_source = build_splits(df, option, direction)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    report_unk(tok, df) if MASK_MATH else None
    # Placeholders are T5's built-in <extra_id_N> tokens: already in the vocab with trained
    # embeddings, so NO add_tokens / resize_token_embeddings is needed.

    t0 = time.time()
    model, history, best_epoch = train_model(tok, train, val)
    train_minutes = (time.time() - t0) / 60

    meta = {"run_id": run_id, "option": option,
            "direction": direction if option == "B" else "-",
            "train_source": train_source, "n_train": len(train), "n_val": len(val),
            "best_epoch": best_epoch, "epochs_run": len(history)}
    rows, pred_rows = evaluate_all(model, tok, eval_sets, slices, meta)

    if SAVE_MODEL:
        run_dir = os.path.join(OUT_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)
        model.save_pretrained(run_dir)
        tok.save_pretrained(run_dir)

    cfg = {"run_id": run_id, "option": option, "direction": direction, "model": MODEL_NAME,
           "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
           "label_smoothing": LABEL_SMOOTHING, "dropout_rate": DROPOUT_RATE, "max_epochs": MAX_EPOCHS,
           "patience": PATIENCE, "early_stop_metric": EARLY_STOP_METRIC,
           "effective_batch": TRAIN_BATCH * GRAD_ACCUM, "max_source_len": MAX_SOURCE_LEN,
           "max_target_len": MAX_TARGET_LEN, "num_beams": NUM_BEAMS, "bertscore_model": BERTSCORE_MODEL,
           "bf16": USE_BF16, "seed": SEED, "train_minutes": round(train_minutes, 1),
           "mask_math": MASK_MATH, "max_placeholders": math_utils.MAX_PLACEHOLDERS,
           "inline_simple": math_utils.INLINE_SIMPLE}

    sheets = {
        "Summary": pd.DataFrame(rows),
        "Training_Log": pd.DataFrame([{"run_id": run_id, **h} for h in history]),
        "Config": pd.DataFrame([cfg]),
    }
    if SAVE_PREDICTIONS:
        sheets["Predictions"] = pd.DataFrame(pred_rows)
    if option == "B":
        sheets["Generalization_Gap"] = gap_table(rows, run_id)
        print("\nGeneralization gap (in-source vs cross-source):")
        print(sheets["Generalization_Gap"].to_string(index=False))

    append_to_excel(RESULTS_XLSX, sheets)

    show = ["eval_set", "n_samples", "pct_with_math", "invalid_placeholder_pct", "test_loss",
            "rouge1_f", "rouge2_f", "rougeL_f", "bertscore_f1"]
    print("\nFinal results:")
    print(sheets["Summary"][show].to_string(index=False))

    del model
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
# 10) MAIN
# =====================================================================
def main():
    option, direction = choose_settings()
    print(f"\nDevice: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "")
          + f" | bf16: {USE_BF16} | option: {option}" + (f" | direction: {direction}" if option == "B" else "")
          + f" | math masking: {MASK_MATH}")
    if DEVICE == "cpu":
        print("WARNING: no CUDA GPU detected; training T5-base on CPU will be extremely slow.")

    df = load_data()
    if option in ("B", "C") and df["source"].nunique() < 2:
        print(f"Option {option} needs two sources but the data has only {df['source'].unique().tolist()}; using option A.")
        option = "A"
    if option == "B":
        dirs = ["gemini_to_claude", "claude_to_gemini"] if direction == "both" else [direction]
        for d in dirs:
            run_experiment(df, "B", d)
    else:
        run_experiment(df, option, None)


if __name__ == "__main__":
    main()
