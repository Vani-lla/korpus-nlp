import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import torch
from datasets import Dataset
from seqeval.metrics import classification_report as span_report
from seqeval.metrics import f1_score as span_f1
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report as token_report
from sklearn.model_selection import train_test_split
from transformers import (
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

SEED = 69
TON = ["zwykły", "agresywny"]
TAGS = ["O", "B-Agresja", "I-Agresja", "B-Kontekst", "I-Kontekst"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    df = pd.DataFrame(json.load(open("annotations/rest-annotations.json")))
    df = df[["text", "ton", "spans"]].assign(
        label=lambda d: d.ton.map(TON.index),
        spans=lambda d: d.spans.map(lambda s: s if isinstance(s, list) else []),
    )
    rest, test = train_test_split(df, test_size=60, stratify=df.label, random_state=SEED)
    train, val = train_test_split(rest, test_size=60, stratify=rest.label, random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def bio_labels(enc, spans, special_ids):
    labels = []
    prev = None
    for tok_id, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
        if tok_id in special_ids or s == e:
            labels.append(-100)
            prev = None
            continue
        span = next((sp for sp in spans if s < sp["end"] and e > sp["start"]), None)
        if span is None:
            labels.append(0)
        else:
            tag = span["labels"][0]
            labels.append(TAGS.index(("I-" if prev is span else "B-") + tag))
        prev = span
    return labels


def seq_dataset(df, tok, suffix=""):
    ds = Dataset.from_pandas(df[["text", "label"]])
    return ds.map(lambda b: tok([t + suffix for t in b["text"]], truncation=True, max_length=256), batched=True, remove_columns=["text"])


def tok_dataset(df, tok, suffix=""):
    special = set(tok.all_special_ids)
    rows = []
    for text, spans in zip(df.text, df.spans):
        enc = tok(text + suffix, truncation=True, max_length=256, return_offsets_mapping=True)
        rows.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": bio_labels(enc, spans, special)})
    return Dataset.from_list(rows)


def seq_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = preds.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds), "f1": f1_score(p.label_ids, preds, average="macro")}


def decode_tags(logits, labels):
    preds = logits.argmax(-1)
    true = [[TAGS[l] for l in row if l != -100] for row in labels]
    pred = [[TAGS[p] for p, l in zip(prow, row) if l != -100] for prow, row in zip(preds, labels)]
    return true, pred


def tok_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    true, pred = decode_tags(logits, p.label_ids)
    return {"f1": span_f1(true, pred)}


def train(model, tok, train_ds, val_ds, token_level=False, epochs=8, lr=2e-4, name="model"):
    args = TrainingArguments(
        output_dir=f"outputs/{name}",
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_strategy="epoch",
        report_to="none",
        seed=SEED,
    )
    collator = DataCollatorForTokenClassification(tok) if token_level else DataCollatorWithPadding(tok)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=tok_metrics if token_level else seq_metrics,
    )
    trainer.train()
    return trainer


def show_spans(text, tok, model, suffix=""):
    enc = tok(text + suffix, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=256)
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        preds = model(**enc.to(model.device)).logits.argmax(-1)[0].tolist()
    spans, cur = [], None
    for (s, e), p in zip(offsets, preds):
        tag = TAGS[p]
        if s == e or tag == "O":
            cur = None
            continue
        if tag.startswith("B-") or cur is None or cur[2] != tag[2:]:
            cur = [s, e, tag[2:]]
            spans.append(cur)
        else:
            cur[1] = e
    return [(label, text[s:e].strip()) for s, e, label in spans if s < len(text)]


@torch.no_grad()
def embed(model, tok, texts, pool, suffix="", batch=32):
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok([t + suffix for t in texts[i : i + batch]], padding=True, truncation=True, max_length=256, return_tensors="pt").to(model.device)
        h = model(**enc, output_hidden_states=True).hidden_states[-1]
        if pool == "cls":
            out.append(h[:, 0])
        else:
            last = enc["attention_mask"].sum(1) - 1
            out.append(h[torch.arange(len(h)), last])
    return torch.cat(out).float().cpu().numpy()


def plot_space(emb, labels, preds, texts, title):
    pts = TSNE(2, perplexity=15, random_state=SEED).fit_transform(PCA(min(50, len(emb) - 1), random_state=SEED).fit_transform(emb))
    df = pd.DataFrame({
        "x": pts[:, 0],
        "y": pts[:, 1],
        "etykieta": [TON[l] for l in labels],
        "predykcja": [TON[p] for p in preds],
        "tekst": ["<br>".join(t[i : i + 80] for i in range(0, min(len(t), 320), 80)) for t in texts],
    })
    fig = px.scatter(df, x="x", y="y", color="etykieta", symbol="predykcja", hover_data={"tekst": True, "x": False, "y": False}, title=title)
    fig.update_traces(marker_size=10)
    Path("plots").mkdir(exist_ok=True)
    fig.write_html(f"plots/{title.replace(' ', '_').replace('/', '_')}.html")
    return fig


@torch.no_grad()
def predict(model, tok, texts, suffix="", batch=32):
    model.eval()
    preds = []
    for i in range(0, len(texts), batch):
        enc = tok([t + suffix for t in texts[i : i + batch]], padding=True, truncation=True, max_length=256, return_tensors="pt").to(model.device)
        preds.append(model(**enc).logits.argmax(-1).cpu())
    return torch.cat(preds).numpy()


def report_tokens(trainer, ds):
    p = trainer.predict(ds)
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    true, pred = decode_tags(logits, p.label_ids)
    flat = lambda tags: [t.split("-")[-1] for row in tags for t in row]
    return span_report(true, pred, digits=3), token_report(flat(true), flat(pred), digits=3)
