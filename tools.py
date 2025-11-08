# tools.py
"""
Utility functions for email dataset preparation, analysis, plotting and modeling.

Designed to be imported into a Colab notebook:
from tools import (
    load_excel, load_json, prepare_df, plot_class_counts, plot_most_common_by_class,
    analyze_html_urls_length, unique_words_by_category, exclusive_words_by_category,
    train_tfidf_logreg, evaluate_model, save_predictions, top_tokens_per_class,
    predict_email
)
"""

from typing import List, Optional, Tuple, Dict, Any
import os
import re
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator

# ---------------- Defaults / small dictionaries ----------------
# Extend COMMON_NAMES if you want name-stripping
COMMON_NAMES: set = set()  # e.g. {"max", "mustermann", ...}

GERMAN_STOPWORDS = {
    "und","der","die","das","in","den","von","mit","ist","ich","nicht","zu","für","auf",
    "ein","eine","im","dem","an","als","auch","am","sind","hat","war","oder","wir",
    "sie","er","es","dass","mehr","bei"
}
STOP_WORDS = ENGLISH_STOP_WORDS.union(GERMAN_STOPWORDS)

# ---------------- I/O / Data preparation ----------------
def load_excel(path: str, sheet_name: Optional[str]=0) -> pd.DataFrame:
    """Load an Excel file into a DataFrame."""
    return pd.read_excel(path, sheet_name=sheet_name)

def load_json(path: str, orient: Optional[str] = None, lines: bool = False) -> pd.DataFrame:
    """Load a JSON file into a DataFrame."""
    return pd.read_json(path, orient=orient, lines=lines)

def prepare_df(
    df: pd.DataFrame,
    order: Optional[List[str]] = None,
    keep_cols: Optional[List[str]] = None,
    subject_col: str = "Betreff",
    text_col: str = "Nachrichtentext",
    combine_col_name: str = "Email"
) -> pd.DataFrame:
    """
    Prepare dataframe for downstream processing:
    - set categorical order for 'Klassen-Label' if order provided
    - limit columns to keep_cols (if provided)
    - create a combined 'Email' column from subject + text
    """
    df = df.copy()
    if keep_cols:
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = df[keep_cols + ["Klassen-Label"]] if "Klassen-Label" not in keep_cols else df[keep_cols]

    if order:
        df["Klassen-Label"] = pd.Categorical(df["Klassen-Label"], categories=order, ordered=True)

    # Ensure columns exist and combine subject+text into Email
    subj = df.get(subject_col, pd.Series([""] * len(df)))
    txt = df.get(text_col, pd.Series([""] * len(df)))
    df[combine_col_name] = "Betreff: " + subj.fillna("").astype(str) + " Nachrichtentext: " + txt.fillna("").astype(str)

    return df

# ---------------- small helpers ----------------
def choose_text_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of candidate text columns found. Tried: {candidates}")

def strip_common_german_names(text: str, common_names: Optional[set]=None) -> str:
    """Remove tokens matching common German first/last names (case-insensitive)."""
    if not text:
        return ""
    if common_names is None or len(common_names) == 0:
        return text
    tokens = re.findall(r"\b\w+\b", text)
    cleaned = [tok for tok in tokens if tok.lower() not in common_names]
    return " ".join(cleaned)

# ---------------- plotting: counts & top words ----------------
def plot_class_counts(
    df: pd.DataFrame,
    label_col: str = "Klassen-Label",
    order: Optional[List[str]] = None,
    figsize: Tuple[int,int]=(8,4),
    savepath: Optional[str]=None,
    show: bool=True
) -> None:
    """Plot bar chart with counts per class preserving provided order (if any)."""
    if order is not None:
        counts = df[label_col].value_counts().reindex(order).fillna(0)
    elif pd.api.types.is_categorical_dtype(df[label_col]):
        counts = df[label_col].value_counts().sort_index()
    else:
        counts = df[label_col].value_counts()
    plt.figure(figsize=figsize)
    counts.plot(kind="bar")
    plt.title("Anzahl gelabelte Emails pro Klasse")
    plt.xlabel("Klasse")
    plt.ylabel("Anzahl")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close()
    
# Example: your combined stopword set
GERMAN_STOPWORDS = {
    "und","der","die","das","in","den","von","mit","ist","ich","nicht","zu","für","auf",
    "ein","eine","im","dem","an","als","auch","am","sind","hat","war","oder","wir",
    "sie","er","es","dass","mehr","bei"
}

from sklearn.feature_extraction import text
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(GERMAN_STOPWORDS)

import re

def clean(df, column_name):
    """
    Clean a pandas DataFrame text column by removing:
    - English and German stopwords
    - Numbers (e.g., 123, 2023)
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the text column.
    column_name : str
        The name of the column to clean.
        
    Returns
    -------
    pandas.Series
        A cleaned text column with stopwords and numbers removed.
    """
    cleaned = (
        df[column_name]
        .astype(str)  # ensure string type
        .apply(lambda x: re.findall(r'\b\w+\b', x.lower()))  # tokenize by words
        .apply(
            lambda tokens: " ".join([
                w for w in tokens
                if w not in STOP_WORDS and not w.isdigit()  # remove stopwords and pure numbers
            ])
        )
    )
    return cleaned

def top_n_words_for_texts(texts: List[str], n: int=30, stop_words: Optional[set]=None) -> List[Tuple[str,int]]:
    vec = CountVectorizer(lowercase=True, token_pattern=r"\b\w{2,}\b", stop_words=stop_words)
    X = vec.fit_transform(texts)
    sums = X.sum(axis=0)
    counts = [(word, int(sums[0, idx])) for word, idx in vec.vocabulary_.items()]
    counts.sort(key=lambda x: x[1], reverse=True)
    return counts[:n]

def plot_most_common_by_class(
    df: pd.DataFrame,
    label_col: str = "Klassen-Label",
    text_candidates: List[str] = ["Email"],
    top_n: int = 30,
    stop_words: Optional[set] = None,
    common_names: Optional[set] = None,
    order: Optional[List[str]] = None,
    figsize_per_plot: Tuple[int,int]=(6,4),
    savepath: Optional[str]=None,
    show: bool=True
) -> None:
    """Plot top-n words per class in a grid, preserving the category order if supplied or categorical order."""
    text_col = choose_text_column(df, text_candidates)
    if pd.api.types.is_categorical_dtype(df[label_col]):
        classes = list(df[label_col].cat.categories)
    elif order:
        classes = order
    else:
        classes = list(dict.fromkeys(df[label_col].astype(str)))

    n_classes = len(classes)
    cols = 2
    rows = math.ceil(n_classes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_plot[0], rows * figsize_per_plot[1]))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_idx, cls in enumerate(classes):
        ax = axes_list[ax_idx]
        subset = df[df[label_col].astype(str) == cls]
        texts = subset[text_col].fillna("").astype(str).apply(lambda s: strip_common_german_names(s, common_names)).tolist()
        if not texts:
            ax.text(0.5, 0.5, f"No data for {cls}", ha="center")
            ax.set_title(str(cls))
            ax.axis("off")
            continue
        top = top_n_words_for_texts(texts, n=top_n, stop_words=stop_words)
        if not top:
            ax.text(0.5, 0.5, f"No tokens for {cls}", ha="center")
            ax.set_title(str(cls))
            ax.axis("off")
            continue
        words, counts = zip(*top)
        ax.bar(range(len(words)), counts)
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right")
        ax.set_title(f"{cls} — Top {len(words)} Wörter")
        ax.set_ylabel("count")
        ax.set_xlabel("word")

    # turn off any unused axes
    for j in range(len(classes), len(axes_list)):
        axes_list[j].axis("off")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

# ---------------- analyze HTML/URLs/length ----------------
HTML_RE = re.compile(r"<\/?[a-zA-Z][^>]*>")
URL_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+)", flags=re.IGNORECASE)

def contains_html(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(HTML_RE.search(text))

def contains_url(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(URL_RE.search(text))

def token_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())

def analyze_email_html_urls_length(
    df: pd.DataFrame,
    label_col: str = "Klassen-Label",
    text_candidates: List[str] = ["Email"],
    order: Optional[List[str]] = None,
    out_summary_csv: Optional[str] = "email_class_summary.csv",
    plot_prefix: Optional[str] = "email_class_",
    show: bool = True
) -> pd.DataFrame:
    """
    Compute per-class statistics about presence of HTML, URLs, char/token lengths.
    Returns summary DataFrame (ordered by 'order' if supplied).
    """
    text_col = choose_text_column(df, text_candidates)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")

    df2 = df.copy()
    df2["_text_clean"] = df2[text_col].fillna("").astype(str)
    df2["_has_html"] = df2["_text_clean"].apply(contains_html)
    df2["_has_url"] = df2["_text_clean"].apply(contains_url)
    df2["_char_len"] = df2["_text_clean"].apply(lambda s: len(s))
    df2["_token_len"] = df2["_text_clean"].apply(token_count)

    grouped = df2.groupby(label_col, sort=False)
    summary_rows = []
    for label, grp in grouped:
        n = len(grp)
        n_html = int(grp["_has_html"].sum())
        n_url = int(grp["_has_url"].sum())
        pct_html = 100.0 * n_html / n if n>0 else 0.0
        pct_url = 100.0 * n_url / n if n>0 else 0.0
        median_chars = float(grp["_char_len"].median()) if n>0 else 0.0
        median_tokens = float(grp["_token_len"].median()) if n>0 else 0.0
        mean_chars = float(grp["_char_len"].mean()) if n>0 else 0.0
        mean_tokens = float(grp["_token_len"].mean()) if n>0 else 0.0

        summary_rows.append({
            label_col: label,
            "n_emails": n,
            "n_with_html": n_html,
            "pct_with_html": round(pct_html, 2),
            "n_with_url": n_url,
            "pct_with_url": round(pct_url, 2),
            "median_chars": round(median_chars, 1),
            "median_tokens": round(median_tokens, 1),
            "mean_chars": round(mean_chars, 1),
            "mean_tokens": round(mean_tokens, 1)
        })

    summary_df = pd.DataFrame(summary_rows).set_index(label_col)
    if order is not None:
        summary_df = summary_df.reindex(order).reset_index()
    else:
        summary_df = summary_df.reset_index()

    if out_summary_csv:
        summary_df.to_csv(out_summary_csv, index=False, encoding="utf-8")

    # quick plots
    if show:
        plt.figure(figsize=(10,4))
        plt.bar(summary_df[label_col], summary_df["pct_with_html"])
        plt.xticks(rotation=45, ha="right"); plt.ylabel("Email-Anteil mit HTML-Tags (%)")
        plt.xlabel("")   
        plt.ylabel("")   
        plt.title("Email-Anteil mit HTML-Tags pro Klasse"); plt.tight_layout()
        if plot_prefix:
            plt.savefig(plot_prefix + "pct_html_by_class.png", bbox_inches="tight", dpi=150)
        plt.show()
        plt.close()

        plt.figure(figsize=(10,4))
        plt.bar(summary_df[label_col], summary_df["pct_with_url"])
        plt.xticks(rotation=45, ha="right"); plt.ylabel("Email-Anteil mit URLs (%)")
        plt.xlabel("")   
        plt.ylabel("")   
        plt.title("Email-Anteil mit URLs pro Klasse"); plt.tight_layout()
        if plot_prefix:
            plt.savefig(plot_prefix + "pct_url_by_class.png", bbox_inches="tight", dpi=150)
        plt.show()
        plt.close()

        plt.figure(figsize=(10,4))
        plt.bar(summary_df[label_col], summary_df["median_chars"])
        plt.xticks(rotation=45, ha="right"); plt.ylabel("Mediane Emaillänge")
        plt.xlabel("")   
        plt.ylabel("")   
        plt.title("Mediane Emaillänge in Zeichen pro Klasse"); plt.tight_layout()
        if plot_prefix:
            plt.savefig(plot_prefix + "median_chars_by_class.png", bbox_inches="tight", dpi=150)
        plt.show()
        plt.close()

    # cleanup temporary cols in returned df2 (not strictly necessary)
    df2.drop(columns=["_text_clean","_has_html","_has_url","_char_len","_token_len"], errors="ignore", inplace=True)
    return #summary_df

# ---------------- vocabulary helpers ----------------
def tokenize(text: str, stop_words: Optional[set]=None) -> List[str]:
    if not isinstance(text, str):
        return []
    tokens = re.findall(r"(?u)\b\w{2,}\b", text.lower())
    if stop_words is None:
        stop_words = STOP_WORDS
    return [t for t in tokens if t not in stop_words]

def unique_words_by_category(df: pd.DataFrame, text_col: str="Email", label_col: str="Klassen-Label") -> pd.DataFrame:
    results = []
    for label, group in df.groupby(label_col):
        tokens = []
        for text in group[text_col].dropna().astype(str):
            tokens.extend(tokenize(text))
        unique_words = set(tokens)
        results.append({
            label_col: label,
            "n_docs": len(group),
            "unique_words": len(unique_words)
        })
    summary = pd.DataFrame(results).sort_values("unique_words", ascending=False).reset_index(drop=True)
    return summary

def exclusive_words_by_category(df: pd.DataFrame, text_col: str="Email", label_col: str="Klassen-Label", top_n: int=20, seed: int=40) -> pd.DataFrame:
    random.seed(seed)
    class_vocab = {
        label: set(sum((tokenize(t) for t in group[text_col].dropna().astype(str)), []))
        for label, group in df.groupby(label_col)
    }
    exclusives = {}
    labels = list(class_vocab.keys())
    for label in labels:
        other_vocab = set().union(*(class_vocab[l] for l in labels if l != label))
        exclusives[label] = list(class_vocab[label] - other_vocab)
    
    print("\n=== Zufällig ausgewählte exklusive Wörter pro Klasse ===")
    for label, words in exclusives.items():
        n_words = len(words)
        sample_size = min(top_n, n_words)
        sampled = random.sample(words, sample_size) if n_words > 0 else []
        print(f"\n{label} ({n_words} exklusive Wörter insgesamt):")
        if sampled:
            print(", ".join(sampled))
        else:
            print("(Keine exklusiven Wörter gefunden)")

    # return a DataFrame summary (not printed)
    summary = pd.DataFrame([
        {label_col: label, "Anzahl verschiedener Wörter": len(class_vocab[label]), "Exklusive Wörter": len(exclusives[label])}
        for label in class_vocab
    ]).sort_values("Exklusive Wörter", ascending=False).reset_index(drop=True)
    return summary#, exclusives

# ---------------- modeling: stable mapping, train, eval ----------------
def build_label_mapping(df: pd.DataFrame, label_col: str="Klassen-Label", order: Optional[List[str]] = None) -> Tuple[Dict[str,int], Dict[int,str]]:
    """
    Build stable label->int and inverse mapping. If order supplied, use that order,
    otherwise preserve first appearance order in dataframe.
    """
    if order is None:
        order = df[label_col].astype(str).drop_duplicates().tolist()
    mapping = {lab: i for i, lab in enumerate(order)}
    inverse_map = {i: lab for lab, i in mapping.items()}
    # sanity check
    unseen = [lab for lab in df[label_col].astype(str).unique() if lab not in mapping]
    if unseen:
        raise ValueError(f"Found labels in dataframe not present in mapping/order: {unseen}")
    return mapping, inverse_map

def train_tfidf_logreg(
    df: pd.DataFrame,
    text_col: str = "Email",
    label_col: str = "Klassen-Label",
    order: Optional[List[str]] = None,
    test_size: float = 0.3,
    cv_folds: int = 3,
    random_state: int = 40,
    param_grid: Optional[dict] = None,
    stop_words: Optional[set] = None
) -> Tuple[BaseEstimator, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Train a TF-IDF + LogisticRegression pipeline with grid search.
    Returns: best_model, X_test, y_test, X_train, y_train, class_names (preserved order)
    """
    if stop_words is None:
        stop_words = STOP_WORDS
    mapping, inverse_map = build_label_mapping(df, label_col=label_col, order=order)
    class_names = [inverse_map[i] for i in sorted(inverse_map.keys())]

    # map to ints
    y = df[label_col].astype(str).map(mapping).to_numpy(dtype=int)
    X = df[text_col].astype(str).values

    # stratify safety
    counts = pd.Series(y).value_counts()
    if counts.min() < 2 and test_size > 0:
        stratify_arg = None
    else:
        stratify_arg = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b", stop_words=stop_words, max_df=0.95, min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", multi_class="ovr", random_state=random_state))
    ])

    if param_grid is None:
        param_grid = {"clf__C": [0.1, 1.0, 5.0]}

    grid = GridSearchCV(pipeline, param_grid, cv=cv_folds, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    return best_model, X_test, y_test, X_train, y_train, class_names

def evaluate_model(
    best_model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    labels_for_cm: Optional[List[int]] = None,
    digits: int = 4,
    show_cm: bool = True,
    cm_figsize: Tuple[int,int] = (8,6),
    save_prefix: Optional[str] = None
) -> str:
    """Evaluate model and print classification report & confusion matrix.
    Returns classification report string."""
    if labels_for_cm is None:
        labels_for_cm = list(range(len(class_names)))
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, labels=labels_for_cm, target_names=class_names, digits=digits)
    print("\nClassification Report (test):")
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=labels_for_cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    if show_cm:
        plt.figure(figsize=cm_figsize)
        disp.plot(ax=plt.gca(), xticks_rotation=45, cmap="Blues")
        plt.title("Confusion matrix (test set)")
        plt.tight_layout()
        if save_prefix:
            plt.savefig(save_prefix + "confusion_matrix.png", bbox_inches="tight", dpi=150)
        plt.show()
        plt.close()
    return report

def save_predictions(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    inverse_map: Dict[int,str],
    out_path: str = "predictions_test.csv"
) -> None:
    """Save predictions DataFrame with text, true_label, pred_label."""
    pred_df = pd.DataFrame({
        "text": X_test,
        "true_label": [inverse_map[int(i)] for i in y_test],
        "pred_label": [inverse_map[int(i)] for i in y_pred]
    })
    pred_df.to_csv(out_path, index=False, encoding="utf-8")

def top_tokens_per_class(best_model: BaseEstimator, class_names: List[str], n_show: int = 15) -> Dict[str, List[str]]:
    """Return a dict mapping class_name -> top tokens from logistic regression coefficients."""
    tfidf = best_model.named_steps.get("tfidf")
    clf = best_model.named_steps.get("clf")
    if tfidf is None or clf is None or not hasattr(clf, "coef_"):
        return {}
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_
    results = {}
    for i, label in enumerate(class_names):
        row = coefs[i] if coefs.shape[0] == len(class_names) else coefs[0]
        top_idx = np.argsort(row)[-n_show:][::-1]
        top_tokens = [feature_names[j] for j in top_idx]
        results[label] = top_tokens
    return results

# ---------------- prediction helper ----------------
def predict_email(best_model: BaseEstimator, email_text: str, class_names: List[str]) -> Tuple[str, List[Tuple[str,float]]]:
    """
    Predict a single email text and return (predicted_label, sorted_prob_pairs).
    sorted_prob_pairs is list of tuples (label, prob) sorted desc by prob.
    """
    probs = best_model.predict_proba([email_text])[0]
    pred_class = int(best_model.predict([email_text])[0])
    pred_label = class_names[pred_class]
    prob_pairs = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
    return pred_label, prob_pairs

# ---------------- end of file ----------------
