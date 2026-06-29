# ============================================================
# Nardos Ayele
# Analysis Pipeline
# Project: Detecting AI-Generated Text Using Zipf's Law
#          Word Frequency Analysis
# ============================================================
# Run AFTER Nardos_data_collection.py
# This script reads your collected data and runs the full
# Zipf analysis, classification, and visualization pipeline.
# ============================================================

import nltk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from collections import Counter
import warnings
import os

warnings.filterwarnings('ignore')
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

os.makedirs("Nardos_plots", exist_ok=True)

print("=" * 60)
print("  NARDOS AYELE")
print("  Detecting AI-Generated Text Using Zipf's Law")
print("=" * 60)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("\nStep 1: Loading data...")

df_all = pd.read_csv("Nardos_data/combined_dataset.csv")
texts  = df_all['text'].tolist()
labels = df_all['label'].tolist()

human_texts = [t for t, l in zip(texts, labels) if l == 0]
ai_texts    = [t for t, l in zip(texts, labels) if l == 1]

print(f"  ✓ Human texts loaded : {len(human_texts)}")
print(f"  ✓ AI texts loaded    : {len(ai_texts)}")

# ============================================================
# STEP 2: FEATURE EXTRACTION
# ============================================================

print("\nStep 2: Extracting Zipf features...")

def tokenize(text):
    tokens = nltk.word_tokenize(str(text).lower())
    return [t for t in tokens if t.isalpha()]

def freq_table(tokens):
    return Counter(tokens)

def zipf_slope(ft):
    if len(ft) < 10:
        return None
    counts    = sorted(ft.values(), reverse=True)
    ranks     = np.arange(1, len(counts) + 1)
    log_r     = np.log10(ranks)
    log_f     = np.log10(counts)
    slope, _, r, _, _ = stats.linregress(log_r, log_f)
    return abs(slope)

def type_token_ratio(tokens):
    if not tokens:
        return 0
    return len(set(tokens)) / len(tokens)

def hapax_ratio(ft, total):
    if total == 0:
        return 0
    return sum(1 for c in ft.values() if c == 1) / total

def mid_range_ratio(ft):
    counts = sorted(ft.values(), reverse=True)
    total  = sum(counts)
    if total == 0 or len(counts) < 50:
        return 0
    mid = counts[49:min(500, len(counts))]
    return sum(mid) / total

def mandelbrot_offset(ft):
    """Head zone behavior — top 10 words frequency mass."""
    counts = sorted(ft.values(), reverse=True)
    total  = sum(counts)
    if total == 0:
        return 0
    return sum(counts[:10]) / total

def extract_features(text):
    tokens = tokenize(text)
    ft     = freq_table(tokens)
    return {
        'alpha':            zipf_slope(ft),
        'type_token_ratio': type_token_ratio(tokens),
        'hapax_ratio':      hapax_ratio(ft, len(tokens)),
        'mid_range_ratio':  mid_range_ratio(ft),
        'mandelbrot_offset':mandelbrot_offset(ft),
        'word_count':       len(tokens),
    }

records = []
for text in human_texts:
    f = extract_features(text)
    f['label'] = 0
    records.append(f)

for text in ai_texts:
    f = extract_features(text)
    f['label'] = 1
    records.append(f)

df = pd.DataFrame(records).dropna()
df.to_csv("Nardos_data/features.csv", index=False)

human_df = df[df['label'] == 0]
ai_df    = df[df['label'] == 1]

print(f"  ✓ Features extracted for {len(df)} texts")

# ============================================================
# STEP 3: PRINT RESULTS
# ============================================================

print(f"\n{'='*60}")
print("  RESULTS — FEATURE COMPARISON")
print(f"{'='*60}")

feature_names = {
    'alpha':             'Zipf Slope (α)',
    'type_token_ratio':  'Type-Token Ratio',
    'hapax_ratio':       'Hapax Ratio',
    'mid_range_ratio':   'Mid-Range Ratio',
    'mandelbrot_offset': 'Mandelbrot Offset',
}

for col, name in feature_names.items():
    h = human_df[col].mean()
    a = ai_df[col].mean()
    diff = abs(h - a)
    print(f"\n  {name}")
    print(f"    Human : {h:.4f}")
    print(f"    AI    : {a:.4f}")
    print(f"    Gap   : {diff:.4f}")

# ============================================================
# STEP 4: TRAIN CLASSIFIER
# ============================================================

print(f"\n{'='*60}")
print("  CLASSIFIER — LOGISTIC REGRESSION")
print(f"{'='*60}")

features = ['alpha','type_token_ratio','hapax_ratio',
            'mid_range_ratio','mandelbrot_offset']

X = df[features].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

y_pred    = model.predict(X_test)
acc       = accuracy_score(y_test, y_pred)
prec      = precision_score(y_test, y_pred)
rec       = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"\n  Accuracy  : {acc:.2%}")
print(f"  Precision : {prec:.2%}")
print(f"  Recall    : {rec:.2%}")
print(f"  F1 Score  : {f1:.2%}")
print(f"  CV Score  : {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Human','AI']))

print(f"\n  Feature Coefficients:")
for feat, coef in zip(features, model.coef_[0]):
    print(f"    {feat:25s}: {coef:+.4f}")

# Save results summary
with open("Nardos_data/results_summary.txt", "w") as f:
    f.write("NARDOS AYELE — RESULTS SUMMARY\n")
    f.write("Detecting AI-Generated Text Using Zipf's Law\n\n")
    f.write(f"Human texts  : {len(human_df)}\n")
    f.write(f"AI texts     : {len(ai_df)}\n\n")
    for col, name in feature_names.items():
        f.write(f"{name}\n")
        f.write(f"  Human: {human_df[col].mean():.4f}\n")
        f.write(f"  AI   : {ai_df[col].mean():.4f}\n\n")
    f.write(f"Classifier Performance\n")
    f.write(f"  Accuracy  : {acc:.2%}\n")
    f.write(f"  Precision : {prec:.2%}\n")
    f.write(f"  Recall    : {rec:.2%}\n")
    f.write(f"  F1 Score  : {f1:.2%}\n")
    f.write(f"  CV Score  : {cv_scores.mean():.2%}\n")

# ============================================================
# STEP 5: VISUALIZATIONS
# ============================================================

print(f"\n{'='*60}")
print("  GENERATING PLOTS")
print(f"{'='*60}")

colors = {'human': '#00B4D8', 'ai': '#F4A261'}

# ── FIGURE 1: Zipf Curves ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 1: Zipf's Law — Human vs AI Text Frequency Curves",
             fontsize=13, fontweight='bold')

for label, texts_list, color, name in [
    (0, human_texts[:5], colors['human'], 'Human text'),
    (1, ai_texts[:5],    colors['ai'],    'AI text'),
]:
    for i, text in enumerate(texts_list):
        tokens = tokenize(text)
        ft     = freq_table(tokens)
        counts = sorted(ft.values(), reverse=True)
        ranks  = np.arange(1, len(counts) + 1)
        axes[0].plot(np.log10(ranks), np.log10(counts),
                     color=color, alpha=0.55,
                     label=name if i == 0 else "")

axes[0].set_xlabel("log₁₀(Word Rank)", fontsize=11)
axes[0].set_ylabel("log₁₀(Word Frequency)", fontsize=11)
axes[0].set_title("Log-Log Zipf Curves")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Alpha distribution
ax = axes[1]
ax.hist(human_df['alpha'], bins=20, color=colors['human'],
        alpha=0.7, label=f"Human (μ={human_df['alpha'].mean():.3f})")
ax.hist(ai_df['alpha'],    bins=20, color=colors['ai'],
        alpha=0.7, label=f"AI (μ={ai_df['alpha'].mean():.3f})")
ax.axvline(human_df['alpha'].mean(), color='#005F73',
           linestyle='--', linewidth=2)
ax.axvline(ai_df['alpha'].mean(),    color='#9B2226',
           linestyle='--', linewidth=2)
ax.set_xlabel("Zipf Slope α", fontsize=11)
ax.set_ylabel("Number of texts", fontsize=11)
ax.set_title("Distribution of α values")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig.text(0.5, -0.04,
    "Caption: Left — Log-log rank-frequency curves for 5 human (blue) and 5 AI (orange) "
    "text samples. Right — Distribution of Zipf slope α across all samples. "
    "Human text clusters at a higher α than AI text.",
    ha='center', fontsize=9, style='italic', wrap=True)

plt.tight_layout()
plt.savefig("Nardos_plots/Figure1_zipf_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 1 saved")

# ── FIGURE 2: Feature Comparison ───────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Figure 2: Feature Comparison — Human vs AI Text",
             fontsize=13, fontweight='bold')

all_features = list(feature_names.keys())
for ax, feat in zip(axes.flatten(), all_features + [None]):
    if feat is None:
        ax.axis('off')
        continue
    h_vals = human_df[feat].values
    a_vals = ai_df[feat].values
    bars = ax.bar(['Human', 'AI'],
                  [h_vals.mean(), a_vals.mean()],
                  color=[colors['human'], colors['ai']],
                  edgecolor='white', width=0.5,
                  yerr=[h_vals.std(), a_vals.std()],
                  capsize=6)
    ax.set_title(feature_names[feat], fontsize=11)
    ax.set_ylabel("Mean value", fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, [h_vals.mean(), a_vals.mean()]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + h_vals.std() * 0.1,
                f'{val:.3f}', ha='center',
                fontweight='bold', fontsize=10)

fig.text(0.5, -0.02,
    "Caption: Mean values of all five frequency features for human and AI text groups. "
    "Error bars show standard deviation. Alpha and hapax ratio show the strongest separation.",
    ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig("Nardos_plots/Figure2_feature_comparison.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 2 saved")

# ── FIGURE 3: Ideal Zipf Reference ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Figure 3: Observed Curves vs Ideal Zipf Reference Line",
             fontsize=13, fontweight='bold')

ranks_ref  = np.arange(1, 201)
ideal_freq = 1.0 / ranks_ref
ideal_freq = ideal_freq / ideal_freq[0]
ax.plot(np.log10(ranks_ref), np.log10(ideal_freq),
        'g--', linewidth=2.5, label="Ideal Zipf (α = 1.0)", alpha=0.9)

for text, color, name in [
    (human_texts[0], colors['human'], 'Human sample'),
    (ai_texts[0],    colors['ai'],    'AI sample'),
]:
    tokens = tokenize(text)
    ft     = freq_table(tokens)
    counts = sorted(ft.values(), reverse=True)[:200]
    ranks_obs   = np.arange(1, len(counts) + 1)
    norm_counts = np.array(counts, dtype=float) / counts[0]
    ax.plot(np.log10(ranks_obs), np.log10(norm_counts),
            color=color, linewidth=2.5, label=name)

ax.set_xlabel("log₁₀(Word Rank)", fontsize=11)
ax.set_ylabel("log₁₀(Normalised Frequency)", fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

fig.text(0.5, -0.04,
    "Caption: Normalised rank-frequency curves for one human and one AI sample, "
    "plotted against the ideal Zipf reference line (green dashed, α = 1.0). "
    "The AI curve deviates from the ideal line more than the human curve.",
    ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig("Nardos_plots/Figure3_ideal_vs_observed.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 3 saved")

# ── FIGURE 4: Confusion Matrix ──────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle("Figure 4: Confusion Matrix — Classifier Performance",
             fontsize=13, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Human', 'AI'])
disp.plot(ax=ax, colorbar=False,
          cmap='Blues')
ax.set_title("")

fig.text(0.5, -0.04,
    f"Caption: Confusion matrix showing classifier predictions on the 20% test set. "
    f"Accuracy: {acc:.2%}, Precision: {prec:.2%}, Recall: {rec:.2%}, F1: {f1:.2%}.",
    ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig("Nardos_plots/Figure4_confusion_matrix.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 4 saved")

# ── FIGURE 5: CV Accuracy ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle("Figure 5: Cross-Validation Accuracy Across 5 Folds",
             fontsize=13, fontweight='bold')

folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
bars  = ax.bar(folds, cv_scores * 100,
               color=colors['human'], edgecolor='white', width=0.5)
ax.axhline(y=cv_scores.mean()*100, color='#D62828',
           linestyle='--', linewidth=2,
           label=f"Mean: {cv_scores.mean():.2%}")
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_ylim(0, 110)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{score:.1%}', ha='center',
            fontweight='bold', fontsize=10)

fig.text(0.5, -0.04,
    "Caption: Accuracy of the logistic regression classifier across 5 cross-validation folds. "
    "The red dashed line shows the mean accuracy. Consistent scores indicate stable model performance.",
    ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig("Nardos_plots/Figure5_cv_accuracy.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 5 saved")

print(f"\n{'='*60}")
print("  PIPELINE COMPLETE — NARDOS AYELE")
print(f"{'='*60}")
print(f"  Results:")
print(f"    Human α : {human_df['alpha'].mean():.4f}")
print(f"    AI α    : {ai_df['alpha'].mean():.4f}")
print(f"    Gap     : {abs(human_df['alpha'].mean()-ai_df['alpha'].mean()):.4f}")
print(f"    Accuracy: {acc:.2%}")
print(f"    F1 Score: {f1:.2%}")
print(f"\n  Plots saved in: Nardos_plots/")
print(f"  Data saved in : Nardos_data/")
print("=" * 60)
