"""
Tâche 1 : CLASSIFICATION (analyse de sentiment + détection de spam)
Métriques : Accuracy, Precision, Recall, F1, matrice de confusion, ROC-AUC,
            macro vs micro vs weighted.
Auteur : Mouhssine Chkarka - Master Data & IA (INPT) 2026

Exécution : python src/1_classification_metrics.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             classification_report)

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)


def titre(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ---------------------------------------------------------------------------
# PARTIE A — Un vrai classifieur de sentiment (TF-IDF + Régression logistique)
# ---------------------------------------------------------------------------
train = [
    ("I love this movie, it is wonderful", 1), ("Great acting and a beautiful story", 1),
    ("Absolutely fantastic, I enjoyed every minute", 1), ("A masterpiece, brilliant and moving", 1),
    ("Very good film, highly recommended", 1), ("The best movie I have seen this year", 1),
    ("Excellent direction and great music", 1), ("Fun, smart and really enjoyable", 1),
    ("I hate this movie, it is terrible", 0), ("Boring story and bad acting", 0),
    ("Awful, a complete waste of time", 0), ("The worst film I have ever seen", 0),
    ("Very bad, I fell asleep", 0), ("Poor script and terrible music", 0),
    ("Disappointing and boring, not recommended", 0), ("Horrible acting, I want my money back", 0),
]
test = [
    ("A wonderful and brilliant film", 1), ("I really enjoyed the story", 1),
    ("Great music and excellent acting", 1), ("Not bad at all, quite good actually", 1),
    ("Recommended for the whole family", 1), ("The plot was predictable but fun", 1),
    ("Terrible and boring movie", 0), ("A waste of money, awful script", 0),
    ("I did not enjoy it, bad direction", 0), ("Not good, the acting was poor", 0),
    ("Long, slow and disappointing", 0), ("Great actors wasted on a bad script", 0),
]
X_tr, y_tr = zip(*train)
X_te, y_te = zip(*test)
y_te = np.array(y_te)

vec = TfidfVectorizer(ngram_range=(1, 2))
clf = LogisticRegression(C=10, max_iter=1000).fit(vec.fit_transform(X_tr), y_tr)
proba = clf.predict_proba(vec.transform(X_te))[:, 1]
y_pred = (proba >= 0.5).astype(int)

titre("PARTIE A — Sentiment (1 = positif, 0 = négatif)")
for (txt, y), p, pr in zip(test, y_pred, proba):
    ok = "OK " if y == p else "ERR"
    print(f"[{ok}] vrai={y} prédit={p} (p={pr:.2f})  {txt}")

# Calcul MANUEL depuis la matrice de confusion (pour expliquer les formules)
tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
acc = (tp + tn) / (tp + tn + fp + fn)
prec = tp / (tp + fp) if tp + fp else 0.0
rec = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

print(f"\nMatrice de confusion : TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"Accuracy  = (TP+TN)/N       = {acc:.3f}   | sklearn : {accuracy_score(y_te, y_pred):.3f}")
print(f"Precision = TP/(TP+FP)      = {prec:.3f}   | sklearn : {precision_score(y_te, y_pred):.3f}")
print(f"Recall    = TP/(TP+FN)      = {rec:.3f}   | sklearn : {recall_score(y_te, y_pred):.3f}")
print(f"F1        = 2PR/(P+R)       = {f1:.3f}   | sklearn : {f1_score(y_te, y_pred):.3f}")
print(f"ROC-AUC   (sur probabilités) = {roc_auc_score(y_te, proba):.3f}")
print("\n" + classification_report(y_te, y_pred, target_names=["négatif", "positif"], digits=3))

# Figure : matrice de confusion
cm = confusion_matrix(y_te, y_pred, labels=[1, 0])
fig, ax = plt.subplots(figsize=(4, 3.6))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        lab = [["TP", "FN"], ["FP", "TN"]][i][j]
        ax.text(j, i, f"{lab}\n{cm[i, j]}", ha="center", va="center", fontsize=13,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["prédit +", "prédit −"]); ax.set_yticks([0, 1], ["vrai +", "vrai −"])
ax.set_title("Matrice de confusion — sentiment")
fig.tight_layout(); fig.savefig(OUT / "classification_confusion_matrix.png", dpi=150)

# ---------------------------------------------------------------------------
# PARTIE B — Le paradoxe de l'accuracy (données déséquilibrées : spam)
# ---------------------------------------------------------------------------
titre("PARTIE B — Spam : 950 ham / 50 spam, modèle naïf « toujours ham »")
y_spam = np.array([0] * 950 + [1] * 50)          # 1 = spam
naif = np.zeros_like(y_spam)                     # prédit toujours « ham »
rng = np.random.default_rng(0)
bon = y_spam.copy()                              # modèle réaliste : quelques erreurs
bon[rng.choice(np.where(y_spam == 1)[0], 8, replace=False)] = 0   # 8 spams ratés
bon[rng.choice(np.where(y_spam == 0)[0], 15, replace=False)] = 1  # 15 fausses alertes

for nom, p in [("Naïf (toujours ham)", naif), ("Modèle réel", bon)]:
    print(f"{nom:22s} Accuracy={accuracy_score(y_spam, p):.3f}  "
          f"Precision={precision_score(y_spam, p, zero_division=0):.3f}  "
          f"Recall={recall_score(y_spam, p):.3f}  F1(spam)={f1_score(y_spam, p):.3f}")
print("-> 95 % d'accuracy pour un modèle qui ne détecte AUCUN spam : il faut regarder F1 / Recall.")

# ---------------------------------------------------------------------------
# PARTIE C — Multi-classe : macro vs micro vs weighted
# ---------------------------------------------------------------------------
titre("PARTIE C — Détection d'intention (3 classes déséquilibrées)")
y_true = ["météo"] * 70 + ["musique"] * 25 + ["alarme"] * 5
y_hat = ["météo"] * 68 + ["musique"] * 2 + ["musique"] * 22 + ["météo"] * 3 + ["météo"] * 4 + ["alarme"] * 1
for avg in ["micro", "macro", "weighted"]:
    print(f"F1 {avg:9s} = {f1_score(y_true, y_hat, average=avg):.3f}")
print("-> macro pénalise la classe rare 'alarme' ; micro ≈ accuracy.")
print(f"\nFigure sauvegardée : {OUT / 'classification_confusion_matrix.png'}")
