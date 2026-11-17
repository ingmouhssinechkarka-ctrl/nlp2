"""
Tâche 3 : TRADUCTION AUTOMATIQUE (FR -> EN)
Métriques : BLEU (n-grammes + brevity penalty), chrF (n-grammes de caractères),
            TER (Translation Edit Rate, version sans « shifts »).
Auteur : Mouhssine Chkarka - Master Data & IA (INPT) 2026

Implémentation from scratch avec numpy uniquement.
Exécution : python src/3_machine_translation_metrics.py
"""
import sys
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)


def ngrams(seq, n):
    return Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))


# ------------------------------- BLEU --------------------------------------
def bleu(candidat, references, max_n=4, details=False):
    """BLEU phrase (Papineni et al., 2002) avec lissage +1 (Lin & Och) pour n>1."""
    c = candidat.lower().split()
    refs = [r.lower().split() for r in references]
    precisions = []
    for n in range(1, max_n + 1):
        cand = ngrams(c, n)
        max_ref = Counter()
        for r in refs:
            max_ref |= ngrams(r, n)                     # comptes max sur les références
        clip = sum(min(cnt, max_ref[g]) for g, cnt in cand.items())   # précision « clippée »
        total = max(sum(cand.values()), 1)
        if n > 1:                                       # lissage pour phrases courtes
            clip, total = clip + 1, total + 1
        precisions.append(clip / total)
    # Brevity penalty : longueur de référence la plus proche
    r_len = min((abs(len(r) - len(c)), len(r)) for r in refs)[1]
    bp = 1.0 if len(c) > r_len else math.exp(1 - r_len / max(len(c), 1))
    score = bp * math.exp(sum(math.log(max(p, 1e-9)) for p in precisions) / max_n)
    if details:
        return score, precisions, bp
    return score


# ------------------------------- chrF --------------------------------------
def chrf(candidat, reference, max_n=6, beta=2):
    """chrF (Popović, 2015) : F-beta sur n-grammes de caractères (recall pèse 2x)."""
    c = candidat.lower().replace(" ", "")
    r = reference.lower().replace(" ", "")
    P, R = [], []
    for n in range(1, max_n + 1):
        cn, rn = ngrams(c, n), ngrams(r, n)
        inter = sum((cn & rn).values())
        if sum(cn.values()) and sum(rn.values()):
            P.append(inter / sum(cn.values()))
            R.append(inter / sum(rn.values()))
    p, rr = sum(P) / len(P), sum(R) / len(R)
    if p + rr == 0:
        return 0.0
    return (1 + beta ** 2) * p * rr / (beta ** 2 * p + rr)


# ------------------------------- TER ---------------------------------------
def ter(candidat, reference):
    """TER simplifié = distance d'édition en mots / longueur référence (plus bas = mieux)."""
    c, r = candidat.lower().split(), reference.lower().split()
    d = list(range(len(r) + 1))
    for i in range(1, len(c) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(r) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (c[i - 1] != r[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(r)] / len(r)


# ---------------------------------------------------------------------------
source = "Le chat est assis sur le tapis rouge depuis ce matin."
reference = "the cat has been sitting on the red mat since this morning"

systemes = {
    "A. Quasi parfaite":      "the cat has been sitting on the red mat since morning",
    "B. Paraphrase correcte": "since this morning the cat has been seated on the red rug",
    "C. Mots mélangés":       "mat the red cat on sitting been has the this since morning",
    "D. Trop courte":         "the cat is sitting",
    "E. Contresens":          "the dog has been eating the red mat since this morning",
}

print("=" * 78)
print(f"Source    : {source}\nRéférence : {reference}")
print("=" * 78)
print(f"{'Système':24s} {'BLEU':>6s} {'chrF':>6s} {'TER↓':>6s}   p1    p2    p3    p4    BP")
res = {}
for nom, hyp in systemes.items():
    b, ps, bp = bleu(hyp, [reference], details=True)
    c, t = chrf(hyp, reference), ter(hyp, reference)
    res[nom] = (b, c, t)
    print(f"{nom:24s} {100*b:6.1f} {100*c:6.1f} {100*t:6.1f}  "
          + " ".join(f"{p:.2f}" for p in ps) + f"  {bp:.2f}")
    print(f"{'':24s} « {hyp} »")

print("\nObservations :")
print(" - B est une BONNE traduction mais BLEU la pénalise (synonymes : seated/rug).")
print("   chrF est plus indulgent (correspondances partielles de caractères).")
print(" - C a les bons mots (p1 = 1.00) mais l'ordre est faux : p3/p4 s'effondrent.")
print(" - D : précision correcte mais la brevity penalty (BP) écrase le score.")
print(" - E : un contresens (dog/eating) garde un BLEU élevé -> limite des métriques")
print("   lexicales ; d'où BERTScore / COMET (métriques neuronales) + évaluation humaine.")

# Plusieurs références : BLEU augmente pour la paraphrase
refs2 = [reference, "the cat has been seated on the red rug since this morning"]
print(f"\nAvec 2 références, BLEU(B) passe de {100*res['B. Paraphrase correcte'][0]:.1f} "
      f"à {100*bleu(systemes['B. Paraphrase correcte'], refs2):.1f}")

# Figure
noms = [n.split(". ")[1] for n in systemes]
x = range(len(noms))
fig, ax = plt.subplots(figsize=(8, 3.8))
w = 0.27
for k, (lab, col) in enumerate([("BLEU", "#1f6f8b"), ("chrF", "#99c24d"), ("TER (↓)", "#e4572e")]):
    ax.bar([i + (k - 1) * w for i in x], [100 * v[k] for v in res.values()], w, label=lab, color=col)
ax.set_xticks(list(x), noms, fontsize=9); ax.set_ylabel("score (%)"); ax.legend()
ax.set_title("Traduction FR→EN : BLEU vs chrF vs TER")
fig.tight_layout(); fig.savefig(OUT / "mt_scores.png", dpi=150)
print(f"\nFigure sauvegardée : {OUT / 'mt_scores.png'}")
