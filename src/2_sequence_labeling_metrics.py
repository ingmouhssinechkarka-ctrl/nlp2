"""
Tâche 2 : ÉTIQUETAGE DE SÉQUENCES (NER au format BIO)
Métriques : Token accuracy vs Precision / Recall / F1 au niveau ENTITÉ
            (strict, style CoNLL / seqeval), rapport par type, correspondance partielle.
Auteur : Mouhssine Chkarka - Master Data & IA (INPT) 2026

Implémentation from scratch (aucune dépendance) — même logique que `seqeval`.
Exécution : python src/2_sequence_labeling_metrics.py
"""
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")


def extraire_entites(tags):
    """Convertit une séquence BIO en ensemble de spans (type, début, fin)."""
    entites, debut, typ = [], None, None
    for i, tag in enumerate(tags + ["O"]):                 # sentinelle de fin
        pref, _, t = tag.partition("-")
        fin_entite = pref in ("O", "B") or (pref == "I" and t != typ)
        if typ is not None and fin_entite:
            entites.append((typ, debut, i - 1))
            typ = None
        if pref == "B" or (pref == "I" and typ is None):   # I- orphelin = début
            debut, typ = i, t
    return set(entites)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# Données : 4 phrases, étiquettes de référence (gold) et prédictions d'un modèle
# Chaque phrase illustre un type d'erreur différent.
# ---------------------------------------------------------------------------
data = [
    ("Erreur de FRONTIÈRE",
     "Mouhssine Chkarka studies at INPT in Rabat".split(),
     ["B-PER", "I-PER", "O", "O", "B-ORG", "O", "B-LOC"],
     ["B-PER", "O",     "O", "O", "B-ORG", "O", "B-LOC"]),
    ("Erreur de TYPE",
     "Apple opened a store in Paris".split(),
     ["B-ORG", "O", "O", "O", "O", "B-LOC"],
     ["B-LOC", "O", "O", "O", "O", "B-LOC"]),
    ("Entité MANQUÉE (FN)",
     "Elon Musk visited Morocco last week".split(),
     ["B-PER", "I-PER", "O", "B-LOC", "O", "O"],
     ["B-PER", "I-PER", "O", "O",     "O", "O"]),
    ("Fausse ALERTE (FP)",
     "The new Python release is fast".split(),
     ["O", "O", "O",     "O", "O", "O"],
     ["O", "O", "B-ORG", "O", "O", "O"]),
]

tok_ok = tok_total = 0
tp = fp = fn = 0
par_type = defaultdict(lambda: [0, 0, 0])           # type -> [tp, fp, fn]
partiel_tp = 0

print("=" * 72)
print("NER — comparaison gold vs prédiction")
print("=" * 72)
for nom, mots, gold, pred in data:
    tok_ok += sum(g == p for g, p in zip(gold, pred))
    tok_total += len(gold)
    G, P = extraire_entites(gold), extraire_entites(pred)
    print(f"\n[{nom}]  {' '.join(mots)}")
    print("  gold :", sorted((t, " ".join(mots[a:b + 1])) for t, a, b in G))
    print("  pred :", sorted((t, " ".join(mots[a:b + 1])) for t, a, b in P))
    tp += len(G & P); fp += len(P - G); fn += len(G - P)
    for t, *_ in G & P: par_type[t][0] += 1
    for t, *_ in P - G: par_type[t][1] += 1
    for t, *_ in G - P: par_type[t][2] += 1
    # Correspondance partielle : même type + chevauchement
    for (tg, a, b) in G:
        if any(tp_ == tg and not (d > b or f < a) for tp_, d, f in P):
            partiel_tp += 1

n_gold = tp + fn
n_pred = tp + fp
print("\n" + "=" * 72)
print("RÉSULTATS")
print("=" * 72)
print(f"Token accuracy           = {tok_ok}/{tok_total} = {tok_ok / tok_total:.3f}   <- trompeur (beaucoup de 'O')")
p, r, f = prf(tp, fp, fn)
print(f"Entité STRICTE (CoNLL)   : TP={tp} FP={fp} FN={fn}")
print(f"   Precision = {p:.3f}   Recall = {r:.3f}   F1 = {f:.3f}")
pp, rp = partiel_tp / n_pred, partiel_tp / n_gold
fpart = 2 * pp * rp / (pp + rp)
print(f"Entité PARTIELLE (chevauchement + même type) : F1 = {fpart:.3f}")

print("\nRapport par type d'entité (strict) :")
print(f"{'type':6s} {'prec':>6s} {'rec':>6s} {'f1':>6s}")
for t in sorted(par_type):
    p_, r_, f_ = prf(*par_type[t])
    print(f"{t:6s} {p_:6.3f} {r_:6.3f} {f_:6.3f}")

print(f"\nConclusion : {tok_ok}/{tok_total} tokens corrects ({100 * tok_ok / tok_total:.0f} %) "
      f"mais seulement {tp}/{n_gold} entités parfaitement reconnues."
      "\nEn NER, on rapporte le F1 au niveau ENTITÉ, pas l'accuracy par token.")
