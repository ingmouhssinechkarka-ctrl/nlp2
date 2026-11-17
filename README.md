# Métriques d'évaluation pour les tâches de NLP

**Auteur :** Mouhssine Chkarka — Master Data & IA, INPT (2026)
**Tâches choisies :** 1. Classification · 2. Étiquetage de séquences (NER) · 3. Traduction automatique

Ce dépôt contient trois scripts Python **exécutables et autonomes** : chacun illustre les
métriques d'une tâche et montre leurs limites sur des exemples choisis exprès. BLEU, chrF, TER
et le F1 au niveau entité sont **implémentés from scratch** pour expliquer les formules. Il n'y a ni
téléchargement ni GPU.

```
.
├── src/
│   ├── 1_classification_metrics.py       # Accuracy, P, R, F1, ROC-AUC, macro/micro
│   ├── 2_sequence_labeling_metrics.py    # NER : token accuracy vs F1 entité (CoNLL)
│   └── 3_machine_translation_metrics.py  # BLEU, chrF, TER
├── outputs/                              # figures générées (PNG)
├── presentation/
│   ├── Evaluation_Metrics_NLP_Mouhssine_Chkarka.pptx   # slides (notes orateur incluses)
│   └── SCRIPT_ORAL.md                    # texte minuté (< 7 min)
├── requirements.txt
└── README.md
```

## Installation et exécution

```bash
pip install -r requirements.txt          # numpy, scikit-learn, matplotlib
python src/1_classification_metrics.py
python src/2_sequence_labeling_metrics.py
python src/3_machine_translation_metrics.py
```

Testé avec Python 3.14, scikit-learn 1.9, numpy 2.5 et matplotlib 3.11. Chaque script s'exécute en moins de 5 secondes.

---

## 1. Classification — `src/1_classification_metrics.py`

| Métrique | Formule | Question à laquelle elle répond |
|---|---|---|
| Accuracy | (TP+TN)/N | Quelle part des prédictions est correcte ? |
| Precision | TP/(TP+FP) | Quand le modèle dit « positif », a-t-il raison ? |
| Recall | TP/(TP+FN) | Le modèle trouve-t-il tous les positifs ? |
| F1 | 2PR/(P+R) | Quel compromis entre précision et rappel (moyenne harmonique) ? |
| ROC-AUC | aire sous la courbe ROC | Le modèle classe-t-il bien les exemples, quel que soit le seuil ? |

**Le script en 3 parties**
- **A – Sentiment.** Le script entraîne TF-IDF (1-2 grammes) avec une régression logistique sur 16 phrases et teste sur 12.
  Il recalcule P, R et F1 **à la main** depuis la matrice de confusion et vérifie qu'on obtient les mêmes valeurs que `sklearn`.
  Résultat : Accuracy 0.917, Precision 1.00, Recall 0.833, F1 0.909, AUC 0.917.
  Il sauvegarde `outputs/classification_confusion_matrix.png`.
- **B – Paradoxe de l'accuracy.** Le jeu contient 950 ham et 50 spam. Un modèle qui répond toujours « ham » obtient
  **95 % d'accuracy mais un F1 spam de 0**. Le modèle réel obtient un F1 de 0.785.
- **C – Multi-classe.** Sur des intentions déséquilibrées, le **F1 micro vaut 0.91** et le **F1 macro 0.72**. Le macro
  donne le même poids à chaque classe, il fait donc ressortir la classe rare mal prédite.

Fonctions clés : `confusion_matrix`, `precision_score`, `recall_score`, `f1_score(average=…)`,
`roc_auc_score`, `classification_report`.

## 2. Étiquetage de séquences (NER) — `src/2_sequence_labeling_metrics.py`

Les étiquettes suivent le format **BIO** : `B-` marque le début d'une entité, `I-` l'intérieur, `O` un token hors entité.

| Fonction | Rôle |
|---|---|
| `extraire_entites(tags)` | Convertit une séquence BIO en ensemble de spans `(type, début, fin)`. Un `I-` orphelin ouvre une entité, comme dans `conlleval`/`seqeval`. |
| `prf(tp, fp, fn)` | Calcule Precision, Recall et F1 en protégeant la division par zéro. |

Les 4 phrases d'exemple illustrent chacune une erreur : **frontière** (« Mouhssine » au lieu de
« Mouhssine Chkarka »), **type** (Apple ORG→LOC), **entité manquée** (Morocco) et **entité inventée** (Python).

Résultats :
- Token accuracy = **0.84** (trompeuse, car la majorité des tokens sont `O`)
- F1 entité **strict** (type et frontières exacts) = **0.571** (TP=4, FP=3, FN=3)
- F1 entité **partiel** (chevauchement avec le bon type) = 0.714
- Le rapport par type (LOC, ORG, PER) indique où le modèle échoue.

Une erreur de type compte **double** : elle produit un FP et un FN.

## 3. Traduction automatique — `src/3_machine_translation_metrics.py`

| Fonction | Métrique | Détail |
|---|---|---|
| `bleu(cand, refs)` | BLEU (Papineni 2002) | Précision n-grammes 1→4 « clippée », moyenne géométrique et pénalité de brièveté `BP = exp(1 − r/c)`. Lissage +1 pour n>1 (phrase seule). Accepte plusieurs références. |
| `chrf(cand, ref)` | chrF (Popović 2015) | F-β (β=2) sur les n-grammes de caractères 1→6. Le rappel pèse 2×. |
| `ter(cand, ref)` | TER simplifié | Distance d'édition en mots divisée par la longueur de la référence (sans *shifts*). **Plus bas = mieux.** |

Le script évalue 5 traductions de la phrase « Le chat est assis sur le tapis rouge depuis ce matin. » :

| Système | BLEU | chrF | TER ↓ | Leçon |
|---|---|---|---|---|
| Quasi parfaite | 84.3 | 86.8 | 8.3 | — |
| Paraphrase correcte | 47.4 | 69.9 | 66.7 | BLEU pénalise les synonymes |
| Mots mélangés | 19.7 | 53.5 | 75.0 | p1 = 1.0, mais p3/p4 s'effondrent |
| Trop courte | 6.8 | 21.9 | 75.0 | La brevity penalty (0.14) écrase le score |
| Contresens | 53.3 | 68.2 | 25.0 | **Faux, mais BLEU plus élevé que la paraphrase** |

Avec une 2ᵉ référence, le BLEU de la paraphrase passe de 47 à 85. La figure est sauvegardée dans `outputs/mt_scores.png`.

> Pour un usage réel, on utilise `sacrebleu` (BLEU/chrF/TER standardisés) et des métriques neuronales
> (BERTScore, COMET), complétées par une évaluation humaine.

## Conclusion

| Tâche | Métrique à rapporter | Piège |
|---|---|---|
| Classification | F1, Recall, macro-F1, ROC-AUC | L'accuracy trompe quand les classes sont déséquilibrées |
| NER | F1 entité strict (CoNLL / seqeval) | La token accuracy surestime la qualité |
| Traduction | BLEU + chrF + COMET, plusieurs références | Un bon recouvrement de surface ne garantit pas le sens |
