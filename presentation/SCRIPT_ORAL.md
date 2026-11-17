# Script oral : Métriques d'évaluation en NLP (≈ 6 min 50)

**Mouhssine Chkarka — Master Data & IA, INPT**

> Conseil d'enregistrement : garder un terminal ouvert à côté de PowerPoint. Sur les slides
> « Démo », basculer 15–20 s sur le terminal et lancer la commande indiquée.

---

### Slide 1 · Titre (0:00 → 0:20)
Bonjour, je suis Mouhssine Chkarka, étudiant en Master Data & IA. Ma présentation porte sur les
métriques d'évaluation pour trois tâches de NLP : la classification, l'étiquetage de séquences et
la traduction automatique. Pour chaque tâche, j'ai écrit un code exécutable que je vais vous montrer.

### Slide 2 · Fil conducteur (0:20 → 0:45)
Une métrique répond toujours à une question : qu'est-ce qu'on compte ? En classification, on compte
des exemples. En NER, on compte des entités complètes. En traduction, on compte les n-grammes
partagés avec des traductions de référence. Chaque choix a ses pièges, et je vais les montrer avec du code.

### Slide 3 · Classification : formules (0:45 → 1:45)
Commençons par la classification, par exemple l'analyse de sentiment ou la détection de spam. Tout part de
la matrice de confusion : vrais positifs, faux positifs, faux négatifs et vrais négatifs.
L'accuracy est la proportion de bonnes réponses. La **précision** répond à la question : quand le modèle
dit « positif », a-t-il raison ? Le **rappel** répond à : a-t-il trouvé tous les positifs ? Le **F1** est
la moyenne harmonique des deux, il n'est élevé que si la précision et le rappel le sont tous les deux. Le **ROC-AUC**
évalue les probabilités, quel que soit le seuil. En multi-classe, on agrège en micro, macro ou weighted.

### Slide 4 · Démo 1 (1:45 → 2:45)
`python src/1_classification_metrics.py`
Dans la partie A, j'entraîne un vrai classifieur TF-IDF avec une régression logistique. Je recalcule la précision, le rappel
et le F1 à la main depuis la matrice de confusion, et je retrouve exactement les valeurs de scikit-learn :
0.92 d'accuracy et 0.91 de F1.
La partie B contient le message clé. Sur un jeu de spam déséquilibré, un modèle naïf qui répond toujours
« ham » obtient **95 % d'accuracy… mais un F1 de zéro** sur le spam.
Dans la partie C, en multi-classe, le F1 micro vaut 0.91 mais le macro seulement 0.72, parce que le macro donne
le même poids à la classe rare.
→ Sur des données déséquilibrées, on regarde le F1, le rappel et le macro-F1, jamais l'accuracy seule.

### Slide 5 · NER : concept (2:45 → 3:35)
La deuxième tâche est l'étiquetage de séquences, ici la reconnaissance d'entités nommées au format BIO :
B marque le début d'une entité, I l'intérieur, O un mot hors entité. Sur cette phrase, le modèle ne se trompe
que sur un token sur sept, ce qui donne 86 % de token accuracy. Pourtant l'entité « Mouhssine Chkarka » est fausse,
car il n'a trouvé que « Mouhssine ». La norme CoNLL, implémentée par seqeval, ne compte une entité comme
correcte que si le type ET les deux frontières sont exacts. On distingue quatre erreurs : frontière, type,
entité manquée et entité inventée.

### Slide 6 · Démo 2 (3:35 → 4:25)
`python src/2_sequence_labeling_metrics.py`
J'ai implémenté l'extraction des entités depuis les tags BIO, avec la même logique que seqeval.
J'utilise quatre phrases, chacune avec un type d'erreur. La token accuracy est de **84 %**, alors que le F1 strict
au niveau entité n'est que de **57 %**. La version partielle, qui accepte un chevauchement avec le bon type,
donne 71 %. Point important : une erreur de type compte double, car elle produit un faux positif et un faux négatif.
→ En NER, on rapporte le F1 au niveau entité.

### Slide 7 · Traduction : métriques (4:25 → 5:20)
La troisième tâche est la traduction automatique. Il n'existe pas une seule bonne traduction, donc on compare la sortie à
des traductions de référence. **BLEU** calcule la précision des n-grammes de 1 à 4 mots. Cette précision est
« clippée » pour ne pas récompenser les répétitions. BLEU en prend la moyenne géométrique et la multiplie par une
pénalité de brièveté qui sanctionne les traductions trop courtes. **chrF** travaille sur des n-grammes de caractères, ce qui le rend
plus robuste aux variantes de mots. **TER** compte le nombre d'éditions nécessaires pour obtenir la
référence : plus il est bas, mieux c'est. Aujourd'hui, on complète ces métriques par des métriques neuronales comme COMET.

### Slide 8 · Démo 3 (5:20 → 6:20)
`python src/3_machine_translation_metrics.py`
BLEU, chrF et TER sont codés from scratch et testés sur cinq traductions de la même phrase.
La traduction quasi parfaite obtient 84. Une paraphrase correcte, avec « seated » et « rug », tombe à
47. Avec les mots mélangés, la précision unigramme reste à 100 %, mais les 4-grammes s'effondrent. La
traduction trop courte est écrasée par la pénalité de brièveté. Et surtout, un **contresens**, « the
dog has been eating », obtient **53, plus que la bonne paraphrase** ! Avec une deuxième référence,
la paraphrase remonte à 85.
→ BLEU mesure un recouvrement de surface, pas le sens.

### Slide 9 · Synthèse (6:20 → 6:50)
Pour conclure : en classification, on rapporte le F1, le rappel et le macro-F1, car l'accuracy est trompeuse quand
les classes sont déséquilibrées. En NER, on rapporte le F1 au niveau entité, parce que l'accuracy par token surestime la qualité. En traduction, on
combine BLEU, chrF et une métrique neuronale comme COMET, avec plusieurs références. Aucune
métrique n'est suffisante seule. Le code et le README sont sur mon dépôt GitHub. Merci de votre attention.
