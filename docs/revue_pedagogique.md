# Revue pédagogique - OpenWatch NLP

Ce document résume ce qu'il faut comprendre et savoir expliquer avant de présenter le projet `OpenWatch NLP` en entretien.

L'objectif n'est pas d'apprendre un discours par coeur. L'objectif est de pouvoir expliquer clairement :

- le problème traité ;
- les données utilisées ;
- le pipeline de machine learning ;
- les métriques d'évaluation ;
- les limites du projet ;
- les améliorations possibles.

## 1. Vue d'ensemble du projet

`OpenWatch NLP` est un projet de classification de texte.

Le principe est simple : on donne au modèle un texte court, et il doit le classer dans l'une des trois catégories suivantes :

- `red` : signal critique ;
- `orange` : signal intermédiaire ;
- `green` : contenu non pertinent.

Le projet démontre une chaîne Data Science complète :

- génération de données synthétiques ;
- nettoyage de texte ;
- vectorisation NLP avec TF-IDF ;
- entraînement d'un modèle de classification ;
- évaluation des performances ;
- prédiction sur un nouveau texte ;
- restitution dans une application Streamlit.

## 2. Phrase courte à utiliser en entretien

> J'ai construit un prototype NLP de classification de signaux faibles. Le projet utilise des données synthétiques, un preprocessing texte simple, une vectorisation TF-IDF et une régression logistique multiclasse. L'objectif est de démontrer un pipeline complet, explicable et publiable, sans prétendre à un modèle de production.

Cette phrase est importante car elle montre trois choses :

- tu sais décrire le projet simplement ;
- tu comprends les choix techniques ;
- tu ne survends pas les résultats.

## 3. Problème métier

Dans beaucoup de contextes, on peut avoir un grand volume de textes publics ou semi-publics à surveiller :

- messages de forums ;
- avis utilisateurs ;
- posts publics ;
- tickets ou discussions de support ;
- commentaires ou signaux faibles.

Le but n'est pas de remplacer l'analyse humaine. Le but est de prioriser les textes qui méritent une attention plus forte.

Dans ce projet :

- les textes `red` doivent être regardés en priorité ;
- les textes `orange` doivent être surveillés ;
- les textes `green` peuvent être considérés comme non prioritaires.

## 4. Pipeline du projet

Le pipeline complet est le suivant :

```text
texte brut
-> nettoyage du texte
-> vectorisation TF-IDF
-> modèle Logistic Regression
-> prédiction red/orange/green
-> affichage dans Streamlit
```

Chaque étape a un rôle précis.

| Etape | Rôle |
|---|---|
| Génération des données | Créer un dataset synthétique publiable |
| Nettoyage | Standardiser les textes avant modélisation |
| TF-IDF | Transformer le texte en variables numériques |
| Logistic Regression | Apprendre à associer des textes à des classes |
| Evaluation | Mesurer les performances du modèle |
| Streamlit | Rendre le résultat compréhensible et testable |

## 5. Rôle des fichiers principaux

| Fichier | Rôle |
|---|---|
| `openwatch-nlp/src/data_generation.py` | Génère le dataset synthétique |
| `openwatch-nlp/src/preprocessing.py` | Nettoie les textes |
| `openwatch-nlp/src/train_model.py` | Entraîne le modèle TF-IDF + Logistic Regression |
| `openwatch-nlp/src/evaluate_model.py` | Calcule les métriques et génère la matrice de confusion |
| `openwatch-nlp/src/predict.py` | Prédit la classe d'un nouveau texte |
| `openwatch-nlp/app/streamlit_app.py` | Fournit l'interface Streamlit |
| `openwatch-nlp/README.md` | Présente le projet pour GitHub |

## 6. Dataset synthétique

Le projet utilise des données synthétiques.

C'est un choix volontaire, pour éviter tout problème de confidentialité et rendre le projet publiable sur GitHub.

Le dataset contient trois classes :

- `red` ;
- `orange` ;
- `green`.

Il contient actuellement 600 textes :

- 200 textes `red` ;
- 200 textes `orange` ;
- 200 textes `green`.

Le dataset est équilibré, ce qui facilite l'entraînement et l'interprétation dans cette première version.

### Pourquoi utiliser des données synthétiques ?

Réponse à donner en entretien :

> Le sujet est inspiré de cas d'usage réalistes, mais le projet est public. J'ai donc choisi des données synthétiques pour éviter toute donnée confidentielle ou sensible. Cela permet de montrer la méthode, le pipeline et les choix techniques sans exposer de données réelles.

## 7. Nettoyage du texte

Le nettoyage est fait dans `preprocessing.py`.

Il consiste notamment à :

- passer le texte en minuscules ;
- retirer certains caractères spéciaux ;
- supprimer les espaces multiples ;
- normaliser le texte.

Exemple :

```text
"Public post mentions leaked credentials!!!"
```

devient :

```text
"public post mentions leaked credentials"
```

Le but est de réduire le bruit avant la vectorisation.

## 8. TF-IDF

TF-IDF signifie `Term Frequency - Inverse Document Frequency`.

C'est une méthode classique de NLP qui transforme du texte en nombres.

L'idée :

- un mot fréquent dans un texte peut être important ;
- mais un mot fréquent dans tous les textes est souvent peu discriminant.

Exemple :

- `credentials leaked` est probablement très informatif ;
- `the`, `and`, `users` sont moins discriminants.

TF-IDF donne donc plus de poids aux termes utiles pour distinguer les classes.

### Ce qu'il faut savoir dire

> TF-IDF permet de représenter un texte sous forme de vecteur numérique. Le modèle ne comprend pas le texte comme un humain, mais il apprend des associations statistiques entre des mots, des groupes de mots et des classes.

## 9. Régression logistique

Même si le nom contient le mot "régression", la régression logistique peut être utilisée pour faire de la classification.

Dans ce projet, elle prédit l'une des trois classes :

- `red` ;
- `orange` ;
- `green`.

Le modèle reçoit les variables TF-IDF et apprend quels mots ou expressions sont associés à chaque classe.

Pour un nouveau texte, il produit une probabilité par classe.

Exemple :

```text
red:    77%
orange: 11%
green:  12%
```

La classe retenue est celle qui a la probabilité la plus élevée.

### Pourquoi ce modèle est pertinent ici ?

La régression logistique est :

- simple ;
- rapide ;
- explicable ;
- adaptée comme baseline ;
- facile à défendre en entretien.

Un modèle plus complexe n'est pas forcément meilleur pour une première version de portfolio.

## 10. Métriques d'évaluation

Le projet utilise les métriques classiques de classification :

- accuracy ;
- precision ;
- recall ;
- F1-score ;
- matrice de confusion.

### Accuracy

L'accuracy mesure la proportion totale de bonnes prédictions.

Exemple :

```text
accuracy = prédictions correctes / nombre total de prédictions
```

Elle est utile, mais elle peut être trompeuse si les classes sont déséquilibrées.

### Precision

La precision répond à la question :

> Parmi les textes prédits dans une classe, combien appartenaient réellement à cette classe ?

Pour la classe `red` :

> Parmi les textes prédits `red`, combien étaient vraiment `red` ?

### Recall

Le recall répond à la question :

> Parmi les vrais textes d'une classe, combien le modèle a-t-il réussi à retrouver ?

Pour la classe `red` :

> Parmi tous les vrais textes critiques, combien ont été détectés comme critiques ?

### F1-score

Le F1-score combine precision et recall.

Il est utile quand on veut un indicateur synthétique qui équilibre les deux.

## 11. Pourquoi le recall de `red` est important

Dans ce projet, la classe `red` représente les signaux critiques.

Un faux négatif serait un texte réellement critique classé comme `orange` ou `green`.

C'est problématique parce qu'on risque de rater une alerte importante.

Phrase à utiliser :

> Pour la classe `red`, je surveille particulièrement le recall, car l'enjeu est de ne pas manquer les signaux critiques. Un faux positif peut créer du bruit, mais un faux négatif peut faire passer une alerte importante sous le radar.

## 12. Matrice de confusion

La matrice de confusion montre les erreurs de classification.

Les lignes représentent les vraies classes.

Les colonnes représentent les classes prédites.

Si les valeurs sont sur la diagonale, cela veut dire que le modèle a bien classé les exemples.

Dans le résultat actuel du projet, tout est sur la diagonale :

```text
red    -> red
orange -> orange
green  -> green
```

Cela indique que le modèle classe correctement les exemples du test set.

## 13. Limite importante du projet

Les scores actuels sont parfaits.

Il ne faut pas les présenter comme une preuve de performance en conditions réelles.

La bonne interprétation est la suivante :

> Les performances sont très élevées car la première version utilise un dataset synthétique contrôlé. La valeur du projet est de montrer un pipeline complet, propre et explicable. Une amélioration naturelle serait d'ajouter plus de diversité, de bruit et de cas ambigus dans les textes.

Cette limite est importante. Elle montre que tu sais prendre du recul sur ton propre modèle.

## 14. Ce que le projet démontre dans un portfolio

Ce projet montre que tu sais :

- structurer un projet Data Science ;
- générer un dataset synthétique publiable ;
- construire un pipeline NLP classique ;
- entraîner un modèle supervisé ;
- évaluer un modèle de classification ;
- interpréter les métriques ;
- exposer un résultat dans une application simple ;
- documenter un projet pour GitHub ;
- expliquer les limites d'une approche.

## 15. Questions possibles en entretien

### Pourquoi ne pas avoir utilisé un LLM ?

Réponse possible :

> Pour une première version, j'ai volontairement choisi un modèle classique et explicable. TF-IDF avec régression logistique permet de construire une baseline solide, rapide et facile à interpréter. Un LLM pourrait être ajouté ensuite pour résumer les alertes ou enrichir l'analyse, mais il n'était pas nécessaire pour démontrer le pipeline de classification.

### Pourquoi ne pas utiliser des données réelles ?

Réponse possible :

> Le projet est destiné à GitHub, donc je voulais éviter toute donnée confidentielle ou sensible. Les données synthétiques permettent de démontrer la méthode sans exposer de données réelles.

### Pourquoi les scores sont-ils parfaits ?

Réponse possible :

> Les textes sont synthétiques et encore assez structurés, donc le modèle apprend facilement les patterns. Je ne présente pas ces scores comme une performance production. Je les présente comme une validation technique du pipeline. Une prochaine étape serait de rendre les données plus diverses et plus ambiguës.

### Quelle serait la prochaine amélioration ?

Réponse possible :

> J'améliorerais d'abord le dataset : plus de variété, plus de bruit, des textes ambigus et des formulations moins évidentes. Ensuite, je ferais une analyse d'erreurs et je comparerais cette baseline à d'autres approches.

### Quelle est la différence entre NLP classique et LLM ?

Réponse possible :

> Le NLP classique repose souvent sur des représentations comme TF-IDF et des modèles supervisés relativement simples. Les LLM sont des modèles beaucoup plus larges, entraînés sur de grands corpus, capables de générer ou résumer du texte. Ici, le besoin était une classification explicable, donc un modèle classique était suffisant pour la V1.

## 16. Conclusion à retenir

La meilleure manière de présenter ce projet est la suivante :

> OpenWatch NLP est un projet portfolio qui montre ma capacité à construire un pipeline NLP complet, depuis la génération de données synthétiques jusqu'à l'évaluation et la restitution dans Streamlit. Le modèle est volontairement simple et explicable. Les limites sont clairement identifiées, notamment le caractère synthétique du dataset et les performances trop parfaites pour être considérées comme réalistes.
