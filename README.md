# Credit risk modeling


---

# 1. Project Information

Fill in the following information.

- **Project Title:** Credit risk modeling
- **Group Name:** FMGR x GFF
- **Group Members:**  
  - Student 1 – Lino Lucas
  - Student 2 – Quentin Abherve Gueguen  
  - Student 3 – Marc Fabrègue
  - Student 4 – Melina Placide
  - Student 5 – Gabin Botrel
  - Student 6 – Antonin Lozach

- **Course Name:** Introduction to machine learning
- **Instructor:** Nicolas De Roux & Mohamed EL FAKIR
- **Submission Date:** [Insert date]

---

# 2. Project Description

Notre projet porte sur le risque de crédit : peut-on prédire le défaut de paiement d'un client d'une carte de crédit ?  Nous utilisons le jeu de données Default of Credit Card Clients de l'UCI. Il décrit 30 000 clients d'une banque taïwanaise en 2005. Pour chaque client, on dispose du plafond de crédit, de quelques infos (sexe, âge, niveau d'études, situation familiale) et de six mois d'historique : retards de paiement, montants facturés et montants remboursés.

Le contexte est important : au milieu des années 2000, Taïwan a connu une crise des cartes de crédit. Les banques avaient distribué beaucoup de crédit sans vérifier assez la solvabilité des clients, et les impayés ont explosé. Le sujet nous intéresse parce que c'est un cas concret où une mauvaise décision coûte vite très cher. Notamment avec l'augmentation significative de l'utilisation des cartes des crédits. Il pose aussi un vrai défi de machine learning : seuls 22 % des clients environ font défaut, donc les classes sont déséquilibrées. Un modèle qui prédit « pas de défaut » pour tout le monde aurait 78 % de bonnes réponses tout en étant inutile.

Plusieurs acteurs profiteraient d'une bonne solution :

Les banques, qui pourraient repérer plus tôt les clients à risque, ajuster les plafonds et réduire leurs pertes.
Les régulateurs, pour qui la maîtrise du risque de crédit compte pour la stabilité financière.
Les clients eux-mêmes, qui pourraient être contactés avant de se retrouver surendettés.

---

# 3. Project Goal

Il s'agit d'un problème de classification binaire : à partir des caractéristiques du client et de son historique sur six mois, prédire s'il fera défaut le mois suivant (1 = défaut, 0 = pas de défaut). Le modèle doit aussi donner une probabilité de défaut, pour que la banque puisse classer ses clients du moins risqué au plus risqué.

Pour y arriver, nous partirons d'un modèle simple de référence, la régression logistique. Nous le comparerons ensuite à des modèles plus puissants : random forest, XGBoost et un petit réseau de neurones. Pour gérer le déséquilibre des classes, nous testerons la pondération des classes, le rééchantillonnage et l'ajustement du seuil de décision.

Une solution réussie, c'est :

De bonnes performances sur les bonnes métriques. On regardera l'AUC, le F1-score et la matrice de confusion plutôt que l'accuracy, et surtout un bon rappel sur les défauts : rater un client qui fera défaut coûte plus cher à la banque qu'une fausse alerte.
Une nette amélioration par rapport à la régression logistique.
Un modèle interprétable. Grâce à l'importance des variables et à SHAP, on veut pouvoir expliquer quelles variables pèsent le plus dans le risque. On s'attend à ce que les retards de paiement récents et le taux d'utilisation du crédit ressortent en premier.

---

# 4. Task Definition

Le projet est un problème d'apprentissage supervisé : pour chaque client, on connaît son profil, ses 6 derniers mois de comportement de paiement et ce qui s'est passé ensuite (défaut ou non). Le modèle apprend à relier les deux pour prédire le risque de nouveaux clients.

- **Task Type :** classification binaire supervisée, sur des données déséquilibrées (77,9 % de non-défauts, 22,1 % de défauts).

- **Input Variables :** 23 variables explicatives décrivant chaque client (30 000 clients) :
  - *Profil* : `LIMIT_BAL` (plafond de crédit), `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
  - *Historique de retard* : `PAY_1` à `PAY_6` (statut de paiement d'avril à septembre 2005)
  - *Montants facturés* : `BILL_AMT1` à `BILL_AMT6`
  - *Montants remboursés* : `PAY_AMT1` à `PAY_AMT6`


- **Target Variable :** `default payment next month` 
  - `1` = le client fait défaut sur son paiement du mois suivant (octobre 2005)
  - `0` = le client paie normalement
  - **Sortie du modèle :** une probabilité de défaut (PD) entre 0 et 1, transformée en décision (défaut / pas défaut) à l'aide d'un seuil choisi par défaut à 0,5.

- **Evaluation Metric(s) :**
  - **ROC-AUC** *(métrique principale)* : mesure la capacité du modèle à classer les clients du plus risqué au moins risqué, sans dépendre du seuil. Adaptée aux données déséquilibrées (0,5 = hasard, 1 = parfait).
  - **F1-score de la classe « défaut »** : moyenne harmonique de la précision (part des alertes qui sont de vrais défauts) et du rappel (part des défauts détectés). Elle évalue le modèle au seuil retenu.
  - **Matrice de confusion** : montre concrètement les deux types d'erreur, les **défauts manqués** (FN) qui sont des pertes pour la banque et les **fausses alertes** (FP) qui sont de bons clients refusés, donc manque à gagner.

- **Non sélection de l'accuracy** : Un modèle qui prédit toujours « pas de défaut » obtient 77,9 % d'accuracy sans détecter un seul défaut. Elle est donc pas utile, voire trompeuse sur ce problème.

**Référence à battre :** la régression logistique (baseline) obtient une **ROC-AUC de XXXXX** et un **F1 de XXXXX** sur le jeu de test. Random Forest, XGBoost et le réseau de neurones devront faire mieux.


---

# 5. Dataset Description

## Dataset Overview

- **Number of samples :** 30 000 clients (29 965 après suppression des doublons)
- **Number of features :** 23 variables explicatives, 1 identifiant (`ID`) et 1 variable cible
- **Target variable :** `default payment next month`
- **Data source :** [UCI Machine Learning Repository – Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Yeh & Lien, 2009)
- **Contexte :** clients d'une banque taïwanaise, observés d'avril à septembre 2005. Les montants sont en dollars taïwanais (NT$).

## Feature Description

Les 23 variables explicatives se répartissent en 4 familles.

### 1. Profil du client (5 variables)

| Variable | Description | Valeurs | Type |
|------|------|------|------|
| `LIMIT_BAL` | Plafond de crédit accordé au client (crédit individuel + crédit familial complémentaire) | 10 000 à 1 000 000 NT$ (médiane : 140 000) | Numérique |
| `SEX` | Genre | 1 = homme (39,6 %), 2 = femme (60,4 %) | Catégorielle |
| `EDUCATION` | Niveau d'études | 1 = master / doctorat, 2 = licence, 3 = lycée, 4 = autres ; 0, 5, 6 = non documentés | Catégorielle |
| `MARRIAGE` | Situation familiale | 1 = marié, 2 = célibataire, 3 = autres ; 0 = non documenté | Catégorielle |
| `AGE` | Âge du client | 21 à 79 ans (médiane : 34) | Numérique |

### 2. Historique de remboursement (6 variables)

`PAY_1` à `PAY_6` indiquent, pour chacun des 6 derniers mois, **si le client a payé à temps ou avec combien de mois de retard**.

| Variable | Mois concerné |
|------|------|
| `PAY_1` *(appelée `PAY_0` dans le fichier d'origine)* | Septembre 2005 |
| `PAY_2` | Août 2005 |
| `PAY_3` | Juillet 2005 |
| `PAY_4` | Juin 2005 |
| `PAY_5` | Mai 2005 |
| `PAY_6` | Avril 2005 |

| Code | Signification |
|------|------|
| -2 | Aucune consommation sur le mois |
| -1 | Facture payée intégralement |
| 0 | Paiement minimum effectué (crédit renouvelable, sans retard) |
| 1 à 8 | Retard de paiement de 1 à 8 mois (9 = 9 mois ou plus) |

### 3. Montants des relevés (6 variables)

| Variable | Description | Valeurs (sept. 2005) | Type |
|------|------|------|------|
| `BILL_AMT1` … `BILL_AMT6` | Montant du relevé de carte de crédit, de septembre 2005 (`BILL_AMT1`) à avril 2005 (`BILL_AMT6`) | −165 580 à 964 511 NT$ (médiane : 22 382) | Numérique |

### 4. Montants remboursés (6 variables)

| Variable | Description | Valeurs (sept. 2005) | Type |
|------|------|------|------|
| `PAY_AMT1` … `PAY_AMT6` | Montant remboursé par le client, de septembre 2005 (`PAY_AMT1`) à avril 2005 (`PAY_AMT6`) | 0 à 873 552 NT$ (médiane : 2 100) | Numérique |

## Target Variable

- **Nom :** `default payment next month` (renommée `DEFAULT` dans le code)
- **Signification :** le client a-t-il fait défaut sur son paiement d'octobre 2005 ?
- **Valeurs :**
  - `0` = pas de défaut : 23 364 clients (77,9 %)
  - `1` = défaut : 6 636 clients (22,1 %)

Le modèle ne prédit pas seulement 0 ou 1 : il estime une **probabilité de défaut (PD)**, transformée ensuite en décision à l'aide d'un seuil.

## Data Types

| Type | Nombre | Variables | Traitement prévu |
|------|------|------|------|
| Numériques continues | 14 | `LIMIT_BAL`, `AGE`, `BILL_AMT1–6`, `PAY_AMT1–6` | Transformation log + standardisation (pour LR et MLP) |
| Catégorielles nominales | 3 | `SEX`, `EDUCATION`, `MARRIAGE` | One-hot encoding (les codes sont des étiquettes, pas des grandeurs) |
| Ordinales | 6 | `PAY_1–6` | Conservées telles quelles (l'ordre a un sens) |
| Binaire (cible) | 1 | `default payment next month` | — |

**Structure temporelle :** les blocs `PAY`, `BILL_AMT` et `PAY_AMT` décrivent les **6 mêmes mois** pour chaque client. On n'a pas une vraie série temporelle, mais on peut en tirer des indicateurs de tendance (nombre de mois en retard, évolution du solde).

## Data Distribution     !!!!! A VOIR !!!!!!

- **Classes déséquilibrées :** 22,1 % de défauts. Un modèle qui prédit toujours « pas de défaut » atteindrait 77,9 % d'accuracy sans détecter un seul défaut.
- **Montants très asymétriques :** la plupart des clients ont des montants modestes et quelques-uns des montants très élevés. Par exemple, `PAY_AMT1` a une médiane de 2 100 NT$ mais un maximum de 873 552 NT$ (asymétrie de **XXXXX**).
- **L'historique de retard est la variable la plus prédictive :**
  - le taux de défaut passe de **13 %** pour un client qui paie le minimum (`PAY_1` = 0) à **34 %** avec 1 mois de retard et **69 %** avec 2 mois ;
  - `PAY_1` est la variable la plus corrélée à la cible (+0,32), et l'effet diminue pour les mois plus anciens (+0,19 pour `PAY_6`).
- **Un plafond élevé est associé à moins de défauts** (corrélation −0,15) : la banque accorde plus aux clients qu'elle juge solvables.
- **Variables peu prédictives seules :** `AGE`, `SEX`, `EDUCATION`, `MARRIAGE` et les `BILL_AMT` ont des corrélations inférieures à 0,05 en valeur absolue avec la cible.
- **Forte corrélation entre variables :** les `BILL_AMT` de deux mois consécutifs sont corrélés à 0,95. Cette redondance gêne l'interprétation de la régression logistique.

## Data Quality

| Point vérifié | Constat | Décision (section 6) |
|------|------|------|
| Valeurs manquantes | Aucune | — |
| Doublons | 35 lignes identiques (avec des `ID` différents) | Supprimées |
| Codes non documentés | `EDUCATION` = 0, 5, 6 (345 clients) ; `MARRIAGE` = 0 (54 clients) | Regroupés dans « autres » |
| Codes -2 et 0 de `PAY_x` | Absents de la documentation officielle, mais expliqués ensuite par l'auteur du dataset | Conservés |
| Montants négatifs | 1 930 clients ont au moins un `BILL_AMT` négatif | Conservés : ce sont des soldes créditeurs, pas des erreurs |
| Valeurs extrêmes | Montants élevés mais plausibles au vu des plafonds | Transformés (log) plutôt que supprimés |
| Comptes inactifs | 795 clients sans aucune facture ni remboursement sur 6 mois | Conservés |
| Variables sensibles | `SEX`, `AGE`, `MARRIAGE` sont des caractéristiques protégées | Modèles comparés avec et sans ces variables |


---

# 6. Data Preprocessing
---


Toutes les étapes sont codées dans [`src/preprocessing.py`](src/preprocessing.py). Tous les notebooks utilisent ce même fichier, donc chaque modèle est entraîné sur des données préparées de façon identique.

## 6.1 Chargement et mise en forme

- **Lecture du fichier :** le fichier UCI a **deux lignes d'en-tête** (`X1 … X23`, puis les vrais noms). On lit donc les noms sur la 2ᵉ ligne (`header=1`).
- **Renommage :** `PAY_0` → `PAY_1` et `default payment next month` → `DEFAULT`.
  *Pourquoi :* dans le fichier d'origine, la série de retards commence à `PAY_0` mais les montants commencent à `BILL_AMT1` et `PAY_AMT1`, alors que ces trois colonnes portent sur le même mois (septembre 2005). Le renommage aligne les numéros et évite les erreurs.
- **Suppression de `ID` :** *Pourquoi :* un numéro de client ne contient aucune information sur le risque. Le garder pourrait faire apprendre au modèle une corrélation artificielle.

## 6.2 Valeurs manquantes

Le dataset ne contient **aucune valeur manquante**, donc aucune imputation n'est nécessaire.
Les seules valeurs manquantes possibles apparaissent lors de la création de ratios (division par une facture nulle). Elles sont traitées directement dans le calcul (voir 6.6).

## 6.3 Doublons

**35 lignes** sont identiques sur toutes les variables (avec des `ID` différents). Elles sont supprimées, ce qui laisse 29 965 clients.
*Pourquoi :* si une même ligne se retrouve à la fois dans le train et dans le test, le modèle est évalué sur un client qu'il a déjà vu, ce qui gonfle artificiellement les scores (**fuite de données**). Retirer 0,1 % des données ne coûte rien.

## 6.4 Nettoyage des variables catégorielles

| Variable | Problème | Traitement |
|------|------|------|
| `EDUCATION` | Codes 0, 5 et 6 non documentés (345 clients) | Regroupés avec 4 = « autres » |
| `MARRIAGE` | Code 0 non documenté (54 clients) | Regroupé avec 3 = « autres » |

*Pourquoi :* ces codes n'ont pas de signification connue et concernent très peu de clients. Des catégories aussi petites seraient mal apprises par les modèles et rendraient les résultats instables.

**Valeurs conservées volontairement :**
- **Codes -2 et 0 de `PAY_x`** : ils ont un sens (pas de consommation / paiement minimum). On garde `PAY_x` comme variable **ordinale** de -2 à 8, car l'ordre « payé → minimum → retard » reflète un risque croissant.
- **Montants `BILL_AMT` négatifs** (1 930 clients) : ce sont des soldes créditeurs (le client a trop payé), donc une vraie information sur son comportement, pas une erreur.
- **Valeurs extrêmes** : elles sont plausibles au vu des plafonds (jusqu'à 1 000 000 NT$). On les transforme (6.7) au lieu de les supprimer, pour ne pas perdre les clients aux gros montants.

## 6.5 Encodage des variables catégorielles

`SEX`, `EDUCATION` et `MARRIAGE` sont transformées par **one-hot encoding** (une colonne 0/1 par catégorie, en retirant la première pour éviter la redondance).
*Pourquoi :* ces variables sont **nominales**. Leurs codes sont des étiquettes, pas des grandeurs. Laisser `MARRIAGE` en 1/2/3 ferait croire à un modèle linéaire que « célibataire (2) » vaut deux fois « marié (1) ».
Les `PAY_x`, eux, ne sont **pas** encodés en one-hot : leur ordre a un sens et doit être conservé.

## 6.6 Feature engineering

On crée 5 variables métier qui résument le comportement du client sur les 6 mois.

| Nouvelle variable | Calcul | Ce qu'elle mesure |
|------|------|------|
| `UTIL_MEAN` | Moyenne de `BILL_AMTi / LIMIT_BAL` sur 6 mois | Taux d'utilisation moyen du crédit |
| `UTIL_LAST` | `BILL_AMT1 / LIMIT_BAL` | Taux d'utilisation le plus récent |
| `PAY_RATIO_MEAN` | Moyenne de `PAY_AMTi / BILL_AMT(i+1)`, bornée entre 0 et 1 | Part de la facture précédente effectivement remboursée |
| `N_MONTHS_LATE` | Nombre de mois avec `PAY_x` > 0 | Fréquence des retards |
| `MAX_DELAY` | Plus grand retard observé sur 6 mois | Gravité du pire retard |

*Pourquoi :* les montants bruts sont peu liés au défaut (les corrélations des `BILL_AMT` sont proches de 0). Ce qui compte pour une banque, c'est **le montant rapporté au plafond** et **la part effectivement remboursée**.
Ces variables ont aussi un sens économique clair, ce qui facilite l'interprétation.

**Résultat :** `N_MONTHS_LATE` atteint une corrélation de **+0,40** avec le défaut, plus que n'importe quelle variable d'origine (`PAY_1` : +0,32). Le taux de défaut passe de **12 %** pour les clients jamais en retard à **70 %** pour ceux en retard les 6 mois.

*Détail technique :* `PAY_AMTi` rembourse la facture du mois précédent, d'où le rapport à `BILL_AMT(i+1)`. Quand cette facture est nulle ou négative, il n'y a rien à rembourser et le ratio vaut 1.

## 6.7 Séparation train / test

- **80 % train (23 972 clients) / 20 % test (5 993 clients)**, avec `random_state=42` pour que les résultats soient reproductibles.
- **Split stratifié** sur `DEFAULT` : le taux de défaut est de 22,1 % dans les deux échantillons.
*Pourquoi :* sans stratification, le test pourrait contenir par hasard plus ou moins de défauts que la réalité, ce qui fausserait l'évaluation.
- Le **test n'est plus utilisé** jusqu'à l'évaluation finale. Le réglage des hyperparamètres et le choix du seuil se font par **validation croisée stratifiée à 5 plis** sur le train.

## 6.8 Transformation et mise à l'échelle

Cette étape concerne la **régression logistique** et le **réseau de neurones (MLP)**. Random Forest et XGBoost utilisent les données sans transformation.

1. **Transformation logarithmique** `sign(x) · log(1 + |x|)` sur `LIMIT_BAL`, `BILL_AMT1–6` et `PAY_AMT1–6`.
   *Pourquoi :* ces montants sont très asymétriques (asymétrie jusqu'à 14,7). Le log réduit l'effet des valeurs extrêmes. La version avec `sign` permet de traiter aussi les montants négatifs.
2. **Standardisation** (`StandardScaler`) : chaque variable est ramenée à une moyenne de 0 et un écart-type de 1.
   *Pourquoi :* la régression logistique régularisée et le MLP sont sensibles à l'échelle. Sans standardisation, `LIMIT_BAL` (en centaines de milliers) écraserait `AGE` (en dizaines). Pour le MLP, c'est aussi nécessaire pour que l'entraînement converge.
   Les arbres de décision, eux, ne comparent les valeurs qu'à des seuils : l'échelle ne change rien pour eux.

⚠️ **Le scaler est ajusté uniquement sur le train**, puis appliqué au test. Il est placé dans un `Pipeline` avec le modèle, et donc ré-ajusté dans chaque pli de la validation croisée.
*Pourquoi :* calculer la moyenne et l'écart-type sur toutes les données ferait passer de l'information du test dans l'entraînement (fuite de données).

## 6.9 Traitement du déséquilibre des classes

Avec 22,1 % de défauts, un modèle non corrigé privilégie la classe majoritaire et **rate la plupart des défauts**. On corrige ce déséquilibre **uniquement pendant l'entraînement** :

| Méthode | Modèles | Principe |
|------|------|------|
| `class_weight="balanced"` | Régression logistique, Random Forest | Une erreur sur un défaut pèse environ 3,5 fois plus dans la fonction de perte |
| `scale_pos_weight ≈ 3,52` | XGBoost | Même principe (rapport non-défauts / défauts du train) |
| SMOTE *(variante testée)* | Tous | Création de défauts synthétiques, appliquée **dans chaque pli** de validation croisée |
| Ajustement du seuil de décision | Tous | Seuil choisi pour maximiser le F1, au lieu de 0,5 |

⚠️ **Le test n'est jamais rééquilibré.**
*Pourquoi :* le modèle sera utilisé sur une population réelle avec environ 22 % de défauts. Il doit être évalué dans ces conditions. Rééquilibrer **avant** le split (par exemple en dupliquant des défauts) placerait des copies des mêmes clients dans le train et le test, et produirait des scores artificiellement élevés.

## 6.10 Résultat du prétraitement

| | Avant | Après |
|------|------|------|
| Clients | 30 000 | 29 965 |
| Variables explicatives | 23 | 28 (23 d'origine + 5 construites) |
| Colonnes en entrée des modèles (LR / MLP) | — | 31 (après one-hot encoding) |
| Train / Test | — | 23 972 / 5 993 (22,1 % de défauts dans chacun) |

Pour lancer le prétraitement seul :

```bash
python src/preprocessing.py data/default_of_credit_card_clients.xls
```

---

# 7. Modeling Approach

📌 **Instructions:**  
Explain how you solved the problem.

---

## Chosen Models

List the models or algorithms used.

Examples:

- Linear Regression
- Logistic Regression
- Random Forest
- Gradient Boosting
- Neural Networks

✏️ **List and describe the models used**

---

## Modeling Strategy

📌 **Instructions:**  
Explain:

- Why you selected these models
- Whether you used a baseline model
- If hyperparameter tuning was performed
- Whether cross-validation was used

✏️ **Explain your modeling strategy**

---

## Evaluation Metrics

📌 **Instructions:**  
Specify the metrics used to evaluate model performance.

Examples:

- Accuracy
- Precision / Recall
- F1-score
- ROC-AUC
- Mean Absolute Error
- RMSE

Also explain **why these metrics are appropriate**.

✏️ **Describe your evaluation metrics**

---

# 8. Project Structure

📌 **Instructions:**  
Explain how the repository is organized.


If you added additional folders, explain them.

---

# 9. Installation

📌 **Instructions:**  
Explain how to install project dependencies.

Example:

```bash
pip install -r requirements.txt
