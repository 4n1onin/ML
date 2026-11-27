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

Notre projet porte sur le risque de crédit : peut-on prédire le défaut de paiement d'un client d'une carte de crédit ?  Nous utilisons le jeu de données Default of Credit Card Clients de l'UCI. Il décrit 30 000 clients d'une banque taïwanaise en 2005. Pour chaque client, on dispose du plafond de crédit, des informations démographiques telles que le sexe, l'âge, le niveau d'études, et la situation familiale ainsi que six mois d'historique : retards de paiement, montants facturés et montants remboursés.

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

De bonnes performances sur les bonnes métriques. On regardera l'AUC, le F1-score et la matrice de confusion plutôt que l'accuracy, et surtout un bon recall sur les défauts : rater un client qui fera défaut coûte plus cher à la banque qu'une fausse alerte.
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
  - **Sortie du modèle :** une probabilité de défaut (PD) entre 0 et 1, transformée en décision (défaut / pas défaut) à l'aide d'un seuil choisi par défaut à 0,5 puis à l'aide d'un seuil qui maximise le F1.

- **Evaluation Metric(s) :**
  - **ROC-AUC** *(métrique principale)* : mesure la capacité du modèle à classer les clients du plus risqué au moins risqué, sans dépendre du seuil. Adaptée aux données déséquilibrées (0,5 = hasard, 1 = parfait).
  - **F1-score de la classe « défaut »** : moyenne harmonique de la précision (part des alertes qui sont de vrais défauts) et du recall (part des défauts détectés). Elle évalue le modèle au seuil retenu.
  - **Matrice de confusion** : montre concrètement les deux types d'erreur, les défauts manqués (FN) qui sont des pertes pour la banque et les fausses alertes (FP) qui sont de bons clients refusés, donc manque à gagner.

- **Non sélection de l'accuracy** : Un modèle qui prédit toujours « pas de défaut » obtient 77,9 % d'accuracy sans détecter un seul défaut. Elle est donc pas utile, voire trompeuse sur ce problème.

**Référence à battre :** la régression logistique (baseline) obtient une ROC-AUC de 0,76 et un F1 de 0,526 sur le jeu de test. Random Forest, XGBoost et le réseau de neurones devront faire mieux.


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

`PAY_1` à `PAY_6` indiquent, pour les 6 derniers mois, si le client a payé à temps ou avec combien de mois de retard.

| Variable | Mois concerné |
|------|------|
| `PAY_1` *(appelée `PAY_0` dans initialement)* | Septembre 2005 |
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

Le modèle ne prédit pas seulement 0 ou 1 : il estime une probabilité de défaut transformée ensuite en décision à l'aide d'un seuil.

## Data Types

| Type | Nombre | Variables | Traitement prévu |
|------|------|------|------|
| Numériques continues | 14 | `LIMIT_BAL`, `AGE`, `BILL_AMT1–6`, `PAY_AMT1–6` | Transformation log + standardisation (pour LR et MLP) |
| Catégorielles nominales | 3 | `SEX`, `EDUCATION`, `MARRIAGE` | One-hot encoding (les codes sont des étiquettes, pas des grandeurs) |
| Ordinales | 6 | `PAY_1–6` | Conservées telles quelles (l'ordre a un sens) |
| Binaire (cible) | 1 | `default payment next month` | — |

**Structure temporelle :** les blocs `PAY`, `BILL_AMT` et `PAY_AMT` décrivent les **6 mêmes mois** pour chaque client. On n'a pas une vraie série temporelle, mais on peut en tirer des indicateurs de tendance (nombre de mois en retard, évolution du solde).

## Data Distribution   

- **Classes déséquilibrées :** 22,1 % de défauts. Un modèle qui prédit toujours « pas de défaut » atteindrait 77,9 % d'accuracy sans détecter un seul défaut.
- **Montants très asymétriques :** la plupart des clients ont des montants modestes et quelques-uns des montants très élevés. Par exemple, `PAY_AMT1` a une médiane de 2 100 NT$ mais un maximum de 873 552 NT$ (asymétrie de 14,7).
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



---
# 6. Data Preprocessing

Toutes les étapes sont codées ici: [`data/credit_risk_logistic_regression.ipynb`](data/credit_risk_logistic_regression.ipynb).

## 6.1 Chargement et nettoyage

- **Lecture :** le fichier UCI a deux lignes d'en-tête, on lit les noms sur la 2ᵉ (`header=1`).
- **Renommage :** `PAY_0` → `PAY_1` (même mois que `BILL_AMT1` et `PAY_AMT1`) et la cible → `DEFAULT`.
- **Suppression de `ID` :** un numéro de client ne dit rien sur le risque.
- **Valeurs manquantes :** aucune, donc pas d'imputation.
- **Doublons :** 35 lignes supprimées (29 965 clients restants), pour qu'un même client ne soit pas à la fois dans le train et le test (fuite de données).
- **Codes non documentés :** `EDUCATION` 0, 5, 6 → 4 (« autres ») ; `MARRIAGE` 0 → 3 (« autres »). Ces catégories sont trop petites pour être bien apprises.
- **Conservés volontairement :** les codes -2 et 0 de `PAY_x` (ils ont un sens), les `BILL_AMT` négatifs (soldes créditeurs) et les valeurs extrêmes (reste possible).

## 6.2 Feature engineering

| Variable | Calcul | Ce qu'elle mesure |
|------|------|------|
| `UTIL_MEAN` | moyenne de `BILL_AMT` / `LIMIT_BAL` | utilisation moyenne du crédit |
| `PAY_RATIO` | `PAY_AMT1` / `BILL_AMT2`, borné entre 0 et 1 | part de la facture d'août remboursée en septembre |
| `N_MONTHS_LATE` | nombre de mois avec `PAY_x` > 0 | fréquence des retards |
| `MAX_DELAY` | plus grand `PAY_x` | gravité du pire retard |


## 6.3 Séparation train / test

- **80 % train (23 972 clients) / 20 % test (5 993 clients)**, `random_state=42`.
- **Split stratifié** : 22,1 % de défauts dans les deux échantillons, pour un test représentatif.
- Le test n'est utilisé qu'à la toute fin. Le choix du seuil se fait par **validation croisée à 5 plis** sur le train.

## 6.4 Encodage et mise à l'échelle

1. **One-hot encoding** de `SEX`, `EDUCATION`, `MARRIAGE` : ce sont des étiquettes, pas des grandeurs.
2. **Log** (`log(1 + x)`) sur `LIMIT_BAL` et les `PAY_AMT` : ces montants sont très asymétriques (asymétrie jusqu'à 14,7). Les `BILL_AMT` pouvant être négatifs, on ne leur applique pas le log.
3. **Standardisation** (`StandardScaler`) : nécessaire pour la régression logistique et le MLP. 

Le scaling est ajusté sur le train uniquement, puis appliqué au test, pour éviter toute fuite de données.

## 6.5 Déséquilibre des classes

- `class_weight="balanced"` : une erreur sur un défaut pèse environ 3,5 fois plus pendant l'entraînement.
- **Ajustement du seuil de décision** pour maximiser le F1, au lieu de 0,5.
- Le **test n'est jamais rééquilibré** : le modèle doit être évalué sur une population réaliste (22 % de défauts).

## 6.6 Résultat

| | Avant | Après |
|------|------|------|
| Clients | 30 000 | 29 965 |
| Variables explicatives | 23 | 27 (23 + 4 construites) |
| Colonnes après one-hot encoding | — | 30 |
| Train / Test | — | 23 972 / 5 993 |

---

# 7. Modeling Approach

## Chosen Models

- **Régression logistique** : modèle linéaire simple, utilisé comme baseline.
- **K plus proches voisins (KNN)** : la probabilité de défaut d'un client est la part de défauts parmi ses *k* voisins les plus proches dans le train. Même prétraitement que la LR (indispensable : le KNN repose sur des distances). Code : partie 2 de [`data/credit_risk_logistic_regression.ipynb`](data/credit_risk_logistic_regression.ipynb).

**(rajoutez vos modèles ici également.)**

---

## Modeling Strategy

- Séparation train / test (80 / 20), stratifiée
- Régression logistique utilisée comme baseline
- Gestion du déséquilibre : `class_weight="balanced"`
- Validation croisée à 5 plis sur le train pour choisir le seuil de décision
- Seuil retenu : 0,62 (au lieu de 0,5), celui qui maximise le F1
- Pas de réglage des hyperparamètres pour la baseline

---

## Evaluation Metrics

- **ROC-AUC** → métrique principale, adaptée aux données déséquilibrées
- **F1-score** → moyenne armonique entre précision et recall
- **Matrice de confusion** → montre les défauts manqués et les fausses alertes

L'accuracy n'est pas utilisée : prédire toujours « pas de défaut » donne 78 % sans détecter aucun défaut.

---

## Results

### Régression logistique (baseline)
- AUC : 0,761
- F1 : 0,526
- Recall : 0,540 (716 défauts détectés sur 1 326)
- Précision : 0,514

### KNN
- Hyperparamètres choisis par validation croisée à 5 plis sur le train (critère : ROC-AUC) : *k* = 250, distance de Manhattan, pondération uniforme
- Sélection des variables pas à pas (ajout à chaque étape de la variable qui fait le plus monter l'AUC en validation croisée), nombre de variables fixé par la règle d'un écart-type : 6 variables (`N_MONTHS_LATE`, `UTIL_MEAN`, `PAY_1`, `BILL_AMT1`, `BILL_AMT5`, `MAX_DELAY`). Elles font aussi bien que les 17 variables d'un filtre par corrélation (AUC test 0,766 contre 0,769, écart non significatif)
- Déséquilibre géré par le seuil (pas de `class_weight` pour le KNN) : 0,27, celui qui maximise le F1 en validation croisée
- AUC : 0,766
- F1 : 0,530
- Recall : 0,578 (767 défauts détectés sur 1 326)
- Précision : 0,489
- Qualité des probabilités (méthodes du papier de Yeh & Lien) : area ratio 0,532 ; régression *Sorting Smoothing Method* B = 0,959, A = 0,006, R² = 0,947 ; Brier 0,137. Les PD du KNN sont réalistes, contrairement à celles de la LR `balanced`, qui les surestime (PD moyenne 0,44 pour un taux réel de 0,22)
- Contribution des variables (on retire la variable, on ré-entraîne et on mesure la baisse d'AUC en validation croisée) : les retards de paiement dominent (baisse de 0,13 si on les retire tous), puis le niveau de dette (0,05). Prises une à une, `PAY_1` et `UTIL_MEAN` apportent le plus d'information propre




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
