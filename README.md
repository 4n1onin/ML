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

# 4. Task Definition

Le projet est un problème d'apprentissage supervisé : pour chaque client, on connaît son profil, ses 6 derniers mois de comportement de paiement et ce qui s'est passé ensuite (défaut ou non). Le modèle apprend à relier les deux pour prédire le risque de nouveaux clients.

- **Task Type :** classification binaire supervisée, sur des données déséquilibrées (77,9 % de non-défauts, 22,1 % de défauts).

- **Input Variables :** 23 variables explicatives décrivant chaque client (30 000 clients) :
  - *Profil* : `LIMIT_BAL` (plafond de crédit), `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
  - *Historique de retard* : `PAY_1` à `PAY_6` (statut de paiement d'avril à septembre 2005)
  - *Montants facturés* : `BILL_AMT1` à `BILL_AMT6`
  - *Montants remboursés* : `PAY_AMT1` à `PAY_AMT6`
  - *Variables construites* (section 6) : taux d'utilisation du crédit (`UTIL_MEAN`, `UTIL_LAST`), part de la facture remboursée (`PAY_RATIO_MEAN`), nombre de mois en retard (`N_MONTHS_LATE`), retard maximal (`MAX_DELAY`)

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

📌 **Instructions:**  
Describe the dataset used in the project.

❗ Do **not simply name the dataset**. Instead explain its structure and contents.

---

## Dataset Overview

Provide general information about the dataset.

Fill in:

- **Number of samples:**  
- **Number of features:**  
- **Target variable:**  
- **Data source:** [Link if applicable]

---

## Feature Description

📌 **Instructions:**  
List and describe the most important variables.

Example table:

| Feature | Description | Type |
|------|------|------|
| age | Age of individual | Numerical |
| income | Annual income | Numerical |
| gender | Gender category | Categorical |

✏️ **Insert your feature description table here**

---

## Target Variable

📌 **Instructions:**  
Explain what the model is trying to predict.

Include:

- Variable name
- Meaning
- Possible values (if classification)

✏️ **Write your explanation here**

---

## Data Types

📌 **Instructions:**  
Describe the types of variables present in the dataset.

Examples:

- Numerical
- Categorical
- Ordinal
- Text
- Time-series

✏️ **Describe the column types here**

---

## Data Distribution

📌 **Instructions:**  
Describe important distribution characteristics.

Examples:

- Class balance or imbalance
- Skewed numerical variables
- Range of key features

✏️ **Describe the data distribution here**

---

## Data Quality

📌 **Instructions:**  
Mention any issues found in the dataset.

Examples:

- Missing values
- Outliers
- Imbalanced classes
- Duplicate entries

✏️ **Describe any data quality issues here**

---

# 6. Data Preprocessing

📌 **Instructions:**  
Explain the preprocessing steps applied before modeling.

Examples:

- Handling missing values
- Removing duplicates
- Encoding categorical variables
- Normalizing or scaling features
- Feature engineering

For each step briefly explain **why it was necessary**.

✏️ **Describe your preprocessing steps here**

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
