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

📌 **Instructions:**  
Write a short paragraph explaining the **problem addressed by the project**.

Include:

- The context of the problem
- Why the problem is interesting or important
- Who might benefit from solving it

✏️ **Write your description below:**

Notre projet porte sur le risque de crédit : peut-on prédire le défaut de paiement d'un client d'une carte de crédit ?  Nous utilisons le jeu de données Default of Credit Card Clients de l'UCI. Il décrit 30 000 clients d'une banque taïwanaise en 2005. Pour chaque client, on dispose du plafond de crédit, de quelques infos (sexe, âge, niveau d'études, situation familiale) et de six mois d'historique : retards de paiement, montants facturés et montants remboursés.

Le contexte est important : au milieu des années 2000, Taïwan a connu une crise des cartes de crédit. Les banques avaient distribué beaucoup de crédit sans vérifier assez la solvabilité des clients, et les impayés ont explosé. Le sujet nous intéresse parce que c'est un cas concret où une mauvaise décision coûte vite très cher. Notamment avec l'augmentation significative de l'utilisation des cartes des crédits. Il pose aussi un vrai défi de machine learning : seuls 22 % des clients environ font défaut, donc les classes sont déséquilibrées. Un modèle qui prédit « pas de défaut » pour tout le monde aurait 78 % de bonnes réponses tout en étant inutile.

Plusieurs acteurs profiteraient d'une bonne solution :

Les banques, qui pourraient repérer plus tôt les clients à risque, ajuster les plafonds et réduire leurs pertes.
Les régulateurs, pour qui la maîtrise du risque de crédit compte pour la stabilité financière.
Les clients eux-mêmes, qui pourraient être contactés avant de se retrouver surendettés.

---

# 3. Project Goal

📌 **Instructions:**  
Clearly explain **what the project aims to achieve**.

Your answer should describe:

- What the system predicts, classifies, or analyzes
- What a successful solution looks like

Example goals:

- Predict housing prices from property features
- Classify medical images into disease categories
- Detect sentiment in text reviews
- Forecast electricity consumption

Il s'agit d'un problème de classification binaire : à partir des caractéristiques du client et de son historique sur six mois, prédire s'il fera défaut le mois suivant (1 = défaut, 0 = pas de défaut). Le modèle doit aussi donner une probabilité de défaut, pour que la banque puisse classer ses clients du moins risqué au plus risqué.

Pour y arriver, nous partirons d'un modèle simple de référence, la régression logistique. Nous le comparerons ensuite à des modèles plus puissants : random forest, XGBoost et un petit réseau de neurones. Pour gérer le déséquilibre des classes, nous testerons la pondération des classes, le rééchantillonnage et l'ajustement du seuil de décision.

Une solution réussie, c'est :

De bonnes performances sur les bonnes métriques. On regardera l'AUC, le F1-score et la matrice de confusion plutôt que l'accuracy, et surtout un bon rappel sur les défauts : rater un client qui fera défaut coûte plus cher à la banque qu'une fausse alerte.
Une nette amélioration par rapport à la régression logistique.
Un modèle interprétable. Grâce à l'importance des variables et à SHAP, on veut pouvoir expliquer quelles variables pèsent le plus dans le risque. On s'attend à ce que les retards de paiement récents et le taux d'utilisation du crédit ressortent en premier.

---

# 4. Task Definition

📌 **Instructions:**  
Define the **machine learning or data analysis task**.

Specify:

- **Task Type:** (classification, regression, clustering, etc.)
- **Input:** What data is used as input
- **Output:** What the model predicts
- **Evaluation Metric:** How performance will be measured

✏️ **Fill in the following:**

- **Task Type:**  
- **Input Variables:**  
- **Target Variable:**  
- **Evaluation Metric(s):**

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
