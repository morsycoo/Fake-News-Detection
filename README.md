<div align="center">

# 📰 Fake News Detection
### Production-Ready NLP Pipeline for Fake News Classification using Machine Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker)
![Pytest](https://img.shields.io/badge/Pytest-Tested-0A9EDC?style=for-the-badge&logo=pytest)
![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-2088FF?style=for-the-badge&logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

<p align="center">

A complete end-to-end Machine Learning project for detecting fake news articles using Natural Language Processing (NLP), TF-IDF feature engineering, statistical feature selection, and Linear Support Vector Machines (SVM).

The project is designed following production-ready software engineering practices including modular architecture, automated testing, Docker support, CI/CD with GitHub Actions, experiment tracking, artifact management, and a REST API built with FastAPI.

</p>

---

</div>

# 🚀 Project Highlights

- ✅ End-to-End NLP Pipeline
- ✅ Advanced Text Cleaning
- ✅ TF-IDF Vectorization
- ✅ SelectKBest (Chi-Square)
- ✅ Multiple Machine Learning Models
- ✅ Hyperparameter Optimization
- ✅ Model Explainability
- ✅ Error Analysis
- ✅ Learning & Validation Curves
- ✅ Production-Ready Pipeline
- ✅ FastAPI REST API
- ✅ Docker Support
- ✅ GitHub Actions CI
- ✅ Automated Unit Testing
- ✅ Artifact Management
- ✅ Experiment Tracking

---

# 📊 Final Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|-------|---------:|----------:|--------:|---------:|---------:|---------:|
| Naive Bayes | 0.9399 | 0.9390 | 0.9346 | 0.9368 | 0.9815 | 0.9796 |
| Logistic Regression | 0.9840 | 0.9805 | 0.9860 | 0.9832 | 0.9983 | 0.9982 |
| **Linear SVM (Final Model)** | **0.9933** | **0.9921** | **0.9939** | **0.9930** | **0.9998** | **0.9998** |

---

# 🎯 Why This Project?

Fake news spreads rapidly across digital platforms and can significantly influence public opinion, elections, financial markets, and public health decisions.

The goal of this project is to build a highly accurate and production-ready fake news classification system capable of distinguishing between legitimate and fabricated news articles using classical Natural Language Processing techniques and Machine Learning.

Unlike many educational notebooks, this repository demonstrates the complete lifecycle of an ML project—from data preprocessing and model development to deployment, testing, and reproducibility.

---

# 🏗️ Project Architecture

```text
                    Raw News Articles
                            │
                            ▼
                 Text Cleaning & Normalization
                            │
                            ▼
                  TF-IDF Feature Extraction
                            │
                            ▼
              SelectKBest (Chi-Square Features)
                            │
                            ▼
                 Linear Support Vector Machine
                            │
                            ▼
                  Prediction & Confidence Score
                            │
                            ▼
                 FastAPI REST API / Batch API
```

---

# ✨ Key Features

### Machine Learning

- Text Classification
- TF-IDF Vectorization
- Feature Selection
- Linear SVM
- Logistic Regression
- Naive Bayes
- Hyperparameter Search
- Cross Validation

### Natural Language Processing

- Text Cleaning
- URL Removal
- HTML Removal
- Punctuation Removal
- Lowercasing
- Tokenization
- Stopword Removal
- Noise Reduction

### Production Engineering

- Modular Codebase
- Config Management
- Logging System
- Artifact Saving
- Experiment Tracking
- Docker Deployment
- REST API
- Automated Tests
- GitHub Actions
- Reproducible Training

---

# 📚 Table of Contents

- Project Overview
- Dataset
- Data Preprocessing
- Feature Engineering
- Machine Learning Models
- Hyperparameter Tuning
- Model Evaluation
- Error Analysis
- Explainability
- Learning Curves
- Production Pipeline
- REST API
- Docker
- Testing
- Installation
- Usage
- Future Improvements
- License

---

# 📖 Problem Statement

## Background

The rapid growth of online news platforms and social media has dramatically increased the spread of misinformation. Fake news can influence public opinion, manipulate financial markets, affect elections, and create widespread confusion.

Traditional manual fact-checking is often too slow to keep pace with the volume of newly published articles. Consequently, automated Fake News Detection has become one of the most important Natural Language Processing (NLP) applications.

This project aims to build a reliable Machine Learning system capable of automatically classifying news articles as **Real** or **Fake** using textual content alone.

---

# 🎯 Project Objectives

The main objectives of this project are:

- Develop a highly accurate fake news classifier.
- Build a complete end-to-end NLP pipeline.
- Compare multiple Machine Learning algorithms.
- Analyze model errors and prediction confidence.
- Produce explainable model predictions.
- Package the model into a production-ready REST API.
- Ensure reproducibility through experiment tracking.
- Follow software engineering best practices.

---

# ❓ Why Linear SVM?

Several machine learning algorithms were evaluated throughout this project.

The final model selected for deployment was **Linear Support Vector Machine (Linear SVM)** because it provided the best overall balance between:

- Classification Accuracy
- Precision
- Recall
- F1 Score
- Generalization Performance
- Inference Speed

Compared to Logistic Regression and Naive Bayes, Linear SVM achieved the highest overall performance on the test set while maintaining excellent robustness.

---

# 🗂 Dataset

This project uses the widely adopted **Fake and Real News Dataset**, containing two independent collections of news articles.

| File | Description |
|------|-------------|
| Fake.csv | Fake news articles |
| True.csv | Legitimate news articles |

The datasets were merged into a single binary classification dataset.

---

# 📊 Dataset Statistics

| Statistic | Value |
|-----------|------:|
| Total Articles | **44,898** |
| Fake Articles | **23,481** |
| Real Articles | **21,417** |
| Number of Classes | **2** |
| Classification Type | Binary |
| Input Feature | News Text |
| Target Variable | Label |

---

# 🏷 Target Labels

The classification labels are encoded as follows:

| Label | Meaning |
|-------|---------|
| 0 | Fake News |
| 1 | Real News |

The label mapping is saved during training as:

```text
artifacts/label_mapping.json
```

---

# 📁 Dataset Structure

```text
data/

├── Fake.csv
└── True.csv
```

Each dataset contains multiple columns describing the news article.

Typical columns include:

| Column | Description |
|--------|-------------|
| title | News headline |
| text | Full article |
| subject | Article category |
| date | Publication date |

Only the textual information is used for model training.

---

# 🔀 Data Pipeline

The complete data preparation workflow follows the pipeline below:

```text
Raw CSV Files
      │
      ▼
Load Dataset
      │
      ▼
Merge Fake & Real Articles
      │
      ▼
Assign Binary Labels
      │
      ▼
Shuffle Dataset
      │
      ▼
Text Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Train / Test Split
```

---

# ⚖ Class Distribution

The dataset is well balanced, reducing the need for aggressive sampling techniques.

| Class | Samples | Percentage |
|------|---------:|-----------:|
| Fake | 23,481 | 52.30% |
| Real | 21,417 | 47.70% |

The near-balanced distribution helps the model learn both classes effectively without severe class imbalance.

---

# 📝 Sample News Article

### Example (Real)

```text
Reuters - The President met with international leaders today to discuss economic cooperation...
```

↓

Label

```text
Real
```

---

### Example (Fake)

```text
BREAKING!!! Scientists confirm that aliens have secretly taken control of the White House...
```

↓

Label

```text
Fake
```

---

# 💡 Challenges in Fake News Detection

Fake news classification is significantly more difficult than standard text classification due to several challenges:

- Highly similar writing styles
- Political bias
- Clickbait headlines
- Long-form articles
- Named entities
- Ambiguous language
- Mixed factual and misleading content

These challenges motivated the extensive preprocessing, feature engineering, and error analysis performed throughout this project.

---

# 📌 Key Dataset Insights

During exploratory analysis, several important observations were made:

- The dataset is nearly balanced.
- News articles vary greatly in length.
- Many articles contain URLs and HTML artifacts.
- Political topics dominate both classes.
- Vocabulary size exceeds one hundred thousand unique tokens before feature selection.
- TF-IDF significantly improves feature representation.
- SelectKBest effectively removes noisy features while preserving informative terms.

These findings guided the design of the preprocessing pipeline and feature engineering strategy used in the final production model.

---

# 🧹 Data Preprocessing & Feature Engineering

One of the most critical stages in any Natural Language Processing (NLP) project is transforming raw, unstructured text into meaningful numerical representations that machine learning algorithms can understand.

The quality of feature engineering often has a greater impact on model performance than the choice of algorithm itself.

This project follows a structured preprocessing pipeline to maximize information while reducing noise.

---

# 🔄 Complete NLP Pipeline

```text
Raw News Articles
        │
        ▼
Load CSV Files
        │
        ▼
Merge Fake & Real Datasets
        │
        ▼
Assign Binary Labels
        │
        ▼
Text Cleaning
        │
        ▼
Train/Test Split
        │
        ▼
TF-IDF Vectorization
        │
        ▼
SelectKBest (Chi²)
        │
        ▼
Machine Learning Model
```

---

# 📝 Text Cleaning Pipeline

Before feature extraction, every article passes through multiple preprocessing steps.

## Step 1 — Convert to Lowercase

Example

```text
The PRESIDENT Announced New Policies
```

↓

```text
the president announced new policies
```

Purpose

- Reduces vocabulary size
- Treats "President" and "president" as the same token

---

## Step 2 — Remove URLs

Input

```text
Visit https://news.com today.
```

↓

Output

```text
Visit today.
```

Purpose

URLs generally carry little semantic information for fake news classification.

---

## Step 3 — Remove HTML Tags

Input

```html
<p>This is breaking news.</p>
```

↓

Output

```text
This is breaking news.
```

Purpose

HTML tags are formatting artifacts and do not contribute to the article's meaning.

---

## Step 4 — Remove Numbers

Input

```text
COVID cases reached 14500 yesterday.
```

↓

Output

```text
COVID cases reached yesterday.
```

Purpose

Large numerical values often introduce noise and increase vocabulary sparsity.

---

## Step 5 — Remove Punctuation

Input

```text
Breaking!!! New discovery???
```

↓

Output

```text
Breaking New discovery
```

Purpose

Machine learning models benefit from standardized text without punctuation symbols.

---

## Step 6 — Remove Extra Spaces

Input

```text
Machine      Learning
```

↓

Output

```text
Machine Learning
```

Purpose

Produces consistent formatting before vectorization.

---

# 📊 Train / Test Split

The dataset is divided before feature engineering.

| Split | Percentage |
|--------|-----------:|
| Training | 80% |
| Testing | 20% |

Why?

The model should never see testing samples during training.

This ensures a fair estimate of real-world performance.

---

# 🎯 TF-IDF Vectorization

## Why TF-IDF?

Machine Learning models cannot understand raw text.

Instead, every article must be transformed into a numerical feature vector.

TF-IDF (Term Frequency–Inverse Document Frequency) assigns higher importance to informative words while reducing the impact of common words.

Example

| Word | Frequency | Importance |
|------|----------:|-----------:|
| government | High | Medium |
| election | Medium | High |
| the | Very High | Very Low |

---

# 📐 TF-IDF Formula

```text
TF-IDF = TF × IDF
```

Where

TF

```text
Term Frequency
```

IDF

```text
Inverse Document Frequency
```

Words that appear in many documents receive lower weights.

Words unique to a small number of documents receive higher weights.

---

# ⚙️ TF-IDF Configuration

```python
TfidfVectorizer(

    stop_words="english",

    ngram_range=(1,2),

    lowercase=True,

    sublinear_tf=True

)
```

Configuration Summary

| Parameter | Value |
|-----------|-------|
| stop_words | english |
| ngram_range | (1,2) |
| lowercase | True |
| sublinear_tf | True |

---

# 🔍 Why Bigrams?

Instead of using only individual words,

the model also learns two-word phrases.

Example

```text
fake
```

↓

```text
fake news
```

↓

```text
white house
```

↓

```text
new york
```

This significantly improves contextual understanding.

---

# 🎯 Feature Selection

After TF-IDF, the feature space becomes extremely large.

Many features contribute little or no useful information.

To reduce noise and improve efficiency, this project applies:

```text
SelectKBest
```

using

```text
Chi-Square (χ²)
```

---

# 📈 Chi-Square Feature Selection

Purpose

Rank every feature according to its relationship with the target class.

Only the strongest features are retained.

Pipeline

```text
TF-IDF

↓

Thousands of Features

↓

Chi-Square Ranking

↓

Top 10,000 Features

↓

Linear SVM
```

---

# ⚙️ Feature Selection Configuration

```python
SelectKBest(

    score_func=chi2,

    k=10000

)
```

| Parameter | Value |
|-----------|-------|
| Method | Chi-Square |
| Selected Features | 10,000 |

---

# 📦 Final Feature Pipeline

```text
Raw Text
    │
    ▼
Cleaning
    │
    ▼
TF-IDF
    │
    ▼
SelectKBest
    │
    ▼
Sparse Feature Matrix
```

The resulting sparse matrix is then passed directly to the machine learning classifier.

---

# 💡 Why This Pipeline?

This combination was chosen because it offers an excellent balance between:

- High predictive performance
- Fast training
- Fast inference
- Low memory consumption
- Good interpretability
- Scalability for large datasets

Compared to dense embeddings, TF-IDF + Chi-Square remains a strong baseline for classical NLP classification tasks.

---

# ✅ Output of This Stage

After preprocessing and feature engineering, every news article is transformed from raw text into a compact numerical representation containing only the most informative features.

This optimized feature matrix becomes the input for the machine learning models discussed in the next section.

---

# 🧹 Data Preprocessing & Feature Engineering

One of the most critical stages in any Natural Language Processing (NLP) project is transforming raw, unstructured text into meaningful numerical representations that machine learning algorithms can understand.

The quality of feature engineering often has a greater impact on model performance than the choice of algorithm itself.

This project follows a structured preprocessing pipeline to maximize information while reducing noise.

---

# 🔄 Complete NLP Pipeline

```text
Raw News Articles
        │
        ▼
Load CSV Files
        │
        ▼
Merge Fake & Real Datasets
        │
        ▼
Assign Binary Labels
        │
        ▼
Text Cleaning
        │
        ▼
Train/Test Split
        │
        ▼
TF-IDF Vectorization
        │
        ▼
SelectKBest (Chi²)
        │
        ▼
Machine Learning Model
```

---

# 📝 Text Cleaning Pipeline

Before feature extraction, every article passes through multiple preprocessing steps.

## Step 1 — Convert to Lowercase

Example

```text
The PRESIDENT Announced New Policies
```

↓

```text
the president announced new policies
```

Purpose

- Reduces vocabulary size
- Treats "President" and "president" as the same token

---

## Step 2 — Remove URLs

Input

```text
Visit https://news.com today.
```

↓

Output

```text
Visit today.
```

Purpose

URLs generally carry little semantic information for fake news classification.

---

## Step 3 — Remove HTML Tags

Input

```html
<p>This is breaking news.</p>
```

↓

Output

```text
This is breaking news.
```

Purpose

HTML tags are formatting artifacts and do not contribute to the article's meaning.

---

## Step 4 — Remove Numbers

Input

```text
COVID cases reached 14500 yesterday.
```

↓

Output

```text
COVID cases reached yesterday.
```

Purpose

Large numerical values often introduce noise and increase vocabulary sparsity.

---

## Step 5 — Remove Punctuation

Input

```text
Breaking!!! New discovery???
```

↓

Output

```text
Breaking New discovery
```

Purpose

Machine learning models benefit from standardized text without punctuation symbols.

---

## Step 6 — Remove Extra Spaces

Input

```text
Machine      Learning
```

↓

Output

```text
Machine Learning
```

Purpose

Produces consistent formatting before vectorization.

---

# 📊 Train / Test Split

The dataset is divided before feature engineering.

| Split | Percentage |
|--------|-----------:|
| Training | 80% |
| Testing | 20% |

Why?

The model should never see testing samples during training.

This ensures a fair estimate of real-world performance.

---

# 🎯 TF-IDF Vectorization

## Why TF-IDF?

Machine Learning models cannot understand raw text.

Instead, every article must be transformed into a numerical feature vector.

TF-IDF (Term Frequency–Inverse Document Frequency) assigns higher importance to informative words while reducing the impact of common words.

Example

| Word | Frequency | Importance |
|------|----------:|-----------:|
| government | High | Medium |
| election | Medium | High |
| the | Very High | Very Low |

---

# 📐 TF-IDF Formula

```text
TF-IDF = TF × IDF
```

Where

TF

```text
Term Frequency
```

IDF

```text
Inverse Document Frequency
```

Words that appear in many documents receive lower weights.

Words unique to a small number of documents receive higher weights.

---

# ⚙️ TF-IDF Configuration

```python
TfidfVectorizer(

    stop_words="english",

    ngram_range=(1,2),

    lowercase=True,

    sublinear_tf=True

)
```

Configuration Summary

| Parameter | Value |
|-----------|-------|
| stop_words | english |
| ngram_range | (1,2) |
| lowercase | True |
| sublinear_tf | True |

---

# 🔍 Why Bigrams?

Instead of using only individual words,

the model also learns two-word phrases.

Example

```text
fake
```

↓

```text
fake news
```

↓

```text
white house
```

↓

```text
new york
```

This significantly improves contextual understanding.

---

# 🎯 Feature Selection

After TF-IDF, the feature space becomes extremely large.

Many features contribute little or no useful information.

To reduce noise and improve efficiency, this project applies:

```text
SelectKBest
```

using

```text
Chi-Square (χ²)
```

---

# 📈 Chi-Square Feature Selection

Purpose

Rank every feature according to its relationship with the target class.

Only the strongest features are retained.

Pipeline

```text
TF-IDF

↓

Thousands of Features

↓

Chi-Square Ranking

↓

Top 10,000 Features

↓

Linear SVM
```

---

# ⚙️ Feature Selection Configuration

```python
SelectKBest(

    score_func=chi2,

    k=10000

)
```

| Parameter | Value |
|-----------|-------|
| Method | Chi-Square |
| Selected Features | 10,000 |

---

# 📦 Final Feature Pipeline

```text
Raw Text
    │
    ▼
Cleaning
    │
    ▼
TF-IDF
    │
    ▼
SelectKBest
    │
    ▼
Sparse Feature Matrix
```

The resulting sparse matrix is then passed directly to the machine learning classifier.

---

# 💡 Why This Pipeline?

This combination was chosen because it offers an excellent balance between:

- High predictive performance
- Fast training
- Fast inference
- Low memory consumption
- Good interpretability
- Scalability for large datasets

Compared to dense embeddings, TF-IDF + Chi-Square remains a strong baseline for classical NLP classification tasks.

---

# ✅ Output of This Stage

After preprocessing and feature engineering, every news article is transformed from raw text into a compact numerical representation containing only the most informative features.

This optimized feature matrix becomes the input for the machine learning models discussed in the next section.

---

# 🤖 Machine Learning Models

After transforming the news articles into numerical feature vectors using **TF-IDF** and **Chi-Square Feature Selection**, several classical Machine Learning algorithms were trained and compared.

Instead of selecting a model based on popularity, each algorithm was evaluated using multiple performance metrics, computational efficiency, and production suitability.

The following models were implemented and benchmarked.

---

# 🧠 Model Selection Workflow

```text
Feature Matrix
      │
      ▼
Naive Bayes
      │
      ▼
Logistic Regression
      │
      ▼
Linear Support Vector Machine
      │
      ▼
Performance Comparison
      │
      ▼
Hyperparameter Optimization
      │
      ▼
Final Production Model
```

---

# 📌 Model 1 — Multinomial Naive Bayes

## Overview

Multinomial Naive Bayes is one of the most widely used baseline algorithms for text classification.

It estimates the probability of each class using **Bayes' Theorem** under the assumption that features are conditionally independent.

Although this assumption is rarely true in natural language, the model often performs surprisingly well on sparse text representations such as TF-IDF.

---

## Advantages

- Extremely fast training
- Very fast inference
- Low memory usage
- Excellent baseline model
- Works well with sparse matrices

---

## Limitations

- Assumes feature independence
- Limited ability to capture complex relationships
- Usually lower accuracy than discriminative models

---

## Training Configuration

```python
MultinomialNB()
```

---

# 📌 Model 2 — Logistic Regression

## Overview

Logistic Regression is a linear classification algorithm that estimates the probability of each class using the sigmoid function.

Unlike Naive Bayes, Logistic Regression learns a decision boundary directly from the data without assuming feature independence.

It is widely used for binary text classification due to its simplicity, interpretability, and strong performance.

---

## Advantages

- High interpretability
- Robust linear classifier
- Supports regularization
- Produces probability estimates
- Excellent baseline for binary classification

---

## Regularization

The model uses regularization to reduce overfitting.

Supported techniques include:

- L1 Regularization
- L2 Regularization
- ElasticNet

This project uses the default configuration optimized during experimentation.

---

## Training Configuration

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

---

# 📌 Model 3 — Linear Support Vector Machine (Final Model)

## Overview

Linear Support Vector Machine (Linear SVM) was selected as the final production model.

Instead of estimating probabilities directly, Linear SVM searches for the optimal hyperplane that maximizes the margin between classes.

For high-dimensional sparse text representations such as TF-IDF, Linear SVM is widely recognized as one of the strongest classical NLP classifiers.

---

## Why Linear SVM?

Linear SVM demonstrated the best overall performance across all evaluation metrics.

It achieved:

- Highest Accuracy
- Highest F1 Score
- Excellent Precision
- Excellent Recall
- Strong Generalization
- Fast Prediction Speed

These characteristics make it particularly suitable for production environments.

---

## Training Configuration

```python
LinearSVC(
    C=10,
    loss="squared_hinge",
    random_state=42,
    max_iter=10000
)
```

---

# ⚙️ Hyperparameter Optimization

Selecting an appropriate algorithm is only part of the optimization process.

The performance of each model is highly dependent on its hyperparameters.

This project evaluates multiple configurations using systematic search strategies.

---

# 🔎 Grid Search

Grid Search evaluates every possible combination of predefined hyperparameter values.

Example

```text
C = [0.1, 1, 10, 100]
```

Evaluation Process

```text
C = 0.1
↓

Cross Validation

↓

F1 Score

↓

C = 1

↓

Cross Validation

↓

...

↓

Best Parameter
```

Advantages

- Exhaustive search
- Reliable
- Finds the optimal value within the predefined search space

Limitations

- Computationally expensive
- Slow for large search spaces

---

# 🎲 Random Search

Random Search samples random combinations of hyperparameters instead of evaluating every possibility.

Advantages

- Much faster
- Efficient on large search spaces
- Often reaches near-optimal performance

Limitations

- May miss the absolute optimum
- Performance depends on the number of iterations

---

# 🔁 Cross Validation

To reduce bias during model evaluation, every experiment uses **K-Fold Cross Validation**.

```text
Dataset

↓

Fold 1 → Validation

Fold 2 → Validation

Fold 3 → Validation

Fold 4 → Validation

Fold 5 → Validation

↓

Average Performance
```

This strategy provides a more reliable estimate of the model's ability to generalize to unseen data.

---

# 📈 Hyperparameter Optimization Workflow

```text
Training Dataset
        │
        ▼
Choose Model
        │
        ▼
Grid Search / Random Search
        │
        ▼
Cross Validation
        │
        ▼
Evaluate Metrics
        │
        ▼
Best Hyperparameters
        │
        ▼
Final Model
```

---

# 🏆 Model Comparison

The three models were compared using multiple evaluation metrics rather than relying solely on accuracy.

Evaluation criteria included:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Training Time
- Prediction Time
- Model Complexity
- Interpretability

The complete benchmark table is presented in the next section.

---

# 🎯 Final Model Selection

After extensive experimentation and hyperparameter tuning, **Linear Support Vector Machine** was selected as the production model.

Reasons for selection:

- Highest overall performance
- Strong generalization capability
- Excellent balance between Precision and Recall
- Fast inference
- Efficient memory usage
- Well suited for sparse TF-IDF representations
- Stable performance across validation folds

The trained model, vectorizer, selector, configuration, and metadata are automatically exported as production artifacts for deployment.

---

# 📊 Model Evaluation & Performance Analysis

After training and optimizing multiple machine learning models, a comprehensive evaluation was performed using several classification metrics.

Instead of relying solely on **Accuracy**, this project evaluates model performance from multiple perspectives, including precision, recall, F1-score, ROC-AUC, Precision-Recall AUC, confusion matrices, and learning behavior.

This comprehensive evaluation ensures that the selected production model generalizes well to unseen data.

---

# 📈 Benchmark Results

The following table summarizes the performance of all evaluated models.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|--------|---------:|----------:|--------:|----------:|--------:|--------:|
| Naive Bayes | 0.9399 | 0.9390 | 0.9346 | 0.9368 | 0.9815 | 0.9796 |
| Logistic Regression | 0.9840 | 0.9805 | 0.9860 | 0.9832 | 0.9983 | 0.9982 |
| **Linear SVM** ⭐ | **0.9933** | **0.9921** | **0.9939** | **0.9930** | **0.9998** | **0.9998** |

---

# 🏆 Best Model

After comparing all models, **Linear Support Vector Machine** was selected as the production model.

### Reasons

- Highest Accuracy
- Highest F1 Score
- Excellent Precision
- Excellent Recall
- Outstanding ROC-AUC
- Excellent Precision-Recall Performance
- Fast Inference
- Low Memory Consumption

---

# 📌 Evaluation Metrics

The following metrics were used throughout this project.

---

## Accuracy

Measures the proportion of correctly classified samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

Best used when classes are relatively balanced.

---

## Precision

Measures how many predicted positive samples are actually positive.

```text
Precision = TP / (TP + FP)
```

High precision means fewer False Positives.

---

## Recall

Measures how many real positive samples are correctly detected.

```text
Recall = TP / (TP + FN)
```

High recall means fewer False Negatives.

---

## F1 Score

The harmonic mean between Precision and Recall.

```text
F1 = 2 × Precision × Recall
     -----------------------
      Precision + Recall
```

The primary optimization metric used during model selection.

---

## ROC-AUC

Measures the classifier's ability to distinguish between classes across all possible thresholds.

Values close to **1.0** indicate excellent separability.

---

## Precision-Recall AUC

Particularly useful when dealing with imbalanced datasets.

A higher PR-AUC indicates strong precision while maintaining high recall.

---

# 📉 Confusion Matrix

The confusion matrix provides a detailed breakdown of prediction outcomes.

```text
                    Predicted

               Fake      Real

Actual Fake      TP         FN

Actual Real      FP         TN
```

The goal is to maximize the diagonal values while minimizing classification errors.

---

## Confusion Matrix Visualization

> Replace the placeholder below with your generated figure.

```text
assets/images/confusion_matrix.png
```

<p align="center">

<img src="assets/images/confusion_matrix.png" width="650"/>

</p>

---

# 📈 ROC Curve

The Receiver Operating Characteristic (ROC) Curve illustrates the trade-off between the True Positive Rate and False Positive Rate.

An ideal classifier approaches the upper-left corner of the graph.

Linear SVM achieved an ROC-AUC score close to **1.0**, indicating excellent class separability.

<p align="center">

<img src="assets/images/roc_curve.png" width="700"/>

</p>

---

# 📈 Precision–Recall Curve

Precision-Recall curves are especially informative for binary classification tasks.

They demonstrate how precision changes as recall increases.

The production model maintains high precision even at high recall values.

<p align="center">

<img src="assets/images/pr_curve.png" width="700"/>

</p>

---

# 📈 Learning Curve

Learning curves help diagnose whether the model benefits from additional training data.

Key observations:

- Training and validation curves converge.
- Small generalization gap.
- No significant overfitting.
- Stable learning behavior.

<p align="center">

<img src="assets/images/learning_curve.png" width="700"/>

</p>

---

# 📉 Validation Curve

Validation curves illustrate the impact of changing hyperparameters.

The optimal **C** parameter was selected after evaluating multiple values using Cross Validation.

<p align="center">

<img src="assets/images/validation_curve.png" width="700"/>

</p>

---

# 📊 Model Stability

Model robustness was evaluated using K-Fold Cross Validation.

Key findings:

- Consistent performance across folds.
- Low standard deviation.
- Stable decision boundary.
- Excellent generalization capability.

---

# 🎯 Why Accuracy Alone Is Not Enough

Accuracy can be misleading, particularly for imbalanced datasets.

Example:

```text
Dataset

95% Real News

5% Fake News
```

A model predicting **Real** for every article would achieve:

```text
Accuracy = 95%
```

while completely failing to detect fake news.

For this reason, model selection was primarily based on:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

rather than Accuracy alone.

---

# 📌 Performance Summary

The final Linear SVM model demonstrated:

- Excellent predictive accuracy.
- Strong generalization.
- Minimal overfitting.
- Robust cross-validation performance.
- Fast inference suitable for real-time prediction.
- High confidence across both classes.

These characteristics make the model suitable for deployment in production environments.

---
