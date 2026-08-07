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
