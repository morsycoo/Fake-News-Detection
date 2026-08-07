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

> **Repository Status:** Production Ready ✅
>
> **Primary Language:** Python
>
> **Machine Learning Framework:** Scikit-Learn
>
> **Deployment:** FastAPI + Docker
>
> **Testing:** Pytest + GitHub Actions
