# 🧠 Tweet Sentiment Classifiers: BERT & DistilBERT

## 📋 Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Results & Visuals](#results--visuals)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

---

## 📝 Overview

This project implements **two tweet sentiment classification pipelines** using HuggingFace Transformers:

✅ **BERT-base binary classifier**  
✅ **DistilBERT binary classifier**  

Both models include:

- Preprocessing: emoji conversion, contraction expansion, punctuation normalization  
- Tokenization: using HuggingFace tokenizers  
- Classification: Binary (Positive vs Negative)  
- Handling class imbalance  
- ROC, Confusion Matrix, t-SNE, and visual analysis  

---

## ✅ Prerequisites

Install the required dependencies:

```bash
pip install torch transformers datasets evaluate contractions scikit-learn matplotlib seaborn textblob emoji wordcloud tqdm
```

The notebook also downloads:
- NLTK tokenizer
- Pretrained BERT & DistilBERT weights

---

## 📂 Project Structure

```bash
tweet-sentiment-bert/
├── bert_classifier.ipynb          # BERT-base model pipeline
├── distilbert_classifier.ipynb    # DistilBERT model pipeline
├── requirements.txt                # Python dependencies
└── LICENSE                         # MIT license
```

---

## 🔄 Pipeline Walkthrough

### 🏗️ Setup & Imports
- Install essential libraries
- Set global seed for reproducibility

### 📥 Data Loading
- Load train / val / test CSV files
- Apply custom tweet preprocessing (emojis, contractions, punctuation, links, etc.)

### ⚙️ Tokenization & Datasets
- HuggingFace tokenizer (`bert-base-uncased` or `distilbert-base-uncased`)
- Custom `Dataset` class  
- DataLoader creation with padding

### 🧠 Model Architecture

**BERT pipeline:**  
- BERT base encoder  
- Dropout  
- Linear classifier  

**DistilBERT pipeline:**  
- DistilBERT encoder  
- Dropout  
- Linear classifier  

### ⚖️ Handling Imbalanced Classes
- Compute class weights with scikit-learn  
- Use weighted CrossEntropy loss

### 🚀 Training Loop
- Optimizer: AdamW  
- Scheduler: Cosine with Restarts  
- Evaluation after each epoch  
- Track accuracy, loss, F1-score  
- Save best model

### 📊 Visual Analysis
- ROC Curve  
- Confusion Matrix  
- Word Clouds (Raw / Clean)  
- t-SNE of CLS embeddings  
- Sentiment polarity distribution  
- Emoji frequency  
- Length distributions  

### 📝 Submission
- Generate `submission.csv`

---

## 📊 Results & Visuals

### BERT-base

| Metric          | Value  |
|-----------------|--------|
| Accuracy        | ~0.85  |
| Precision       | ~0.85  |
| Recall          | ~0.85  |
| F1‑Score        | ~0.85  |

### DistilBERT

| Metric          | Value  |
|-----------------|--------|
| Accuracy        | ~0.85  |
| Precision       | ~0.85  |
| Recall          | ~0.85  |
| F1‑Score        | ~0.85  |

### Sample Visuals
✅ Word clouds (Raw vs Cleaned)  
✅ Top emoji frequency  
✅ Sentiment polarity histogram  
✅ ROC curve (AUC shown)  
✅ Confusion matrix  
✅ t-SNE plots of [CLS] embeddings  
✅ Learning curves  

---

## 🚀 Usage

### Clone the repo

```bash
git clone https://github.com/Alphawastaken/tweet-sentiment-classifiers-bert-distilbert.git
cd tweet-sentiment-classifiers-bert-distilbert
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Notebooks

```bash
jupyter notebook
```

Then open:

✅ `bert.ipynb`  
✅ `Distilbert.ipynb`  

And **Run All** cells.

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for full terms.

---

## 📬 Contact
  GitHub: [Vaios-Panagiotou]  
