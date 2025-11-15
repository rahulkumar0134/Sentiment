# 🎯 SentimentScope: YouTube Comment Sentiment Analyzer

## 🚀 [Live Demo](https://sentimentscope-s8shgt9krsvzsglgcwcjmd.streamlit.app/)

**SentimentScope** is a real-time sentiment analysis tool for YouTube videos. It fetches comments using the YouTube Data API and analyzes them using a deep learning model built with NLP libraries like SpaCy, NLTK, and a BiLSTM architecture powered by TensorFlow/Keras. The application is deployed with a clean and interactive Streamlit UI.

---

## 🧠 Features

- 🔍 Fetches **live YouTube comments** using YouTube Data API
- 🧾 Cleans, preprocesses, and tokenizes comments
- 💬 Analyzes sentiment (Positive / Negative / Neutral)
- ⚡ Built using **BiLSTM neural network**
- 📊 Displays **charts** showing sentiment distribution
- 💻 Web-based interface using **Streamlit**

---

## 📂 Dataset

- **Dataset Used**: Publicly available YouTube comments dataset
- **Download Link**: [Click here to access the dataset](link) <!-- Replace `link` with your actual URL -->

---

## 🧰 Tech Stack

| Layer        | Technology                            |
|--------------|----------------------------------------|
| Frontend     | Streamlit                              |
| Backend      | Python                                 |
| NLP          | NLTK, SpaCy                            |
| ML Model     | BiLSTM using TensorFlow + Keras        |
| Visualization| Matplotlib, Seaborn                    |
| API          | YouTube Data API v3                    |

---

## 🔬 Model Overview

- **Architecture**: Bidirectional LSTM
- **Embedding Layer**: Pre-trained word embeddings (GloVe)
- **Accuracy**: Achieved **high performance** on validation set
- **Preprocessing**:
  - Removal of stop words
  - Lemmatization (SpaCy)
  - Tokenization & padding

---

## ⚙️ Installation & Running

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SentimentScope.git
   cd SentimentScope
