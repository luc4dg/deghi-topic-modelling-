# Deghi Trustpilot Topic Modeling

Topic modeling pipeline for analyzing Trustpilot reviews of **Deghi**, combining text preprocessing, sentence embeddings, clustering with **BERTopic**, and **LLM-based topic labeling**.

---

## 📌 Project Overview
This project applies **Natural Language Processing (NLP)** techniques to customer reviews in order to extract recurring themes, identify pain points, and provide actionable insights.

**Main steps:**
1. **Data loading** from Trustpilot reviews (`deghi_dataset.csv`).
2. **Text preprocessing**:
   - Lowercasing
   - Removing punctuation, numbers, and symbols
   - Cleaning stopwords
3. **Chunking**: splitting long reviews into smaller sentence blocks using **spaCy**.
4. **Embeddings**: generating dense vector representations with `paraphrase-multilingual-MiniLM-L12-v2`.
5. **Topic Modeling**: 
   - Dimensionality reduction with **UMAP**
   - Clustering with **HDBSCAN**
   - Topic refinement with **ClassTfidfTransformer** and `CountVectorizer`
6. **Topic Labeling**:
   - **KeyBERTInspired** for keyword-based labels
   - **Mistral LLM** (`mistralai/Mistral-Nemo-Instruct-2407`) for short descriptive labels
7. **Visualization**:
   - Topic hierarchy
   - 2D embedding map with **datamapplot**

---

## 📂 Repository Structure
