# Credit Card Fraud Detection (Unsupervised Learning)

This project analyzes credit card transaction data for the purpose of fraud detection using **unsupervised machine learning techniques**. The dataset was generated through a multi-agent virtual world simulation developed by IBM and includes user details, transaction metadata, merchant info, and fraud/error flags.

---

##  Objective
Apply and compare **unsupervised anomaly detection models** (LOF, Isolation Forest, One-Class SVM) and **deep learning methods** (Stacked Autoencoder and Variational Autoencoder) in classifying fraudulent versus normal transactions.

---

## 3 Preprocessing & Methodology
- **Cleaning**:  Dropped irrelevant features.
- **Feature Engineering**: Split time into hours/minutes; handled categorical fields as strings.
- **Scaling**: Applied `StandardScaler` on benign (non-fraudulent) data only.
- **Training Setup**:
  - Trained on benign samples only.
  - Tested on both benign test data and all fraudulent transactions.
- **Evaluation Metrics**: Precision, Recall, F1 Score, ROC-AUC.

---

##  Tools Used
- Python, Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras

---

##  Results

### Traditional Models

| **Model**    | **Parameter** | **Precision** | **Recall** | **F1 Score** | **ROC AUC (labels)** | **ROC AUC (scores)** |
|-------------|----------------|---------------|------------|--------------|------------------------|------------------------|
| LOF         | n = 25         | 0.0548        | 0.2963     | 0.0925       | 0.6377                 | 0.7953                 |
| Isolation Forest | n = 500    | 0.0162        | 0.5926     | 0.0315       | 0.7223                 | 0.8126                 |
| One-Class SVM | RBF kernel   | 0.0076        | 0.9259     | 0.0150       | 0.7140                 | 0.8399                 |

---

### Deep Learning Models

#### Stacked Autoencoder (SAE)

| **Threshold (Percentile)** | **Precision** | **Recall** | **F1 Score** | **ROC AUC Score** |
|----------------------------|---------------|------------|--------------|--------------------|
| 90th                       | 0.0182        | 0.4444     | 0.0350       | –                  |
| 95th                       | 0.0208        | 0.2593     | 0.0385       | –                  |
| 99th                       | 0.0000        | 0.0000     | 0.0000       | 0.8258             |

#### Variational Autoencoder (VAE)

| **Threshold (Percentile)** | **Precision** | **Recall** | **F1 Score** | **ROC AUC Score** |
|----------------------------|---------------|------------|--------------|--------------------|
| 25th                       | 0.0051        | 0.9259     | 0.0101       | –                  |
| 50th                       | 0.0076        | 0.9259     | 0.0151       | –                  |
| 95th                       | 0.0218        | 0.2593     | 0.0402       | 0.8324             |

---

##  Summary
- **Best traditional model**: One-Class SVM (RBF kernel) with ROC AUC of **0.8399**.
- **Best deep model**: VAE with 95th percentile threshold, ROC AUC of **0.8324**.
- Most models achieved **high recall but low precision**, which is common for imbalanced fraud datasets.

---

##  Future Work
- Improve feature selection and address class imbalance.
- Experiment with ensemble and hybrid approaches.
- Tune hyperparameters more thoroughly.
- Consider semi-supervised models with partial labels.

