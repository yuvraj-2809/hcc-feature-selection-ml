# Feature Reduction for Hepatocellular Carcinoma using HHO and Adaptive Ensemble Learning

## Overview
This project focuses on predicting hepatocellular carcinoma (HCC) survival outcomes using a hybrid machine learning framework.

The approach combines:
- Statistical feature filtering
- Harris Hawks Optimization (HHO)
- Adaptive ensemble learning (Bagging vs Boosting)

---

## Key Features
- Multi-stage feature reduction
- Metaheuristic optimization using HHO
- Adaptive model selection based on F1-score
- Stratified 5-fold cross-validation
- Feature stability analysis

---

## Methodology
1. Data preprocessing and normalization  
2. Statistical filtering (top features selection)  
3. HHO-based feature selection  
4. Ensemble model training:
   - Bagging
   - Boosting  
5. Best model selection per fold  
6. Performance evaluation  

---

## 📊 Results
- Accuracy: **73.33 ± 5.42 %**
- F1 Score: **78.20 ± 5.37 %**
- Feature reduction: **~62%**
- Average selected features: **18**

---

## Key Insights
- Significant reduction in feature space improves interpretability
- Adaptive ensemble improves robustness
- Stable feature selection across folds

---

## Tools & Technologies
- MATLAB
- Machine Learning Toolbox
- Optimization Techniques

---

## Dataset
The dataset is sourced from the UCI Machine Learning Repository and contains clinical attributes for HCC prediction.

---

## Research Paper
The detailed methodology and results are documented in the attached paper:

📎 See: `HCC_Dissertation_Paper.pdf`

---

## Author
Yuvraj Singh
