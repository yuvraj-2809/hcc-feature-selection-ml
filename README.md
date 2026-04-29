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

## Experimental Results

The model was evaluated using stratified 5-fold cross-validation.

- Accuracy: **73.33 ± 10.37 %**
- F1 Score: **79.10 ± 8.68 %**
- Average selected features: **~16**

### Sample Output

![Results](results_output.png)
The variation in performance across folds reflects the complexity and variability of clinical datasets, highlighting the importance of robust feature selection and adaptive modeling.
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
