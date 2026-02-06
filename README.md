# Bias and Fairness Analysis in Income Prediction

This project studies bias and fairness in machine learning models for income
prediction using the Folktables (ACS Income) dataset. The goal is to understand
whether commonly used models treat demographic groups fairly and how fairness
behaves when models are improved or deployed in new settings.

Rather than focusing only on accuracy, this project explicitly evaluates
fairness metrics and examines how they change with model choice and data
distribution shift.

---

## Motivation

Machine learning systems are increasingly used in high-stakes decision-making
domains such as hiring, lending, and income estimation. While these systems are
often optimized for predictive accuracy, high accuracy alone does not guarantee
fair treatment across demographic groups.

This project is motivated by the following questions:
- Can a standard baseline model exhibit bias even if it performs well?
- Does using a stronger model automatically reduce bias?
- Do fairness properties generalize when a model is deployed on new data?

---

## Dataset

- **Dataset:** Folktables (ACS Income, 2018)
- **Prediction Task:** Binary classification of income (> $50K)
- **Sensitive Attribute:** Sex (Male = 1, Female = 2)
- **Training State:** California (CA)
- **Deployment / Test State:** Texas (TX)

The Folktables dataset is derived from U.S. Census data and is commonly used in
fairness research due to its real-world relevance and demographic attributes.

---

## Baseline Model and Fairness Analysis

We begin with a **Logistic Regression** model trained on California data. This
model is chosen as a simple and interpretable baseline.

The baseline model is evaluated using:
- **Accuracy**
- **Equal Opportunity (EO):** True Positive Rate gap between groups
- **Demographic Parity (DP):** Difference in positive prediction rates

### Baseline Results (California)

- Accuracy: ~0.79
- EO Gap: ~0.12
- DP Gap: ~0.16

These results show that even a standard baseline model with reasonable accuracy
exhibits significant gender-based disparities.

---

## Experimental Extensions

Building on the baseline analysis, we conduct several experiments to study how
fairness behaves under different conditions.

---

### 1. Model Improvement

A **Gradient Boosting** classifier is trained on the same California data and
compared against the baseline Logistic Regression model.

#### Results (California – Gradient Boosting)
- Accuracy increases to ~0.81
- Equal Opportunity gap decreases significantly
- Demographic Parity gap decreases slightly

This experiment shows that improved modeling can sometimes reduce bias by
reducing underfitting, but fairness is not guaranteed by accuracy alone.

---

### 2. Fairness Mitigation

We apply a simple post-processing mitigation strategy using group-specific
decision thresholds. The threshold for the disadvantaged group (Female) is
lowered to reduce the Equal Opportunity gap.

This experiment illustrates the **trade-off between fairness and accuracy** and
demonstrates that fairness interventions often require explicit design choices.

---

### 3. Cross-State Deployment Evaluation

To simulate real-world deployment, the Gradient Boosting model trained on
California data is evaluated on Texas data **without retraining**.

This experiment tests whether fairness improvements observed in-distribution
generalize under data distribution shift.

#### Results (Texas – CA-trained Gradient Boosting)
- Accuracy remains high (~0.79)
- Equal Opportunity gap remains relatively small
- Demographic Parity gap increases

These results highlight that fairness properties are **not stable** across
populations and can change under deployment, even when accuracy remains strong.

---

## Key Observations

- A simple baseline model can exhibit substantial bias
- Improving model accuracy can reduce bias in some cases, but not reliably
- Different fairness metrics behave differently under distribution shift
- Fairness observed during training does not necessarily generalize at deployment
- Fairness must be explicitly evaluated and monitored

---

---

## Reproducibility

All experiments are run on CPU and do not require a GPU. Results can be
reproduced by running the notebooks in order.

---

## Conclusion

This project demonstrates that fairness is not an automatic byproduct of model
accuracy. While better models can sometimes reduce bias, fairness is sensitive
to metric choice and deployment context. Explicit fairness evaluation is
necessary for responsible machine learning systems.

---

## Notes

This project prioritizes clarity, interpretability, and real-world relevance
over model complexity.


