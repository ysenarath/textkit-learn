# KBERT - Feature Selection Methods

The `KBertTokenizer` class uses feature selection methods to score and rank predicate-object pairs (triples) extracted from text. This page describes the three available methods and their advantages and disadvantages.

## Overview

When augmenting text with knowledge triples, not all triples are equally useful for classification. Feature selection methods help identify which triples are most relevant by scoring them based on their relationship with class labels.

## Methods

### Chi-Square (chi2_score)

Chi-Square measures the statistical dependence between predicate-object pairs and class labels using the chi-squared test.

**Advantages**

- **Fast computation**: Chi-square is computationally efficient, making it suitable for large datasets.
- **Interpretable**: The scores have a clear statistical interpretation based on the chi-squared distribution.
- **Works well with term counts**: Naturally suited for text classification where features are word or term frequencies.
- **No assumptions about feature distribution**: Does not require features to follow a specific distribution.

**Disadvantages**

- **Assumes feature independence**: Does not account for correlations between features, which may lead to selecting redundant features.
- **Requires non-negative values**: Only works with non-negative feature values (e.g., term frequencies or counts).
- **Sensitive to sample size**: Results can be unreliable with small sample sizes or sparse data.
- **Linear relationships only**: Captures only linear dependencies between features and class labels.

---

### Mutual Information (mutual_info)

Mutual Information quantifies the amount of information a feature provides about the class label, based on information theory.

**Advantages**

- **Captures non-linear dependencies**: Unlike chi-square, mutual information can detect both linear and non-linear relationships between features and class labels.
- **Information-theoretic foundation**: Grounded in well-established information theory principles.
- **Feature independence not required**: Can identify informative features regardless of their correlations with other features.
- **Scale-invariant**: Not affected by linear transformations of the feature values.

**Disadvantages**

- **Computationally more expensive**: Requires density estimation, which can be slower than chi-square, especially for large datasets.
- **Sensitive to binning/discretization**: For continuous features, the results depend on how features are discretized.
- **Estimation challenges**: Accurate estimation requires sufficient samples; may be unreliable with sparse data.
- **No built-in redundancy handling**: Does not inherently account for redundancy among selected features.

---

### SVM Coefficients (svm_score)

SVM Coefficients uses the absolute values of LinearSVC (linear Support Vector Machine) coefficients as feature importance scores.

**Advantages**

- **Considers feature interactions**: Through margin optimization, SVM implicitly considers how features work together for classification.
- **Direct classification relevance**: Provides feature importance directly tied to classification performance.
- **Robust to outliers**: SVMs are generally robust to outliers due to the margin-based optimization.
- **Effective in high-dimensional spaces**: Works well when the number of features is large relative to the number of samples.

**Disadvantages**

- **Requires training a model**: More computationally expensive as it requires training a full SVM classifier.
- **May not generalize well**: The importance scores are specific to the linear SVM model and may not reflect importance for other classifiers.
- **Sensitive to hyperparameters**: Results depend on regularization parameters (e.g., C parameter).
- **Assumes linear separability**: Uses a linear kernel, which may not capture complex non-linear relationships.

## Comparison Table

| Method             | Computation Speed | Non-linear Dependencies | Feature Interactions | Model-Free |
| ------------------ | ----------------- | ----------------------- | -------------------- | ---------- |
| Chi-Square         | Fast              | No                      | No                   | Yes        |
| Mutual Information | Moderate          | Yes                     | No                   | Yes        |
| SVM Coefficients   | Slow              | No                      | Partial              | No         |

## Recommendations

- **Use Chi-Square** when you need fast feature selection and your data has clear linear relationships between features and class labels.
- **Use Mutual Information** when you suspect non-linear relationships between features and class labels, and computational cost is not a primary concern.
- **Use SVM Coefficients** when you want feature importance that directly reflects classification relevance and are willing to invest in training time.
