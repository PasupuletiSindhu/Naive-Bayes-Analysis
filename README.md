# Naive Bayes Analysis

This repository contains a Jupyter Notebook that explores the application of the Naive Bayes algorithm, including preprocessing of text data, training, and evaluation of multiple versions of the classifier. The dataset is processed into a feature table, and different variants of the Naive Bayes classifier are compared.

## Contents

- Main notebook implementing the entire pipeline
- Preprocessing and feature extraction
- Naive Bayes classification
- Laplace smoothing
- Model evaluation

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- nltk
- scikit-learn

You can install the required packages using pip:

```bash
pip install pandas numpy nltk scikit-learn
```

### Running the Notebook

1. Open the notebook:
   ```bash
   jupyter notebook ProblemSet3.ipynb
   ```
2. Run all cells step by step or use the "Restart & Run All" option in the Kernel menu.

## Notebook Structure

### 1. Initializing and Preprocessing the Dataset
- Reads the input dataset.
- Cleans and tokenizes the data using nltk.
- Converts text into a feature table for classification.

### 2. Naive Bayes Algorithm
- Implements the standard version of the Naive Bayes classifier.
- Trains the model on the preprocessed dataset.
- Evaluates the model's performance.

### 3. Naive Bayes with Laplace Smoothing
- Enhances the standard Naive Bayes classifier using Laplace smoothing to handle zero-frequency problems.

### 4. Results and Evaluation
- Compares accuracy and precision of different models.
- Uses standard classification metrics for evaluation.

## Results

The notebook provides output for:
- Feature extraction
- Confusion matrix
- Accuracy, precision, and recall of the models
