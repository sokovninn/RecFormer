# RecFormer - Recipe Transformer - KAIST AI506 term project

This project explores several strategies to develop the most accurate
models to solve two problems: identifying a cuisine when a recipe
is given and inferring an ingredient missing from a recipe. Therefore, we focus on two tasks: classification and completion. As an
exploratory stage, we implement two categories of models: baseline
and target. Baseline models consist of Logistic Regression, Random
Forest Classifier, Multinomial Naive Bayes, Linear Support Vector
Classifier, and Collaborative Filtering, while our target model is
based on Transformer architecture. We propose a model named
RecFormer - Recipe Transformer - that can perform both
classification and completion tasks in a single model with
similar or better accuracy than our baseline models. Finally,
we employ model blending to achieve maximum performance. An
ensemble of 2 models, LSVC and RecFormer, achieves **79.20%** and
**14.74%** validation accuracy for classification and completion tasks.

Further details are available in the [report](data/RecFormer_report.pdf) and [presentation](data/RecFormer_presentation.pdf).


## Model architecture:


![Alt text](/data/RecFormer_architecture.png "Optional Title")

## Requirements
You must have the `pytorch, tensorflow, numpy, pandas, sklearn, matplotlib, jupyterlab` libraries installed.

## Running
- Train and evaluate RecFormer: [recipe_transformers.ipynb](recipe_transformers.ipynb)
- Train and evaluate baselines: [baselines/BaseLineModelExperiment_final.ipynb](baselines/BaseLineModelExperiment_final.ipynb)


