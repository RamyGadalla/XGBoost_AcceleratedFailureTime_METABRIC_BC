# AFT Survival Modeling with XGBoost on METABRIC Dataset

## Overview

This project involves using the **Accelerated Failure Time (AFT)** model with **XGBoost** to predict **survival times** for breast cancer patients based on clinical characteristics from the **METABRIC** dataset. The METABRIC dataset provides a comprehensive set of clinical and genetic information, which is used here to model patient survival outcomes.

The goal of this project is to estimate **how long a patient will survive** given specific clinical features such as age at diagnosis, tumor histology, hormone receptor status and others.&#x20;

## Dataset

The dataset used in this project is the **METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)** dataset, which was sourced from Kaggle: [Breast Cancer METABRIC Data](https://www.kaggle.com/code/alexandervc/breast-cancer-metabric-data-survival-curves/notebook).

### Data Preprocessing

1. **Feature Selection and Encoding**:
   - Removed unnecessary columns, such as `ID` and other non-informative features.
   - **Categorical features** were encoded using **Label Encoding**.
2. **Imputation**:
   - Missing values were imputed using **KNN Imputer**.
3. **Handling Right-Censored Data**:
   - The labels for survival analysis were split into **lower** and **upper bounds** to handle right-censored observations, where an event may not have occurred by the end of the observation period.

## Model Description

The **XGBoost** model was trained using its native interface to implement an **AFT model** for survival analysis. This approach helps in modeling survival times directly, taking into account the censored nature of the data.

### Hyperparameter Optimization

- A **random search** approach was used to optimize the hyperparameters of the XGBoost model.
- The following hyperparameters were optimized:
  - `max_depth`, `learning_rate`, `aft_loss_distribution_scale`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_weight`, and `early_stopping_rounds`.
- The model training was conducted with **early stopping** to prevent overfitting.

## Model Evaluation

The following metrics were used to evaluate the performance of the model:

1. **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)**: Used to measure the average squared difference and its root between the predicted survival times and the actual observed times.
2. **Concordance Index (C-Index)**: Used to evaluate how well the predicted survival times were ranked compared to the actual survival times. The C-Index is particularly well-suited for survival analysis as it measures the **ranking correlation** between predicted and observed survival times.

## Explainability

To understand the contributions of individual features towards survival predictions, **SHAP (SHapley Additive exPlanations)** values were calculated.

- **Summary Plot**: Visualizes the average impact of each feature on the model predictions.
- **Dependence Plot**: Shows interactions between specific features, helping understand how combinations of features affect patient survival.

## Visualizations

- **Feature Importance**: Plots the top 10 most important features contributing to survival predictions.
- **SHAP Dependence Plots**: Show how individual features, such as `Tumor Stage` and `Oncotree Code` , influence the survival predictions.

## Installation and Setup

To run this project, make sure you have the following dependencies installed:

- Python 3.x
- XGBoost
- Pandas
- Scikit-Learn
- Lifelines
- SHAP
- Seaborn
- Matplotlib



## Environment Setup

To recreate the environment used for this project, you can use the provided YAML file to create a conda environment. Run the following command:

```sh
conda env create -f environment.yml
```

This will create an environment with all the necessary dependencies specified in the `environment.yml` file.

## Acknowledgments

This project uses data from the [Breast Cancer METABRIC Data](https://www.kaggle.com/code/alexandervc/breast-cancer-metabric-data-survival-curves/notebook), available on Kaggle.
