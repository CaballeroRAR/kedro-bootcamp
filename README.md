# Credit Risk Analysis - Kedro Bootcamp

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This project implements credit risk modeling using machine learning pipelines built on the Kedro framework.

### Dataset Context
The project utilizes payment data from October 2005 of an important bank (a cash and credit card issuer) in Taiwan. The target population consists of credit card holders of the bank.

This work is based on the research case:
> Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications, 36(2), 2473-2480. 
> Paper link: [Semantic Scholar](https://www.semanticscholar.org/paper/The-comparisons-of-data-mining-techniques-for-the-Yeh-Lien/1cacac4f0ea9fdff3cd88c151c94115a9fddcf33)

- **Total Observations**: 25,000
- **Default Payments**: 5,529 (22.12%)
- **Response Variable**: `default payment` (binary: 1 = Yes/Default, 0 = No/Non-default)

### Feature Definitions
- **X1**: Amount of the given credit (NT dollar). Includes individual consumer credit and family (supplementary) credit.
- **X2**: Gender (1 = male; 2 = female).
- **X3**: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
- **X4**: Marital status (1 = married; 2 = single; 3 = others).
- **X5**: Age (year).
- **X6–X11**: History of past payment (tracked monthly from April to September 2005):
  - X6 = repayment status in September, 2005
  - X7 = repayment status in August, 2005
  - ...
  - X11 = repayment status in April, 2005
  - *Measurement Scale:* -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; ...; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
- **X12–X17**: Amount of bill statement (NT dollar):
  - X12 = bill statement amount in September, 2005
  - X13 = bill statement amount in August, 2005
  - ...
  - X17 = bill statement amount in April, 2005
- **X18–X23**: Amount of previous payment (NT dollar):
  - X18 = amount paid in September, 2005
  - X19 = amount paid in August, 2005
  - ...
  - X23 = amount paid in April, 2005

---

## Kedro Project Structure & Data Flow

This section explains the relationship between the YAML configurations, the pipeline registry, and the Python logic in this project.

### 1. Data Catalog (`conf/base/catalog.yml`)
The [catalog.yml](conf/base/catalog.yml) registers your datasets. It maps logical names used in the code to physical storage.
*   **Logic:** When a node refers to a dataset name (e.g., `train_ready_catboost`), Kedro uses this file to determine the file path and dataset type (e.g., `pandas.ParquetDataset`).

### 2. Parameters (`conf/base/parameters.yml`)
Configuration values and hyperparameters are stored in [parameters.yml](conf/base/parameters.yml). 
*   **Logic:** These are injected into pipelines using the `params:` prefix or as the `parameters` dictionary.

### 3. Pipeline Definitions (`src/credit_risk_project/pipelines/`)
Pipelines define the execution graph (DAG).
*   **Nodes:** Call Python functions from `nodes.py`.
*   **Inputs/Outputs:** Link to either the **Catalog** (for persistent data) or **Memory** (for intermediate results).

### 4. Pipeline Registry (`src/credit_risk_project/pipeline_registry.py`)
This file is the entry point for the Kedro CLI.
*   **Logic:** It maps string aliases to pipeline objects.
*   **CLI Mapping:**
    *   `kedro run` executes the `__default__` pipeline.
    *   `kedro run --pipeline=feature_eng` executes the `feature_eng` alias.
    *   `kedro run --pipeline=training` executes the `training` alias.
