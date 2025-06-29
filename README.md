# Credit Risk Probability Model for Alternative Data

An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model

## Overview

This project develops a credit scoring model for Bati Bank, enabling their buy-now-pay-later service in partnership with an eCommerce company. The model transforms behavioral data (Recency, Frequency, and Monetary patterns) into predictive risk signals to evaluate potential borrowers.

## Credit Scoring Business Understanding

### Basel II Accord's Influence on Model Requirements

The Basel II Capital Accord emphasizes robust risk measurement frameworks, requiring financial institutions to maintain transparent and auditable risk assessment models. This directly influences our need for an interpretable and well-documented model in several ways:

1. **Regulatory Compliance**: Basel II requires banks to document their risk assessment methodologies thoroughly, including model development, validation procedures, and underlying assumptions.

2. **Risk Sensitivity**: Our model must demonstrate sensitivity to varying risk levels, with clear documentation of how different factors contribute to risk assessment.

3. **Transparency**: Regulators need visibility into how credit decisions are made, necessitating model interpretability to explain individual risk assessments.

4. **Validation Requirements**: Basel II mandates regular model validation, which is significantly easier with interpretable models where performance can be monitored and understood clearly.

### Proxy Variable Necessity and Business Risks

Since we lack direct "default" labels in our eCommerce behavioral data, creating a proxy variable is necessary, though it comes with specific business risks:

1. **Necessity for Proxy Variables**:
   - Traditional credit scoring relies on historical default data that we don't have for new eCommerce customers
   - Behavioral patterns (RFM metrics) can serve as effective proxies for creditworthiness
   - Without proxies, we couldn't leverage the rich behavioral data available to predict future payment behavior

2. **Business Risks of Proxy-Based Predictions**:
   - **Misalignment Risk**: The proxy may not perfectly correlate with actual repayment behavior
   - **Sampling Bias**: The proxy might accurately represent only certain customer segments
   - **Temporal Validity**: The relationship between proxy and actual default risk may change over time
   - **Model Drift**: Market conditions or consumer behavior changes could invalidate assumptions behind the proxy
   - **Regulatory Scrutiny**: Proxies must be demonstrably fair and non-discriminatory to meet compliance requirements

### Trade-offs: Simple vs. Complex Models in Financial Context

In a regulated financial context, there are significant trade-offs between simple, interpretable models and complex, high-performance models:

1. **Simple, Interpretable Models (e.g., Logistic Regression with WoE)**:
   - **Advantages**: Easy to interpret, explain to stakeholders and regulators; clear variable impact; straightforward implementation; established regulatory acceptance
   - **Disadvantages**: Potentially lower predictive power; may miss complex interactions; requires more manual feature engineering

2. **Complex, High-Performance Models (e.g., Gradient Boosting)**:
   - **Advantages**: Higher predictive accuracy; automatic feature interaction detection; better handling of non-linear relationships; potentially stronger discrimination between risk levels
   - **Disadvantages**: "Black box" nature creates regulatory challenges; difficult to explain decisions; harder to diagnose biases; requires more robust validation frameworks

3. **Key Considerations**:
   - Regulatory environments often prioritize interpretability over marginal performance gains
   - The cost of false negatives (approving bad risks) vs. false positives (declining good risks)
   - Institutional capability to explain and defend complex models
   - Model validation and monitoring requirements
   - Implementation and maintenance complexity

For this project, we'll implement a balanced approach that optimizes for both interpretability and performance, ensuring regulatory compliance while maximizing business value.

## Project Structure

```
.
├── .github/workflows/   # CI/CD pipeline
├── data/                # Data folder (added to .gitignore)
│   ├── raw/             # Original, immutable data
│   └── processed/       # Processed data used for modeling
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Source code
│   ├── __init__.py
│   ├── data_processing.py  # Data processing scripts
│   ├── train.py           # Model training functionality
│   ├── predict.py         # Inference functionality
│   └── api/              # FastAPI application
│       ├── main.py
│       └── pydantic_models.py
├── tests/               # Unit tests
├── Dockerfile           # Container definition
├── docker-compose.yml   # Service definition
├── requirements.txt     # Project dependencies
├── .gitignore           # Files to ignore in git
└── README.md            # Project documentation
```

## Getting Started

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 