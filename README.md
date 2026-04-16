# Kaggle Intermediate Machine Learning

This repository contains my work from the Kaggle Intermediate Machine Learning course. 
It builds on the Intro to Machine Learning course and focuses on improving model performance through better preprocessing, pipelines, and advanced models like XGBoost.

## Topics Covered

- Handling missing values
- Working with categorical variables
- Pipelines and preprocessing
- Cross-validation
- XGBoost and hyperparameter tuning (Optuna)
- Data leakage

## Key Concepts and Skills Learned

- One-hot vs ordinal encoding
- Pipelines for clean preprocessing
- Cross-validation for reliable evaluation
- Hyperparameter tuning (manual + Optuna)
- Boosting methods (XGBoost)
- Deeper understanding of ML concepts
- Comprehending documentation and applying to practice

## How to setup
**1. Clone the repository:**  

SSH
```bash
git clone git@github.com:notduey/KaggleIntermediateML.git
```
HTTPS
```bash
git clone https://github.com/notduey/KaggleIntermediateML.git
```
**2. CD into the directory:**
```bash
cd.../.../KaggleIntermediateML
```
**3. Create a virtual environment (recommended):**
```
python -m venv .venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```
**4. Install dependencies:**
```
pip install -r requirements.txt
```

⚠️ macOS Users (XGBoost Dependency)

If you encounter an error related to `libomp` when using XGBoost, install it with:

```bash
brew install libomp
```

## Notes
- The lessons/ scripts contain clean implementations of each concept.
- The notebooks/ directory includes more detailed experimentation and explanations.
- The src/utilities.py file contains reusable helper functions (e.g., model scoring).
- I used the Optuna library to to automatically search for better hyperparameters using cross-validation, it wasn't mentioned in the Kaggle course but was implemented as a more efficient way to find the optimal hyperparameters.
