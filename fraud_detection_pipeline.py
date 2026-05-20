# fraud_detection_pipeline.py
"""
End-to-end pipeline for Online Payment Fraud Detection.

Usage (script):
    python fraud_detection_pipeline.py --data data/your_dataset.csv --out model.joblib

Or open as a notebook and split into cells:
  - EDA cell
  - Preprocessing / modeling cell
  - Evaluation cell
  - Save / Flask demo cell
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, precision_recall_curve, auc)
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ----------------------------
# Helper / Feature Functions
# ----------------------------
def add_features(df):
    """
    Add derived features and drop obvious IDs that leak.
    Expects columns similar to the dataset:
      step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
      nameDest, oldbalanceDest, newbalanceDest, isFraud
    """
    df = df.copy()
    # Basic derived numeric features
    df['deltaOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['deltaDest'] = df['newbalanceDest'] - df['oldbalanceDest']
    # Avoid divide-by-zero
    df['amt_to_oldbal_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1.0)
    # flag for same account (could signal internal transfers)
    df['is_same_account'] = (df['nameOrig'] == df['nameDest']).astype(int)
    # if 'step' is hourly, extract hour-of-day; otherwise it's a generic time-step feature
    try:
        df['hour'] = df['step'] % 24
    except Exception:
        df['hour'] = 0
    # Drop raw identifiers that likely leak (keep only engineered flags)
    drop_cols = []
    if 'nameOrig' in df.columns:
        drop_cols.append('nameOrig')
    if 'nameDest' in df.columns:
        drop_cols.append('nameDest')
    if 'step' in df.columns:
        # we created hour; keep step only if you want time order; for modeling we drop
        drop_cols.append('step')
    df = df.drop(columns=drop_cols, errors='ignore')
    return df

# ----------------------------
# Build pipeline / training
# ----------------------------
def build_and_train(X_train, y_train, X_val, y_val, random_state=42, n_iter_search=40):
    """
    Build preprocessing + SMOTE + classifier pipeline and perform randomized search.
    Returns best_estimator_, cv_results_
    """
    # Identify column types
    categorical_cols = [c for c in X_train.columns if X_train[c].dtype == 'object' or c == 'type']
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    # Numeric preprocessing
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    # Categorical preprocessing
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(handle_unknown='ignore', sparse=False)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop',
        sparse_threshold=0
    )

    # Classifier (random forest baseline)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state, class_weight='balanced')

    # Build an imbalanced-learn Pipeline: preprocessing -> SMOTE -> classifier
    pipeline = ImbPipeline([
        ('preproc', preprocessor),
        ('smote', SMOTE(random_state=random_state, n_jobs=-1)),
        ('clf', clf)
    ])

    # Hyperparameter grid for RandomizedSearch (coarse)
    param_distributions = {
        'clf__n_estimators': [100, 200, 400],
        'clf__max_depth': [6, 10, 20, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    }

    # Use StratifiedKFold for CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    rs = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=min(n_iter_search, 30),
        scoring='f1',   # focus on balanced detection
        n_jobs=-1,
        cv=cv,
        verbose=2,
        random_state=random_state,
        refit=True
    )

    print("Starting RandomizedSearchCV (this can take some time)...")
    rs.fit(X_train, y_train)

    print("Best params:", rs.best_params_)
    print("Best CV score (f1):", rs.best_score_)

    # Evaluate on validation
    val_preds = rs.predict(X_val)
    val_proba = rs.predict_proba(X_val)[:, 1] if hasattr(rs, 'predict_proba') else None

    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_preds, digits=4))
    if val_proba is not None:
        print("ROC-AUC:", roc_auc_score(y_val, val_proba))
        print("PR-AUC (average precision):", average_precision_score(y_val, val_proba))

    return rs

# ----------------------------
# Evaluation plotting
# ----------------------------
def plot_pr_curve(y_true, y_score, outpath=None):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.grid(True)
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
    else:
        plt.show()

# ----------------------------
# Main runner
# ----------------------------
def main(args):
    # 1) load
    df = pd.read_csv(args.data)
    print("Loaded data:", df.shape)
    print("Columns:", df.columns.tolist())

    # 2) quick EDA printouts (small)
    print("\nValue counts for 'type':")
    if 'type' in df.columns:
        print(df['type'].value_counts())

    print("\nisFraud distribution:")
    print(df['isFraud'].value_counts(normalize=True))

    # 3) add features
    df2 = add_features(df)
    print("\nAfter feature engineering, columns:", df2.columns.tolist())

    # 4) split features/target
    if 'isFraud' not in df2.columns:
        raise ValueError("Target column 'isFraud' not found in dataset.")
    X = df2.drop(columns=['isFraud'])
    y = df2['isFraud'].astype(int)

    # If dataset is large and step/time exists, optionally do a time-based split.
    # We'll do a stratified random split for generality:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    # further split temp into val/test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"Train/Val/Test sizes: {X_train.shape[0]}, {X_val.shape[0]}, {X_test.shape[0]}")

    # 5) build and train
    rs = build_and_train(X_train, y_train, X_val, y_val,
                         random_state=args.random_state,
                         n_iter_search=args.n_iter_search)

    # 6) Final evaluate on holdout test set
    print("\n--- Final evaluation on TEST set ---")
    test_preds = rs.predict(X_test)
    test_proba = rs.predict_proba(X_test)[:, 1] if hasattr(rs, 'predict_proba') else None

    print(classification_report(y_test, test_preds, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_preds))
    if test_proba is not None:
        print("ROC-AUC (test):", roc_auc_score(y_test, test_proba))
        print("PR-AUC (test):", average_precision_score(y_test, test_proba))
        plot_pr_curve(y_test, test_proba, outpath=args.pr_curve if args.pr_curve else None)

    # 7) Save the whole pipeline (best estimator) + metadata
    output_path = args.out if args.out else 'fraud_pipeline.joblib'
    joblib.dump({
        'model': rs.best_estimator_,
        'best_params': rs.best_params_,
        'cv_results': rs.cv_results_
    }, output_path)
    print(f"Saved best pipeline to {output_path}")

    # Optional: show feature importance if classifier supports (RandomForest)
    try:
        # best_estimator_ is a pipeline: ['preproc', 'smote', 'clf']
        model_pipeline = rs.best_estimator_
        clf = model_pipeline.named_steps['clf']
        preproc = model_pipeline.named_steps['preproc']
        if hasattr(clf, 'feature_importances_'):
            # Build feature names after preprocessing
            # numeric columns
            cat_cols = []
            num_cols = []
            for name, trans, cols in preproc.transformers_:
                if name == 'num':
                    num_cols = cols
                elif name == 'cat':
                    # get categories from OneHotEncoder
                    cat_pipeline = trans
                    # extract underlying OneHotEncoder
                    ohe = None
                    for step in cat_pipeline.steps:
                        if step[0] == 'onehotencoder' or isinstance(step[1], OneHotEncoder):
                            ohe = step[1]
                    if ohe is None:
                        # try to find OneHotEncoder by type
                        import sklearn
                        from sklearn.preprocessing import OneHotEncoder as _OHE
                        for step in cat_pipeline.steps:
                            if isinstance(step[1], _OHE):
                                ohe = step[1]
                    if ohe is not None:
                        ohe_cols = []
                        # transformer's columns are in `cols` variable sometimes not accessible - fallback to names
                        # Build names: feature__category
                        try:
                            categories = ohe.categories_
                            # but we need the original categorical column names - attempt to get from preprocessor
                            cat_feature_names = []
                            # preproc.transformers_ provides the (name, transformer, columns)
                            for nm, tr, cols_in in preproc.transformers_:
                                if nm == 'cat':
                                    cat_feature_names = cols_in
                            for feat, cats in zip(cat_feature_names, categories):
                                for c in cats:
                                    ohe_cols.append(f"{feat}__{c}")
                        except Exception:
                            ohe_cols = ['cat_' + str(i) for i in range(sum(len(cats) for cats in getattr(ohe, 'categories_', [])))]
                        cat_cols = ohe_cols
            feature_names = list(num_cols) + list(cat_cols)
            importances = clf.feature_importances_
            # If length mismatch, skip printing names
            if len(importances) == len(feature_names):
                imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                imp_df = imp_df.sort_values('importance', ascending=False).head(25)
                print("\nTop feature importances:")
                print(imp_df.to_string(index=False))
            else:
                print("\nFeature importance vector length != feature name count - skipping pretty print.")
        else:
            print("Classifier does not expose feature_importances_")
    except Exception as e:
        print("Could not compute feature importances (non-fatal). Error:", e)

# ----------------------------
# Optional minimal Flask demo
# ----------------------------
FLASK_TEMPLATE = r"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
art = joblib.load("{model_path}")['model']

@app.route("/predict", methods=["POST"])
def predict():
    # expecting JSON like: {"step": 1, "type": "PAYMENT", "amount": 123.45, ...}
    data = request.get_json()
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    # apply same feature engineering function used in training
    from types import SimpleNamespace
    # naive re-implementation of add_features (small)
    df['deltaOrig'] = df.get('oldbalanceOrg', 0) - df.get('newbalanceOrig', 0)
    df['deltaDest'] = df.get('newbalanceDest', 0) - df.get('oldbalanceDest', 0)
    df['amt_to_oldbal_ratio'] = df['amount'] / (df.get('oldbalanceOrg', 0) + 1.0)
    df['is_same_account'] = (df.get('nameOrig') == df.get('nameDest')).astype(int)
    df['hour'] = df.get('step', 0) % 24
    # drop id columns
    for c in ['nameOrig','nameDest','step']:
        if c in df.columns:
            df = df.drop(columns=[c])
    preds = art.predict(df)
    probs = art.predict_proba(df)[:,1] if hasattr(art, 'predict_proba') else None
    out = {"predictions": preds.tolist()}
    if probs is not None:
        out["probability"] = probs.tolist()
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
"""

def write_flask_demo(path_model, out_file='flask_demo.py'):
    s = FLASK_TEMPLATE.format(model_path=path_model)
    with open(out_file, 'w') as f:
        f.write(s)
    print(f"Wrote Flask demo to {out_file}. Run with: python {out_file}")

# ----------------------------
# Command-line
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud detection pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--out", type=str, default="fraud_pipeline.joblib", help="Path to save the trained pipeline")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--n_iter_search", type=int, default=20, help="RandomizedSearch iterations")
    parser.add_argument("--pr_curve", type=str, default=None, help="Output path to save PR curve image (PNG)")
    parser.add_argument("--write_flask", action="store_true", help="Write a small Flask demo file after training")
    args = parser.parse_args()
    main(args)
    if args.write_flask:
        write_flask_demo(args.out, out_file='flask_demo.py')
