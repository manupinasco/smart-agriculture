import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score,make_scorer, precision_recall_fscore_support

INPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features_data.parquet"
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "src" / "artifacts"

def custom_f1_class_1(y_true, y_pred):
    """ Custom scorer strictly focused on class 1 """
    return f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)

def custom_recall_class_1(y_true, y_pred):
    """ Custom scorer for recall strictly focused on class 1 """
    return recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)

def train_decision_tree(X, y, cv):
    print("\n" + "=" * 80)
    print("Decision Tree (temp and wind)")
    print("=" * 80)
    
    param_grid = {
        "max_depth": [3, 5, 7, None],
        "min_samples_leaf": [1, 5, 10, 20]
    }

    scoring_dict = {
        'f1': make_scorer(custom_f1_class_1),
        'recall': make_scorer(custom_recall_class_1)
    }
    
    grid = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring_dict,
        refit="f1"
    )
    
    grid.fit(X, y)
    
    cv_res = grid.cv_results_
    best_idx = grid.best_index_
    print(f"Best parameters: {grid.best_params_}")
    print(f"Validation F1 Class 1:     {cv_res['mean_test_f1'][best_idx]:.3f} +/- {cv_res['std_test_f1'][best_idx]:.3f}")
    print(f"Validation Recall Class 1: {cv_res['mean_test_recall'][best_idx]:.3f} +/- {cv_res['std_test_recall'][best_idx]:.3f}")

    joblib.dump(grid.best_estimator_, ARTIFACTS_DIR / "model_dt.pkl")
    return grid.best_estimator_

def train_random_forest(X, y, cv):
    print("\n" + "=" * 80)
    print("Random Forest (all features)")
    print("=" * 80)
    
    param_grid = {
        "n_estimators": [100, 500],
        "max_depth": [5, 20],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt"]
    }

    scoring_dict = {
        'f1': make_scorer(custom_f1_class_1),
        'recall': make_scorer(custom_recall_class_1)
    }
    
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring_dict,
        refit='f1'
    )
    
    grid.fit(X, y)
    
    cv_res = grid.cv_results_
    best_idx = grid.best_index_
    print(f"Best parameters: {grid.best_params_}")
    print(f"Validation F1 Class 1:     {cv_res['mean_test_f1'][best_idx]:.3f} +/- {cv_res['std_test_f1'][best_idx]:.3f}")
    print(f"Validation Recall Class 1: {cv_res['mean_test_recall'][best_idx]:.3f} +/- {cv_res['std_test_recall'][best_idx]:.3f}")
    
    joblib.dump(grid.best_estimator_, ARTIFACTS_DIR / "model_rf.pkl")
    return grid.best_estimator_

def train_random_forest_pca(X_raw, y, cv):
    print("\n" + "=" * 80)
    print("Random Forest + PCA")

    print("=" * 80)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pd.DataFrame(pca.fit_transform(X_scaled), index=X_scaled.index, columns=["PC1", "PC2"])
    
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
    joblib.dump(pca, ARTIFACTS_DIR / "pca.pkl")
    
    rf = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=5, random_state=42)
    
    metrics_folds = []
    for train_idx, val_idx in cv.split(X_pca, y):
        model = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=5, random_state=42)
        model.fit(X_pca.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X_pca.iloc[val_idx])
        
        precision, recall, f1, _ = precision_recall_fscore_support(y.iloc[val_idx], y_pred, labels=[0, 1], zero_division=0)
        
        metrics_folds.append({
            "f1_class_1": f1[1],
            "recall_class_1": recall[1]
        })
        
    metrics_df = pd.DataFrame(metrics_folds)
    print(f"Validation F1 Class 1:     {metrics_df['f1_class_1'].mean():.3f} +/- {metrics_df['f1_class_1'].std():.3f}")
    print(f"Validation Recall Class 1: {metrics_df['recall_class_1'].mean():.3f} +/- {metrics_df['recall_class_1'].std():.3f}")

    rf.fit(X_pca, y)
    joblib.dump(rf, ARTIFACTS_DIR / "model_rf_pca.pkl")
    return rf

def main():
    df = pd.read_parquet(INPUT_PATH)
    df = df.sort_values(["Campaña_ID", "FECHA_INICIO"]).reset_index(drop=True)
    df["mes"] = df["FECHA_INICIO"].dt.month
    
    test_mask = df["Campaña_ID"].str.endswith("Camp_5")
    df_trainval = df.loc[~test_mask].copy().reset_index(drop=True)
    
    df_trainval = pd.get_dummies(df_trainval, columns=["estado_fenologico_dominante"], dtype=int)
    feno_cols = [c for c in df_trainval.columns if c.startswith("estado_fenologico_dominante_")]
    joblib.dump(df_trainval.columns.tolist(), ARTIFACTS_DIR / "train_columns.pkl")
    
    y_train = df_trainval["Estado_Malo"]
    gkf = TimeSeriesSplit(n_splits=4)
    
    features_dt = ["temp_acum_1m", "viento_prom_1m"]
    train_decision_tree(df_trainval[features_dt], y_train, gkf)
    
    features_rf = ["temp_acum_1m", "viento_prom_1m", "hum_prom_1m", "pres_prom_1m"] + feno_cols
    train_random_forest(df_trainval[features_rf], y_train, gkf)
    
    train_random_forest_pca(df_trainval[features_rf], y_train, gkf)

if __name__ == "__main__":
    main()