import pandas as pd
import joblib
from pathlib import Path

# IO Paths
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "features_data.parquet"
ARTIFACTS_DIR = BASE_DIR / "src" / "artifacts"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "predictions.parquet"

def predict_benchmark(df):
    """
    Persistence benchmark: Predicts that the current state will be the same 
    as the previous state for each campaign.
    """
    print("   -> Running Benchmark (Persistence)...")
    # Shift by 1 within each campaign. Fill first values with 0 (Good state)
    predictions = df.groupby("Campaña_ID")["Estado_Malo"].shift(1).fillna(0).astype(int)
    return predictions

def predict_decision_tree(model, X):
    """ Generates predictions using the Decision Tree model. """
    print("   -> Running Decision Tree...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities

def predict_random_forest(model, X):
    """ Generates predictions using the Random Forest model. """
    print("   -> Running Random Forest...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities

def predict_random_forest_pca(model, scaler, pca, X):
    """ Generates predictions using the Random Forest model with PCA. """
    print("   -> Running Random Forest + PCA...")
    
    # Scale and transform the features using the saved artifacts
    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    X_pca = pd.DataFrame(pca.transform(X_scaled), index=X_scaled.index, columns=["PC1", "PC2"])
    
    predictions = model.predict(X_pca)
    probabilities = model.predict_proba(X_pca)[:, 1]
    return predictions, probabilities

def main():

    # 1. Load the data
    df = pd.read_parquet(INPUT_PATH)
    
    # Taking a sample (last 50 rows) to simulate new incoming data
    df_new = df.tail(50).copy().reset_index(drop=True)

    # 2. Load models
    try:
        train_columns = joblib.load(ARTIFACTS_DIR / "train_columns.pkl")
        model_dt      = joblib.load(ARTIFACTS_DIR / "model_dt.pkl")
        model_rf      = joblib.load(ARTIFACTS_DIR / "model_rf.pkl")
        model_rf_pca  = joblib.load(ARTIFACTS_DIR / "model_rf_pca.pkl")
        scaler        = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
        pca           = joblib.load(ARTIFACTS_DIR / "pca.pkl")
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        print("Run the training script first")
        return None

    # 3. Preprocess the incoming data
    df_processed = pd.get_dummies(df_new, columns=["estado_fenologico_dominante"], dtype=int)
    df_processed = df_processed.reindex(columns=train_columns, fill_value=0)

    # Define feature subsets
    features_dt = ["temp_acum_1m", "viento_prom_1m"]
    feno_cols = [c for c in train_columns if c.startswith("estado_fenologico_dominante_")]
    features_rf = ["temp_acum_1m", "viento_prom_1m", "hum_prom_1m", "pres_prom_1m"] + feno_cols

    X_dt = df_processed[features_dt]
    X_rf = df_processed[features_rf]

    # 4. Generating predictions for all models
    
    # Benchmark
    df_new["Pred_Benchmark"] = predict_benchmark(df_new)
    
    # Decision Tree
    preds_dt, probs_dt = predict_decision_tree(model_dt, X_dt)
    df_new["Pred_DT"] = preds_dt
    df_new["Prob_Class_1_DT"] = probs_dt
    
    # Random Forest
    preds_rf, probs_rf = predict_random_forest(model_rf, X_rf)
    df_new["Pred_RF"] = preds_rf
    df_new["Prob_Class_1_RF"] = probs_rf
    
    # Random Forest + PCA
    preds_pca, probs_pca = predict_random_forest_pca(model_rf_pca, scaler, pca, X_rf)
    df_new["Pred_RF_PCA"] = preds_pca
    df_new["Prob_Class_1_RF_PCA"] = probs_pca

    # 5. Save the output
    print(f"\nSaving results to: {OUTPUT_PATH.name}")
    
    # Make the output predictions readable
    output_cols = [
        "Campaña_ID", "FECHA_INICIO", "Estado_Malo",
        "Pred_Benchmark", 
        "Pred_DT", "Prob_Class_1_DT",
        "Pred_RF", "Prob_Class_1_RF",
        "Pred_RF_PCA", "Prob_Class_1_RF_PCA"
    ]
    
    final_cols = [col for col in output_cols if col in df_new.columns]
    df_new[final_cols].to_parquet(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()