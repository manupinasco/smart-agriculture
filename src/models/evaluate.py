import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, roc_curve, auc

INPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features_data.parquet"
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "src" / "artifacts"
PLOTS_DIR = Path(__file__).resolve().parents[2] / "src" / "plots" / "models"

PALETTE = {1: "red", 0: "darkgreen"}

def evaluate_metrics(name, y_true, y_pred):
    """Calculates and reports F1 and Recall for class 1."""
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    print(f"{name:<25} | F1 Class 1: {f1:.3f} | Recall Class 1: {recall:.3f}")

def plot_decision_boundary(model, X, y, var_names, title, filename):
    """Plots decision boundary in 2D."""
    var1, var2 = var_names[0], var_names[1]
    
    x_min, x_max = X[var1].min() - 0.5, X[var1].max() + 0.5
    y_min, y_max = X[var2].min() - 0.5, X[var2].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=[var1, var2])
    Z = model.predict(grid_df)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    plt.scatter(X[var1], X[var2], c=[PALETTE[i] for i in y], edgecolor="k", s=50, alpha=0.8)
    
    plt.xlabel(f"{var1}")
    plt.ylabel(f"{var2}")
    plt.title(title, fontsize=12)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Malo',
               markerfacecolor='red', markeredgecolor='k', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Bueno',
               markerfacecolor='darkgreen', markeredgecolor='k', markersize=8)
    ]
    plt.legend(handles=legend_elements, title="Crop state")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.show()

def plot_roc_curves(y_true, probas_dict, filename):
    """Plots ROC-AUC curves for multiple models."""
    plt.figure(figsize=(8, 6))
    
    for name, y_prob in probas_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.show()

def plot_feature_importance(model, feature_names, filename):
    """Plots variable importance for Random Forest."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_importances)), sorted_importances[::-1], align='center', color='skyblue', edgecolor='k')
    plt.yticks(range(len(sorted_importances)), sorted_features[::-1])
    plt.xlabel("Importance")
    plt.title("Variable importance - Random Forest")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.show()

def main():
    # 1. Loads data and splits train/test (last campaign is used for testing)
    df = pd.read_parquet(INPUT_PATH)
    df = df.sort_values(["Campaña_ID", "FECHA_INICIO"]).reset_index(drop=True)
    
    test_mask = df["Campaña_ID"].str.endswith("Camp_5")
    df_test = df.loc[test_mask].copy().reset_index(drop=True)
    
    # 2. Benchmark (Persistance)
   
    df_test["pred_persistencia"] = df_test.groupby("Campaña_ID")["Estado_Malo"].shift(1).fillna(0)
    
    # 3. Preprocessing
    df_test = pd.get_dummies(df_test, columns=["estado_fenologico_dominante"], dtype=int)
    train_columns = joblib.load(ARTIFACTS_DIR / "train_columns.pkl")
    
    y_test = df_test["Estado_Malo"]
    
    # 4. Loads models
    model_dt = joblib.load(ARTIFACTS_DIR / "model_dt.pkl")
    model_rf = joblib.load(ARTIFACTS_DIR / "model_rf.pkl")
    model_rf_pca = joblib.load(ARTIFACTS_DIR / "model_rf_pca.pkl")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    pca = joblib.load(ARTIFACTS_DIR / "pca.pkl")
    
    features_dt = ["temp_acum_1m", "viento_prom_1m"]
    feno_cols = [c for c in train_columns if c.startswith("estado_fenologico_dominante_")]
    features_rf = ["temp_acum_1m", "viento_prom_1m", "hum_prom_1m", "pres_prom_1m"] + feno_cols
    
    X_test_dt = df_test[features_dt]
    X_test_rf = df_test[features_rf]
    
    # Transforming PCA
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_rf), index=X_test_rf.index, columns=X_test_rf.columns)
    X_test_pca = pd.DataFrame(pca.transform(X_test_scaled), index=X_test_scaled.index, columns=["PC1", "PC2"])
    
    # 5. Reports metrics
    print("\n" + "=" * 55)
    print("TESTING IN LAST CAMPAIGN (Camp_5)")
    print("=" * 55)
    
    evaluate_metrics("Benchmark (Persistencia)", y_test, df_test["pred_persistencia"])
    evaluate_metrics("Decision Tree", y_test, model_dt.predict(X_test_dt))
    evaluate_metrics("Random Forest", y_test, model_rf.predict(X_test_rf))
    evaluate_metrics("Random Forest + PCA", y_test, model_rf_pca.predict(X_test_pca))
    print("=" * 55 + "\n")
    
    # Processing full dataframe for decision boundary plots
    df_full = pd.get_dummies(df, columns=["estado_fenologico_dominante"], dtype=int)
    df_full = df_full.reindex(columns=train_columns, fill_value=0)
    y_full = df_full["Estado_Malo"]
    
    X_full_dt = df_full[features_dt]
    X_full_rf = df_full[features_rf]
    X_full_scaled = pd.DataFrame(scaler.transform(X_full_rf), index=X_full_rf.index, columns=X_full_rf.columns)
    X_full_pca = pd.DataFrame(pca.transform(X_full_scaled), index=X_full_scaled.index, columns=["PC1", "PC2"])
    
    # 7. Generating plots
    print("Generating plots in dir 'plots/':")

    # Decision boundaries
    plot_decision_boundary(model_dt, X_full_dt, y_full, features_dt, 
                           "Frontera de Decisión - Árbol de Decisión", "boundary_dt.png")
    print("- boundary_dt.png generado.")
    
    plot_decision_boundary(model_rf_pca, X_full_pca, y_full, ["PC1", "PC2"], 
                           "Frontera de Decisión - Random Forest (PCA)", "boundary_pca.png")
    print("- boundary_pca.png generado.")
    
    # ROC Curves
    probas = {
        "Random Forest": model_rf.predict_proba(X_test_rf)[:, 1],
        "Random Forest + PCA": model_rf_pca.predict_proba(X_test_pca)[:, 1]
    }
    plot_roc_curves(y_test, probas, "roc_curves.png")
    print("- roc_curves.png generado.")
    
    # Variable importance
    plot_feature_importance(model_rf, features_rf, "feature_importance_rf.png")
    print("- feature_importance_rf.png generado.\n")

if __name__ == "__main__":
    main()