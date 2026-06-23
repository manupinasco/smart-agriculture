import os
import pandas as pd
import numpy as np
from pathlib import Path

# IO Paths
DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
INPUT_PATH = os.path.join(DIR, "transformed_data.parquet")
OUTPUT_PATH = os.path.join(DIR, "features_data.parquet")

ESTADOS_COLS = ['Estado General MB', 'Estado General B', 'Estado General R', 'Estado General M']
ESTADOS_FENOLOGICOS = ['Estadio Fenológico E', 'Estadio Fenológico C', 'Estadio Fenológico F', 'Estadio Fenológico L', 'Estadio Fenológico M']

def run_feature_engineering():
    print("Running feature engineering")
    
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing transformed data checkpoint at {INPUT_PATH}")
        
    df = pd.read_parquet(INPUT_PATH)

    # --- 1. Target Setting ---
    # Drop rows that don't make sense before filtering features
    df.dropna(inplace=True)
    df["Camp_ID"] = df["Campaña_ID"].str.extract(r'(Camp_\d+)')
    df = df[df["Estado General B"] < 100]
    df = df[df[ESTADOS_COLS].sum(axis=1) > 95]

    # Calculate current score based on weight vectors
    df['Score'] = (-2 * df['Estado General M']) + (-1 * df['Estado General R']) + \
                  (1 * df['Estado General B']) + (2 * df['Estado General MB'])

    # Shift -4 to look exactly 4 weeks ahead into the future inside the same campaign
    df['Estado_Cultivo'] = df.groupby('Campaña_ID')['Score'].shift(-4)

    # Binary classification target
    df['Estado_Malo'] = np.select([df['Estado_Cultivo'] > 0], [0], default=1).astype(float)

    # --- 2. Phenological Features ---
    df[ESTADOS_FENOLOGICOS] = df[ESTADOS_FENOLOGICOS].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["estado_fenologico_dominante"] = df[ESTADOS_FENOLOGICOS].idxmax(axis=1)

    # --- 3. Scope Filtering ---
    
    # Filter by Wheat and target specific stable campaigns
    df_features = df[df["CULTIVO"]=="TRIGO"].copy()

    # Final selection of relevant columns
    final_cols = [
        'DELEGACIÓN', 'FECHA_INICIO', 'FECHA_FIN', 'PROVINCIA', 'Campaña_ID', 'Camp_ID', 'CULTIVO',
        'Estadio Fenológico E', 'Estadio Fenológico C', 'Estadio Fenológico F', 'Estadio Fenológico L', 'Estadio Fenológico M',
        'estado_fenologico_dominante', 'Estado General MB', 'Estado General B', 'Estado General R', 'Estado General M', 'temp_acum_1m', 'viento_prom_1m', 'hum_prom_1m', 'pres_prom_1m',
        'Estado_Malo'
    ]
    
    available_cols = [c for c in final_cols if c in df_features.columns]
    df_features = df_features[available_cols]

    # Save final dataframe
    df_features.to_parquet(OUTPUT_PATH, index=False)
    print(f"Success. Shape: {df_features.shape}")
    print(f"File stored at {OUTPUT_PATH}")

if __name__ == "__main__":
    run_feature_engineering()