import os
import pandas as pd
from pathlib import Path

# IO Paths
DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
ORIGINAL_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
INPUT_PATH = os.path.join(ORIGINAL_DIR, "agricultural_dataset.parquet")
OUTPUT_PATH = os.path.join(DIR, "transformed_data.parquet")

DICT_MAPEAR = {
    "RAFAELA": "SAN FRANCISCO", "VENADO TUERTO": "PERGAMINO", "VILLA MARÍA": "SAN FRANCISCO",
    "PERGAMINO": "JUNÍN", "BRAGADO": "JUNÍN", "CHARATA": "ROQUE SÁENZ PEÑA",
    "25 DE MAYO": "BOLÍVAR", "LINCOLN": "JUNÍN", "SALLIQUELÓ": "PEHUAJÓ", "ROSARIO DEL TALA": "PARANÁ"
}

VARS_CLIMATICAS = ['temp_acum_1m', 'pres_prom_1m', 'viento_prom_1m', 'hum_prom_1m']
ESTADOS_COLS = ['Estado General MB', 'Estado General B', 'Estado General R', 'Estado General M']

def run_transformation(df_input=None):
    if df_input is not None:
        df=df_input.copy()
    else:
        os.makedirs(DIR, exist_ok=True)
        
        if not os.path.exists(INPUT_PATH):
            raise FileNotFoundError(f"Missing input base data at {INPUT_PATH}")
            
        df = pd.read_parquet(INPUT_PATH)

    df['FECHA_INICIO'] = pd.to_datetime(df['FECHA_INICIO'])
    df['FECHA_FIN'] = pd.to_datetime(df['FECHA_FIN'])

    # String cleaning
    df["CULTIVO"] = df["CULTIVO"].str.replace(" (PAN)", "", regex=False)

    # --- 1. Campaign Generation Logic ---
    df = df.sort_values(by=['CULTIVO', 'DELEGACIÓN', 'FECHA_INICIO']).reset_index(drop=True)
    
    df['Dias_Desde_Anterior'] = df.groupby(['CULTIVO', 'DELEGACIÓN'])['FECHA_INICIO'].diff().dt.days
    df['Nueva_Campaña'] = (df['Dias_Desde_Anterior'] > 30) | df['Dias_Desde_Anterior'].isna()
    
    campaign_num = df.groupby(['CULTIVO', 'DELEGACIÓN'])['Nueva_Campaña'].cumsum().astype(str)
    df['Campaña_ID'] = df['CULTIVO'] + "_" + df['DELEGACIÓN'] + "_Camp_" + campaign_num

    # --- 2. 1-Month Rolling Weather Features ---
    # 4 periods = 4 weeks (approx 1 month)
    df['temp_acum_1m'] = df.groupby(["DELEGACIÓN", "CULTIVO"])['Total_7d_temp'].transform(lambda s: s.rolling(4, min_periods=1).sum())
    df['viento_prom_1m'] = df.groupby(["DELEGACIÓN", "CULTIVO"])['Promedio_7d_wspd'].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df['hum_prom_1m'] = df.groupby(["DELEGACIÓN", "CULTIVO"])['Promedio_7d_rhum'].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df['pres_prom_1m'] = df.groupby(["DELEGACIÓN", "CULTIVO"])['Promedio_7d_pres'].transform(lambda s: s.rolling(4, min_periods=1).mean())
    
    # --- 3. Fixing Missing Weather using Nearby Regions Mapping ---
    aux = df[['DELEGACIÓN', 'FECHA_INICIO'] + VARS_CLIMATICAS].copy()
    aux = aux.rename(columns={'DELEGACIÓN': 'DELEGACION_ORIGEN', **{c: f'{c}_origen' for c in VARS_CLIMATICAS}})

    df['DELEGACION_ORIGEN'] = df['DELEGACIÓN'].map(DICT_MAPEAR)
    df = df.merge(aux, on=['DELEGACION_ORIGEN', 'FECHA_INICIO'], how='left')

    # Replace specific empty/bad weather cells with values from fallback regions
    for c in VARS_CLIMATICAS:
        mask = df['DELEGACIÓN'].isin(DICT_MAPEAR.keys())
        df.loc[mask, c] = df.loc[mask, f'{c}_origen']

    df = df.drop(columns=['DELEGACION_ORIGEN'] + [f'{c}_origen' for c in VARS_CLIMATICAS])

    df[ESTADOS_COLS] = df[ESTADOS_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Success. Cleaned data saved to {OUTPUT_PATH}. Sample:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    run_transformation()