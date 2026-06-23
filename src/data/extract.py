import re
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from tabula import read_pdf

# Internal imports
from src.utils.sagyp_downloader import download_all_pdfs, DOWNLOAD_DIR
from src.utils.weather_functions import fetch_weather_data, compute_7d_rolling_features

FINAL_DATASET_DOWNLOAD_DIR=Path(__file__).resolve().parents[2] / "data" / "raw"

MAPEO_PROVINCIAS = {
    'BAHÍA BLANCA': 'BUENOS AIRES', 'BOLÍVAR': 'BUENOS AIRES', 'BRAGADO': 'BUENOS AIRES',
    'GENERAL MADARIAGA': 'BUENOS AIRES', 'JUNÍN': 'BUENOS AIRES', 'LA PLATA': 'BUENOS AIRES',
    'LINCOLN': 'BUENOS AIRES', 'PEHUAJÓ': 'BUENOS AIRES', 'PERGAMINO': 'BUENOS AIRES',
    'PIGÜÉ': 'BUENOS AIRES', 'SALLIQUELÓ': 'BUENOS AIRES', 'TANDIL': 'BUENOS AIRES',
    'TRES ARROYOS': 'BUENOS AIRES', '25 DE MAYO': 'BUENOS AIRES', 'LABOULAYE': 'CÓRDOBA',
    'MARCOS JUÁREZ': 'CÓRDOBA', 'RÍO CUARTO': 'CÓRDOBA', 'SAN FRANCISCO': 'CÓRDOBA',
    'VILLA MARÍA': 'CÓRDOBA', 'PARANÁ': 'ENTRE RÍOS', 'ROSARIO DEL TALA': 'ENTRE RÍOS',
    'GENERAL PICO': 'LA PAMPA', 'SANTA ROSA': 'LA PAMPA', 'AVELLANEDA': 'SANTA FE',
    'CAÑADA de GÓMEZ': 'SANTA FE', 'CASILDA': 'SANTA FE', 'RAFAELA': 'SANTA FE',
    'VENADO TUERTO': 'SANTA FE', 'CATAMARCA': 'CATAMARCA', 'CORRIENTES': 'CORRIENTES',
    'CHARATA': 'CHACO', 'ROQUE SÁENZ PEÑA': 'CHACO', 'SALTA': 'SALTA', 'SAN LUIS': 'SAN LUIS',
    'SANTIAGO DEL ESTERO': 'SANTIAGO DEL ESTERO', 'QUIMILÍ': 'SANTIAGO DEL ESTERO', 'TUCUMÁN': 'TUCUMÁN'
}

def extract_tables_with_tabula(pdf_path):
    """ Tries to extract raw tables from a single report """
    print(f"Processing PDF: {pdf_path}")
    df_list = read_pdf(pdf_path, pages='all', multiple_tables=True, stream=True)
    if df_list:
        print(f"Success: Found {len(df_list)} tables.")
    else:
        print("Warning: No tables found in this file.")
    return df_list

def process_raw_agricultural_table(df_untidy):
    """ Cleans messy Tabula structures into a neat tidy DataFrame """
    # 1. Spot the crop name (usually hidden in first headers)
    crop = "UNKNOWN"
    for col in df_untidy.columns:
        if "Unnamed" not in str(col):
            crop = str(col).upper()
            break

    # 2. Skip first 3 useless rows
    df = df_untidy.iloc[3:].copy().reset_index(drop=True)

    # 3. Parse line by line to extract info
    clean_rows = []
    for _, row in df.iterrows():
        raw_line = " ".join([str(x) for x in row])
        parts = raw_line.split()

        # Find which region fits this row text
        delegacion = next((d for d in MAPEO_PROVINCIAS.keys() if d in raw_line.upper()), None)

        if delegacion is not None:
            # Filter out region name and NaNs to keep numeric features
            values = [x for x in parts if x not in delegacion.split(" ") and x != 'nan']
            
            if len(values) == 14:
                # Last 4 items contain date traces like 21_05_2024
                date_path = "_".join(values[-4:])
                date_match = re.search(r'(\d{2})_(\d{2})_(\d{4})', date_path)
                
                if date_match:
                    end_date = datetime.strptime(f'{date_match.group(1)}/{date_match.group(2)}/{date_match.group(3)}', "%d/%m/%Y")
                    start_date = end_date - timedelta(days=7)
                    
                    numeric_features = values[:-4]
                    new_row = [delegacion, start_date, end_date] + numeric_features
                    clean_rows.append(new_row)

    if not clean_rows:
        return None

    df_final = pd.DataFrame(clean_rows)
    df_final.columns = [
        'DELEGACIÓN', 'FECHA_INICIO', 'FECHA_FIN',
        'Estadio Fenológico E', 'Estadio Fenológico C', 'Estadio Fenológico F',
        'Estadio Fenológico L', 'Estadio Fenológico M',
        'Estado General MB', 'Estado General B', 'Estado General R', 'Estado General M',
        'Humedad'
    ]
    df_final['PROVINCIA'] = df_final['DELEGACIÓN'].map(MAPEO_PROVINCIAS)
    df_final['CULTIVO'] = crop.replace(" (PAN)", "")
    return df_final

def main():
    # --- Step 1: Download reports ---
    download_all_pdfs()

    # --- Step 2: Parse PDFs with Tabula ---
    pdf_files = [f for f in Path(DOWNLOAD_DIR).iterdir() if f.is_file()]
    all_extracted_tables = []
    
    for path in pdf_files:
        tables = extract_tables_with_tabula(path)
        if tables:
            for df in tables:
                df["PDF_PATH"] = str(path)
            all_extracted_tables.extend(tables)

    # --- Step 3: Tidy and Filter ---
    processed_dfs = []
    for raw_df in all_extracted_tables:
        # Check if it's the specific table type we want by looking for 'Humedad'
        if any("Humedad" in str(cell) for cell in raw_df.values.flatten()):
            tidy_df = process_raw_agricultural_table(raw_df)
            if tidy_df is not None:
                processed_dfs.append(tidy_df)

    if not processed_dfs:
        print("Error: No tables were successfully processed.")
        exit()

    df_agri_final = pd.concat(processed_dfs, ignore_index=True)

    # --- Step 4: Weather Enrichment ---
    min_date = min(df_agri_final["FECHA_INICIO"])
    max_date = max(df_agri_final["FECHA_FIN"])
    
    raw_weather = fetch_weather_data(min_date, max_date)
    df_weather_features = compute_7d_rolling_features(raw_weather)

    # --- Step 5: Master Merge ---
    weather_features_list = [
        'time', 'NOMBRE', 'Total_7d_temp', 'Total_7d_rhum', 'Total_7d_prcp', 
        'Total_7d_wspd', 'Total_7d_pres', 'Promedio_7d_temp', 'Promedio_7d_rhum', 
        'Promedio_7d_prcp', 'Promedio_7d_wspd', 'Promedio_7d_pres', 'Min_7d_temp', 
        'Min_7d_rhum', 'Min_7d_prcp', 'Min_7d_wspd', 'Min_7d_pres', 'Max_7d_temp', 
        'Max_7d_rhum', 'Max_7d_prcp', 'Max_7d_wspd', 'Max_7d_pres'
    ]

    final_dataset = df_agri_final.merge(
        df_weather_features[weather_features_list],
        left_on=['FECHA_INICIO', 'DELEGACIÓN'],
        right_on=['time', 'NOMBRE'],
        how='left'
    )

    output_path = Path(FINAL_DATASET_DOWNLOAD_DIR) / "agricultural_dataset.parquet"
    final_dataset.to_parquet(output_path, index=False)

    print(f"Success. Final dataset saved to {output_path}. Sample:")
    print(final_dataset.head())

if __name__ == "__main__":
    main()