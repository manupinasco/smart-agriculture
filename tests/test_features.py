import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# Assuming the function is imported from your src path
from src.features.build_features import run_feature_engineering

captured_data = {}
def mock_to_parquet_func(self, path, **kwargs):
    captured_data['df'] = self

@patch('src.features.build_features.pd.DataFrame.to_parquet', new=mock_to_parquet_func)
@patch('src.features.build_features.pd.read_parquet')
@patch('src.features.build_features.os.path.exists', return_value=True)
def test_feature_engineering_pipeline(mock_exists, mock_read_parquet):
    """
    Tests the complete feature engineering pipeline
    """
    # 1. Create mock data
    mock_data = {
        "Campaña_ID": ["TRIGO_Camp_1"] * 7 + ["SOJA_Camp_2"],
        "CULTIVO": ["TRIGO"] * 7 + ["SOJA"],
        
        # Row 0: B=100 (Should be dropped)
        # Row 1: Sum=90 (Should be dropped because <= 95)
        # Row 2 to 6: Valid TRIGO rows
        # Row 7: Valid SOJA row (Should be dropped at the end)
        "Estado General MB": [0,   0,  10, 10, 10, 10, 0,  50],
        "Estado General B":  [100, 90, 90, 90, 90, 90, 40, 50],
        "Estado General R":  [0,   0,  0,  0,  0,  0,  10, 0],
        "Estado General M":  [0,   0,  0,  0,  0,  0,  50, 0], # Row 6 sum = 100, Score = -100 -10 + 40 = -70
        
        # Phenological columns
        "Estadio Fenológico E": [10] * 8,
        "Estadio Fenológico C": [20] * 8,
        "Estadio Fenológico F": [50] * 8, # Max value -> Should be dominant
        "Estadio Fenológico L": [0] * 8,
        "Estadio Fenológico M": [0] * 8,
        
        # Environmental and dummy columns to pass the column selection
        "temp_acum_1m": [15] * 8,
        "viento_prom_1m": [10] * 8,
        "hum_prom_1m": [60] * 8,
        "pres_prom_1m": [1013] * 8,
        "DELEGACIÓN": ["TEST_DEL"] * 8,
        "FECHA_INICIO": ["2026-01-01"] * 8,
        "FECHA_FIN": ["2026-01-08"] * 8,
        "PROVINCIA": ["TEST_PROV"] * 8
    }
    
    df_mock_input = pd.DataFrame(mock_data)
    mock_read_parquet.return_value = df_mock_input

    # 2. Execute the function
    run_feature_engineering()

    
    # 3. Extract the dataframe
    df_result = captured_data['df']

    # --- ASSERTIONS ---
    
    # A. Crop Filtering
    assert "SOJA" not in df_result["CULTIVO"].unique(), "Crop filtering failed: non-TRIGO crops remain"
    
    # B. Row Dropping Logic
    assert len(df_result) == 5, f"Expected 5 rows after filtering, got {len(df_result)}"
    
    # C. Target Shifting and Binary Classification
    assert df_result.iloc[0]["Estado_Malo"] == 1.0, "Target calculation or shift(-4) logic failed"

    # D. Phenological Dominance
    assert df_result.iloc[0]["estado_fenologico_dominante"] == "Estadio Fenológico F", "Dominant phenological state was incorrectly identified"

    # E. Regex Extraction
    assert df_result.iloc[0]["Camp_ID"] == "Camp_1", "Regex extraction for Camp_ID failed"
    
    # F. Column structure
    expected_cols = [
        'DELEGACIÓN', 'FECHA_INICIO', 'FECHA_FIN', 'PROVINCIA', 'Campaña_ID', 'Camp_ID', 'CULTIVO',
        'Estadio Fenológico E', 'Estadio Fenológico C', 'Estadio Fenológico F', 'Estadio Fenológico L', 'Estadio Fenológico M',
        'estado_fenologico_dominante', 'Estado General MB', 'Estado General B', 'Estado General R', 'Estado General M', 
        'temp_acum_1m', 'viento_prom_1m', 'hum_prom_1m', 'pres_prom_1m', 'Estado_Malo'
    ]
    assert list(df_result.columns) == expected_cols, "Final columns do not match the expected output"