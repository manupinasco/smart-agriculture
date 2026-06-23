import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.extract import process_raw_agricultural_table
from src.data.transform import run_transformation
from unittest.mock import patch


@patch("pandas.DataFrame.to_parquet")
def test_extract_and_tidy_happy_path(mock_to_parquet):
    """
    Tests that the main parser correctly extracts data when the format is right.
    We mock a typical messy dataframe.
    """
    # Simulating a raw untidy dataframe from Tabula
    mock_numeric_string = "1 2 3 4 5 6 7 8 9 10 dummy1 dummy2 dummy3 16_05_2024"
    mock_data = {
        "SOJA": ["Header1", "Header2", "Header3", "BAHÍA BLANCA", "random_row"],
        "Unnamed: 1": ["nan", "nan", "nan", mock_numeric_string, "nan"]
    }
    df_raw_mock = pd.DataFrame(mock_data)
    df_raw_mock.loc[0, "Unnamed: 1"] = "Humedad"
    # Run the transformation logic on the extracted mock
    result_df = process_raw_agricultural_table(df_raw_mock)

    # Asserts to make sure it's not empty and shapes the data correctly
    assert result_df is not None, "The parser returned None for a valid mock table"
    assert not result_df.empty, "The resulting DataFrame is empty"
    assert "CULTIVO" in result_df.columns
    assert result_df.iloc[0]["CULTIVO"] == "SOJA"
    assert result_df.iloc[0]["PROVINCIA"] == "BUENOS AIRES"

@patch("pandas.DataFrame.to_parquet")
def test_transform_logic_with_mock(mock_to_parquet):
    """
    Tests campaign generation, 1-month rolling weather grouping, 
    and dictionary fallback mapping by executing the real script function.
    """
    mock_extract_data = {
        'CULTIVO': ['TRIGO (PAN)'] * 3 + ['TRIGO'] * 2,
        'DELEGACIÓN': ['RAFAELA', 'RAFAELA', 'RAFAELA', 'JUNÍN', 'JUNÍN'],
        'FECHA_INICIO': pd.to_datetime(['2026-05-01', '2026-05-08', '2026-06-15', '2026-05-01', '2026-05-08']),
        'FECHA_FIN': pd.to_datetime(['2026-05-08', '2026-05-15', '2026-06-22', '2026-05-08', '2026-05-15']), # <-- AGREGAR ESTA LÍNEA
        'Total_7d_prcp': [10.0, 20.0, 15.0, 30.0, 40.0],
        'Total_7d_temp': [15.0, 14.0, 10.0, 16.0, 15.0],
        'Promedio_7d_wspd': [5.0, 6.0, 4.0, 8.0, 7.0],
        'Promedio_7d_rhum': [80.0, 85.0, 90.0, 70.0, 75.0],
        'Promedio_7d_pres': [1013.0, 1012.0, 1015.0, 1010.0, 1011.0],
        'Estado General MB': ['20']*5, 'Estado General B': ['50']*5, 
        'Estado General R': ['20']*5, 'Estado General M': ['10']*5
    }
    df_input = pd.DataFrame(mock_extract_data)

    # Execute actual transformation function passing the custom mock
    df_result = run_transformation(df_input=df_input)

    # --- ASSERTIONS FOR TRANSFORM ---
    assert "TRIGO" in df_result['CULTIVO'].unique()
    assert "TRIGO (PAN)" not in df_result['CULTIVO'].unique(), "Crop names should be clean of '(PAN)'"

    rafaela_df = df_result[df_result['DELEGACIÓN'] == 'RAFAELA'].sort_values('FECHA_INICIO').reset_index(drop=True)
    assert rafaela_df.loc[0, 'Campaña_ID'] == "TRIGO_RAFAELA_Camp_1"
    assert rafaela_df.loc[2, 'Campaña_ID'] == "TRIGO_RAFAELA_Camp_2", "A time gap > 30 days should spawn a new campaign number"