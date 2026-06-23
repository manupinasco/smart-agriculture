import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# IO Paths
INPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features_data.parquet"
PLOTS_DIR = Path(__file__).resolve().parents[2] / "src" / "plots" / "EDA"

PALETTE = {"Bueno": "darkgreen", "Malo": "red"}

def load_visualization_data():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing processed feature data at {INPUT_PATH}. Run build_features.py first.")
        
    df = pd.read_parquet(INPUT_PATH)

    df["state"] = df["Estado_Malo"].replace({1.0: "Malo", 0.0: "Bueno"})

    return df

def plot_phenological_distribution(df,filename):
    """ Plots the crop state percentage across dominant phenological stages """
    
    # We calculate the current state locally just for this plot
    df_local = df.copy()
    df_local["state_actual"] = df_local["state"].shift(4)

    # Calculate cross-tabulation normalized by index (%)
    tabla = pd.crosstab(
        df_local["state"],
        df_local["estado_fenologico_dominante"],
        normalize="index"
    ) * 100
    
    # Transpose to put stages on X axis and stack/group by state
    tabla.T.plot(
        kind="bar",
        figsize=(6, 4),
        color=["darkgreen", "red"]
    )

    plt.ylabel("% de Estado")
    plt.xlabel("Estadío fenológico")
    plt.title("Distribución del estado de cultivo por estadío fenológico")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.show()

def plot_pressure_temperature_dashboard(df,filename):
    """ Plots KDE densities and Boxplots """

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
    
    # --- Row 1, Col 1: Pressure KDE ---
    sns.kdeplot(data=df, x="pres_prom_1m", hue="state", palette=PALETTE, fill=True, ax=axes[0, 0], alpha=0.4)
    axes[0, 0].set_title("Distribución de presión promedio (1 mes) por estado")
    axes[0, 0].set_xlabel("Presión promedio (1 mes) [hPa]")
    axes[0, 0].set_ylabel("Densidad")
    
    # --- Row 1, Col 2: Temperature KDE ---
    sns.kdeplot(data=df, x="temp_acum_1m", hue="state", palette=PALETTE, fill=True, ax=axes[0, 1], alpha=0.4)
    axes[0, 1].set_title("Distribución de temperatura acumulada (1 mes) por estado")
    axes[0, 1].set_xlabel("Temperatura acumulada (1 mes) [°C]")
    axes[0, 1].set_ylabel("Densidad")
    
    # --- Row 2, Col 1: Pressure Boxplot ---
    sns.boxplot(data=df, x="state", y="pres_prom_1m", palette=PALETTE, hue="state", legend=False, ax=axes[1, 0])
    axes[1, 0].set_title("Presión promedio (1 mes) por estado del cultivo")
    axes[1, 0].set_xlabel("Estado del cultivo")
    axes[1, 0].set_ylabel("Presión promedio (1 mes) [hPa]")
    
    # --- Row 2, Col 2: Temperature Boxplot ---
    sns.boxplot(data=df, x="state", y="temp_acum_1m", palette=PALETTE, hue="state", legend=False, ax=axes[1, 1])
    axes[1, 1].set_title("Temperatura acumulada (1 mes) por estado del cultivo")
    axes[1, 1].set_xlabel("Estado del cultivo")
    axes[1, 1].set_ylabel("Temperatura acumulada (1 mes) [°C]")
    
    plt.suptitle("Análisis de Presión y Temperatura vs Condición de Cultivo", fontsize=14, weight='bold', y=0.96)
    plt.savefig(PLOTS_DIR / filename)
    plt.show()

def plot_weather_scatter_plots(df,filename):
    """ Plots the most significant scatter correlations """
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'hspace': 0.3})
    
    # --- Scatter 1: Temp vs Humidity ---
    sns.scatterplot(data=df, x="temp_acum_1m", y="hum_prom_1m", hue="state", palette=PALETTE, alpha=0.8, ax=axes[0])
    axes[0].set_title("Temperatura acumulada vs humedad promedio por estado del cultivo")
    axes[0].set_xlabel("Temperatura acumulada (1 mes) [°C]")
    axes[0].set_ylabel("Humedad promedio (1 mes) [%]")
    axes[0].legend(title="Estado")
    
    # --- Scatter 2: Temp vs Wind Speed ---
    sns.scatterplot(data=df, x="temp_acum_1m", y="viento_prom_1m", hue="state", palette=PALETTE, alpha=0.8, ax=axes[1])
    axes[1].set_title("Temperatura acumulada vs velocidad del viento promedio por estado del cultivo")
    axes[1].set_xlabel("Temperatura acumulada (1 mes) [°C]")
    axes[1].set_ylabel("Velocidad del viento promedio (1 mes) [km/h]")
    axes[1].legend(title="Estado")
    
    plt.savefig(PLOTS_DIR / filename)
    plt.show()

if __name__ == "__main__":

    df = load_visualization_data()

    plot_phenological_distribution(df,"phenological_distribution")
    plot_pressure_temperature_dashboard(df,"pressure_temperature_dashboard")
    plot_weather_scatter_plots(df,"weather_scatter_plots")
    