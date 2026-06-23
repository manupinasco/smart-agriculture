import pandas as pd
import meteostat as ms

MAP_DELEGACIONES_LOCACIONES = {
    'BAHÍA BLANCA': [-38.7167, -62.2667, 20], 'BOLÍVAR': [-36.2333, -61.1167, 102],
    'BRAGADO': [-35.1194, -60.4781, 50], 'GENERAL MADARIAGA': [-37.0000, -57.1333, 8],
    'JUNÍN': [-34.5833, -60.9500, 81], 'LA PLATA': [-34.9214, -57.9544, 26],
    'LINCOLN': [-34.8500, -61.5167, 76], 'PEHUAJÓ': [-35.8158, -61.8939, 84],
    'PERGAMINO': [-33.8833, -60.5667, 56], 'PIGÜÉ': [-37.6000, -62.4000, 304],
    'SALLIQUELÓ': [-36.7500, -62.8333, 121], 'TANDIL': [-37.3189, -59.1347, 188],
    'TRES ARROYOS': [-38.3667, -60.2667, 98], '25 DE MAYO': [-35.4167, -60.1667, 58],
    'LABOULAYE': [-34.1225, -63.3878, 131], 'MARCOS JUÁREZ': [-32.7000, -62.0333, 102],
    'RÍO CUARTO': [-33.1333, -64.3333, 452], 'SAN FRANCISCO': [-31.4167, -62.0833, 104],
    'VILLA MARÍA': [-32.4119, -63.2422, 196], 'PARANÁ': [-31.7331, -60.5300, 77],
    'ROSARIO DEL TALA': [-32.3008, -59.1389, 41], 'GENERAL PICO': [-35.6611, -63.7506, 103],
    'SANTA ROSA': [-36.6203, -64.2906, 175], 'AVELLANEDA': [-29.1175, -59.6583, 44],
    'CAÑADA de GÓMEZ': [-32.8225, -61.3950, 88], 'CASILDA': [-33.0442, -61.1681, 73],
    'RAFAELA': [-31.2508, -61.4719, 104], 'VENADO TUERTO': [-33.7458, -61.9681, 111],
    'CATAMARCA': [-28.4686, -65.7792, 519], 'CORRIENTES': [-27.4833, -58.8167, 52],
    'CHARATA': [-27.2150, -61.1917, 100], 'ROQUE SÁENZ PEÑA': [-26.7833, -60.4500, 90],
    'SALTA': [-24.7833, -65.4167, 1187], 'SAN LUIS': [-33.3000, -66.3333, 709],
    'SANTIAGO DEL ESTERO': [-27.7844, -64.2669, 187], 'QUIMILÍ': [-27.6333, -62.4167, 137],
    'TUCUMÁN': [-26.8167, -65.2167, 450]
}

def fetch_weather_data(start_date, end_date):
    """ Downloads weather data from Meteostat for all our custom regions """
    weather_df = pd.DataFrame()

    for region, coords in MAP_DELEGACIONES_LOCACIONES.items():
        try:
            point = ms.Point(*coords)
            stations = ms.stations.nearby(point, limit=4)
            daily_data = ms.daily(stations, start_date, end_date)
            df = ms.interpolate(daily_data, point).fetch()
            df["NOMBRE"] = region

            if weather_df.empty:
                weather_df = df
            else:
                weather_df = pd.concat([weather_df, df])
        except Exception as e:
            print(f"Skipping weather for {region} due to error")

    weather_df["time"]=weather_df.index 
    weather_df.reset_index(drop=True,inplace=True) 
    return weather_df

def compute_7d_rolling_features(weather_df):
    """ Calculates rolling stats (sum, mean, min, max) for a 7-day window """
    df_clima = weather_df.copy()
    weather_vars = ["temp", "rhum", "prcp", "wspd", "pres"]

    for var in weather_vars:
        df_clima[f"Total_7d_{var}"] = df_clima[var].iloc[::-1].rolling(window=7).sum().iloc[::-1]
        df_clima[f"Promedio_7d_{var}"] = df_clima[var].iloc[::-1].rolling(window=7).mean().iloc[::-1]
        df_clima[f"Min_7d_{var}"] = df_clima[var].iloc[::-1].rolling(window=7).min().iloc[::-1]
        df_clima[f"Max_7d_{var}"] = df_clima[var].iloc[::-1].rolling(window=7).max().iloc[::-1]
        
    return df_clima