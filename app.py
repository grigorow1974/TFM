import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

team_logos = {
    "ANA": "https://a.espncdn.com/i/teamlogos/mlb/500/laa.png",   # Los Angeles Angels
    "ARI": "https://a.espncdn.com/i/teamlogos/mlb/500/ari.png",   # Arizona Diamondbacks
    "ATL": "https://a.espncdn.com/i/teamlogos/mlb/500/atl.png",   # Atlanta Braves
    "BAL": "https://a.espncdn.com/i/teamlogos/mlb/500/bal.png",   # Baltimore Orioles
    "BOS": "https://a.espncdn.com/i/teamlogos/mlb/500/bos.png",   # Boston Red Sox
    "CHN": "https://a.espncdn.com/i/teamlogos/mlb/500/chc.png",   # Chicago Cubs
    "CIN": "https://a.espncdn.com/i/teamlogos/mlb/500/cin.png",   # Cincinnati Reds
    "CLE": "https://a.espncdn.com/i/teamlogos/mlb/500/cle.png",   # Cleveland Guardians
    "COL": "https://a.espncdn.com/i/teamlogos/mlb/500/col.png",   # Colorado Rockies
    "CHA": "https://a.espncdn.com/i/teamlogos/mlb/500/chw.png",   # Chicago White Sox
    "DET": "https://a.espncdn.com/i/teamlogos/mlb/500/det.png",   # Detroit Tigers
    "HOU": "https://a.espncdn.com/i/teamlogos/mlb/500/hou.png",   # Houston Astros
    "KCR": "https://a.espncdn.com/i/teamlogos/mlb/500/kc.png",    # Kansas City Royals
    "LAA": "https://a.espncdn.com/i/teamlogos/mlb/500/laa.png",   # Los Angeles Angels (nuevo ID)
    "LAN": "https://a.espncdn.com/i/teamlogos/mlb/500/lad.png",   # Los Angeles Dodgers
    "MIA": "https://a.espncdn.com/i/teamlogos/mlb/500/mia.png",   # Miami Marlins
    "MIL": "https://a.espncdn.com/i/teamlogos/mlb/500/mil.png",   # Milwaukee Brewers
    "MIN": "https://a.espncdn.com/i/teamlogos/mlb/500/min.png",   # Minnesota Twins
    "NYN": "https://a.espncdn.com/i/teamlogos/mlb/500/nym.png",   # New York Mets
    "NYA": "https://a.espncdn.com/i/teamlogos/mlb/500/nyy.png",   # New York Yankees
    "OAK": "https://a.espncdn.com/i/teamlogos/mlb/500/oak.png",   # Oakland Athletics
    "PHI": "https://a.espncdn.com/i/teamlogos/mlb/500/phi.png",   # Philadelphia Phillies
    "PIT": "https://a.espncdn.com/i/teamlogos/mlb/500/pit.png",   # Pittsburgh Pirates
    "SDN": "https://a.espncdn.com/i/teamlogos/mlb/500/sd.png",    # San Diego Padres
    "SEA": "https://a.espncdn.com/i/teamlogos/mlb/500/sea.png",   # Seattle Mariners
    "SFG": "https://a.espncdn.com/i/teamlogos/mlb/500/sf.png",    # San Francisco Giants
    "STL": "https://a.espncdn.com/i/teamlogos/mlb/500/stl.png",   # St. Louis Cardinals
    "TBR": "https://a.espncdn.com/i/teamlogos/mlb/500/tb.png",    # Tampa Bay Rays
    "TEX": "https://a.espncdn.com/i/teamlogos/mlb/500/tex.png",   # Texas Rangers
    "TOR": "https://a.espncdn.com/i/teamlogos/mlb/500/tor.png",   # Toronto Blue Jays
    "WSN": "https://a.espncdn.com/i/teamlogos/mlb/500/wsh.png",   # Washington Nationals
}

st.set_page_config(page_title="Predicción MLB OPS", page_icon="⚾", layout="wide",initial_sidebar_state="collapsed")

# ========================
# 1. FUNCIONES AUXILIARES Y DE CARGA - ACTUALIZADAS
# ========================

@st.cache_resource
def cargar_artifacts():
    """Carga todos los artifacts usando el sistema de producción actualizado."""
    try:
        # ACTUALIZADO: Usar archivos de la Sección 12
        with open('models/production_model.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        with open('models/app_functions.pkl', 'rb') as f:
            app_functions = pickle.load(f)
        
        # ACTUALIZADO: Cargar clusters desde production
        cluster_assignments = pd.read_csv('data/production/cluster_data.csv')
        
        return model_info, app_functions, cluster_assignments
        
    except FileNotFoundError as e:
        st.error(f"No se encontró archivo necesario: {e}")
        st.error("Ejecuta las Secciones 12 y 13 primero para generar los artifacts de producción")
        return None
    except Exception as e:
        st.error(f"Error cargando artifacts: {e}")
        return None

@st.cache_data
def cargar_datos():
    """Carga datos completos y lista curada de jugadores 2023."""
    try:
        # USAR DATOS COMPLETOS para predicciones
        df = pd.read_csv('data/batting_fe.csv')
        
        # USAR LISTA CURADA de jugadores activos 2023 con 3+ años experiencia
        jugadores_2023 = pd.read_csv('data/Jugadores_Prediccion.csv')
        
        print(f"Datos completos cargados: {df.shape}")
        print(f"Jugadores curados 2023: {len(jugadores_2023)}")
        
        return df, jugadores_2023
        
    except FileNotFoundError as e:
        st.error(f"Archivo no encontrado: {e}")
        st.error("Archivos necesarios: 'data/batting_fe.csv'")
        return None, None
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None, None

def get_player_id_by_name(nombre, jugadores_df):
    """Obtiene playerID por nombre con búsqueda más robusta."""
    if jugadores_df is None:
        return None
    
    # Búsqueda exacta primero
    res = jugadores_df[jugadores_df['full_name'] == nombre]
    if len(res) > 0:
        return res['playerID'].values[0]
    
    # Búsqueda parcial si no encuentra exacta
    res_partial = jugadores_df[jugadores_df['full_name'].str.contains(nombre, case=False, na=False)]
    if len(res_partial) > 0:
        return res_partial['playerID'].values[0]
    
    return None

# ========================
# 2. FUNCIONES DE PREDICCIÓN RESTAURADAS - VERSIÓN ORIGINAL
# ========================

def predict_player_multi_year(player_input, df_modern, best_model, cluster_assignments, model_mae, cluster_next_ops, hybrid_features, position_feature_names):
    """
    FUNCIÓN ORIGINAL DE LA SECCIÓN 11 - Sistema temporal exacto
    """
    # Buscar jugador por nombre o ID
    player_found = None

    if 'full_name' in df_modern.columns:
        name_matches = df_modern[df_modern['full_name'].str.contains(player_input, case=False, na=False)]
        if len(name_matches) > 0:
            player_found = name_matches['playerID'].iloc[0]

    if player_found is None:
        id_matches = df_modern[df_modern['playerID'].str.contains(player_input, case=False, na=False)]
        if len(id_matches) > 0:
            player_found = id_matches['playerID'].iloc[0]

    if player_found is None:
        return {'error': f"Jugador '{player_input}' no encontrado"}

    # Obtener datos del jugador
    player_career = df_modern[df_modern['playerID'] == player_found].sort_values('yearID')
    
    if len(player_career) < 4:
        return {'error': f"Historial insuficiente para {player_found} (mínimo 4 años)"}

    # Usar datos más recientes (2023 como último año disponible)
    player_2023 = player_career[player_career['yearID'] == 2023]
    if len(player_2023) == 0:
        # Usar último año disponible
        last_year = player_career['yearID'].max()
        player_latest = player_career[player_career['yearID'] == last_year]
        if len(player_latest) == 0:
            return {'error': f"No hay datos recientes para {player_found}"}
        current_season = player_latest.iloc[0]
    else:
        current_season = player_2023.iloc[0]

    # Info del jugador
    player_name = current_season.get('full_name', player_found)
    player_age_2024 = int(current_season['age'] + 1)
    player_team = current_season.get('teamID', 'N/A')
    career_ops = player_career['OPS'].mean()
    ops_latest = current_season['OPS']

    def predict_single_year(target_year):
        """Predice un año específico usando los 3 años anteriores"""
        
        # Obtener los 3 años históricos
        historical_years = [target_year - 3, target_year - 2, target_year - 1]
        historical_data = []
        
        for year in historical_years:
            year_data = player_career[player_career['yearID'] == year]
            if len(year_data) == 1:
                historical_data.append(year_data.iloc[0])
            else:
                return None  # No hay datos suficientes
        
        if len(historical_data) != 3:
            return None
        
        # Calcular features como en la función original
        seasons_3yr = pd.DataFrame(historical_data)
        current_season_pred = historical_data[-1]  # Año más reciente
        
        # Trends de 3 años
        trend_ops = (seasons_3yr.iloc[2]['OPS'] - seasons_3yr.iloc[0]['OPS']) / 2
        trend_avg = (seasons_3yr.iloc[2]['AVG'] - seasons_3yr.iloc[0]['AVG']) / 2
        trend_iso = (seasons_3yr.iloc[2]['ISO'] - seasons_3yr.iloc[0]['ISO']) / 2
        
        ops_3yr_avg = seasons_3yr['OPS'].mean()
        volatility_3yr = seasons_3yr['OPS'].std()
        
        # Recent form weight
        recent_form_weight = (seasons_3yr.iloc[0]['OPS'] + 
                             2*seasons_3yr.iloc[1]['OPS'] + 
                             3*seasons_3yr.iloc[2]['OPS']) / 6
        
        # Years since peak
        player_peak_year = seasons_3yr.loc[seasons_3yr['OPS'].idxmax(), 'yearID']
        years_since_peak = current_season_pred['yearID'] - player_peak_year
        
        # Features del año actual
        current_age = current_season_pred['age']
        current_ops = current_season_pred['OPS']
        current_avg = current_season_pred['AVG']
        current_iso = current_season_pred['ISO']
        current_k_pct = current_season_pred['K_PCT']
        current_bb_pct = current_season_pred['BB_PCT']
        current_pa = current_season_pred['PA']
        current_bmi = current_season_pred['BMI']
        current_babip = current_season_pred['BABIP']
        
        # Years experience
        years_experience = current_season_pred['yearID'] - player_career['yearID'].min() + 1
        
        # Features derivados
        age_squared = current_age ** 2
        ops_age_interaction = current_ops * current_age
        is_veteran = 1 if current_age >= 32 else 0
        is_rookie_era = 1 if current_age <= 25 else 0
        high_pa = 1 if current_pa >= 500 else 0
        
        # Cluster information
        beltre_cluster_data = cluster_assignments[
            (cluster_assignments['playerID'] == player_found) & 
            (cluster_assignments['yearID'] == current_season_pred['yearID'])
        ]
        
        if len(beltre_cluster_data) > 0:
            cluster_id = beltre_cluster_data.iloc[0]['cluster']
            cluster_avg_next_ops_val = cluster_next_ops.get(cluster_id, ops_3yr_avg)
        else:
            cluster_id = -1
            cluster_avg_next_ops_val = ops_3yr_avg
        
        # Posición
        primary_position = current_season_pred['primary_position']
        
        # Construir features dict
        features_dict = {
            'current_age': current_age,
            'current_OPS': current_ops,
            'current_AVG': current_avg,
            'current_ISO': current_iso,
            'current_K_PCT': current_k_pct,
            'current_BB_PCT': current_bb_pct,
            'current_PA': current_pa,
            'current_BMI': current_bmi,
            'current_BABIP': current_babip,
            'trend_ops': trend_ops,
            'trend_avg': trend_avg,
            'trend_iso': trend_iso,
            'ops_3yr_avg': ops_3yr_avg,
            'volatility_3yr': volatility_3yr,
            'years_experience': years_experience,
            'recent_form_weight': recent_form_weight,
            'years_since_peak': years_since_peak,
            'age_squared': age_squared,
            'ops_age_interaction': ops_age_interaction,
            'is_veteran': is_veteran,
            'is_rookie_era': is_rookie_era,
            'high_pa': high_pa,
            'cluster_avg_next_ops': cluster_avg_next_ops_val
        }
        
        # Clusters one-hot
        for cluster_col in [col for col in hybrid_features if col.startswith('cluster_') and col != 'cluster_avg_next_ops']:
            try:
                cluster_str = cluster_col.split('_')[1]
                if '.' in cluster_str:
                    cluster_num = int(float(cluster_str))
                else:
                    cluster_num = int(cluster_str)
                features_dict[cluster_col] = 1 if cluster_id == cluster_num else 0
            except ValueError:
                features_dict[cluster_col] = 0
        
        # Posiciones one-hot
        for pos_col in [col for col in hybrid_features if col.startswith('pos_')]:
            pos_name = pos_col.replace('pos_', '')
            features_dict[pos_col] = 1 if primary_position == pos_name else 0
        
        # Crear vector final
        feature_vector = []
        for feature_name in hybrid_features:
            if feature_name in features_dict:
                feature_vector.append(features_dict[feature_name])
            else:
                feature_vector.append(0)
        
        # Predicción
        prediction = best_model.predict([feature_vector])[0]
        return prediction

    # Generar predicciones secuenciales para 2024-2026
    predictions = []
    simulated_career = player_career.copy()
    
    for year_offset in range(1, 4):  # 2024, 2025, 2026
        target_year = current_season['yearID'] + year_offset
        target_age = current_season['age'] + year_offset
        
        # Para 2024: usar años históricos reales
        # Para 2025+: usar combinación de históricos + simulados
        if year_offset == 1:  # 2024
            prediction = predict_single_year(target_year)
        else:  # 2025, 2026
            # Usar los últimos 3 años disponibles (incluyendo simulaciones)
            if len(simulated_career) >= 3:
                # Tomar los últimos 3 años para calcular features
                recent_3_years = simulated_career.tail(3)
                
                # Calcular features usando estos 3 años
                seasons_3yr = recent_3_years
                current_season_pred = recent_3_years.iloc[-1]  # Más reciente
                
                # Trends de 3 años
                trend_ops = (seasons_3yr.iloc[2]['OPS'] - seasons_3yr.iloc[0]['OPS']) / 2
                trend_avg = (seasons_3yr.iloc[2]['AVG'] - seasons_3yr.iloc[0]['AVG']) / 2
                trend_iso = (seasons_3yr.iloc[2]['ISO'] - seasons_3yr.iloc[0]['ISO']) / 2
                
                ops_3yr_avg = seasons_3yr['OPS'].mean()
                volatility_3yr = seasons_3yr['OPS'].std()
                
                # Recent form weight
                recent_form_weight = (seasons_3yr.iloc[0]['OPS'] + 
                                     2*seasons_3yr.iloc[1]['OPS'] + 
                                     3*seasons_3yr.iloc[2]['OPS']) / 6
                
                # Years since peak (del historial completo)
                player_peak_year = simulated_career.loc[simulated_career['OPS'].idxmax(), 'yearID']
                years_since_peak = current_season_pred['yearID'] - player_peak_year
                
                # Features del año actual
                current_age = target_age  # Edad del año que queremos predecir
                current_ops = current_season_pred['OPS']
                current_avg = current_season_pred['AVG']
                current_iso = current_season_pred['ISO']
                current_k_pct = current_season_pred['K_PCT']
                current_bb_pct = current_season_pred['BB_PCT']
                current_pa = current_season_pred['PA']
                current_bmi = current_season_pred['BMI']
                current_babip = current_season_pred['BABIP']
                
                # Years experience
                years_experience = current_season_pred['yearID'] - player_career['yearID'].min() + 1
                
                # Features derivados
                age_squared = current_age ** 2
                ops_age_interaction = current_ops * current_age
                is_veteran = 1 if current_age >= 32 else 0
                is_rookie_era = 1 if current_age <= 25 else 0
                high_pa = 1 if current_pa >= 500 else 0
                
                # Cluster information (usar año real más reciente)
                real_years = simulated_career[simulated_career['yearID'] <= current_season['yearID']]
                if len(real_years) > 0:
                    latest_real_year = real_years.iloc[-1]['yearID']
                    cluster_data = cluster_assignments[
                        (cluster_assignments['playerID'] == player_found) & 
                        (cluster_assignments['yearID'] == latest_real_year)
                    ]
                    
                    if len(cluster_data) > 0:
                        cluster_id = cluster_data.iloc[0]['cluster']
                        cluster_avg_next_ops_val = cluster_next_ops.get(cluster_id, ops_3yr_avg)
                    else:
                        cluster_id = -1
                        cluster_avg_next_ops_val = ops_3yr_avg
                else:
                    cluster_id = -1
                    cluster_avg_next_ops_val = ops_3yr_avg
                
                # Posición (usar la más reciente real)
                primary_position = current_season['primary_position']
                
                # Construir features dict
                features_dict = {
                    'current_age': current_age,
                    'current_OPS': current_ops,
                    'current_AVG': current_avg,
                    'current_ISO': current_iso,
                    'current_K_PCT': current_k_pct,
                    'current_BB_PCT': current_bb_pct,
                    'current_PA': current_pa,
                    'current_BMI': current_bmi,
                    'current_BABIP': current_babip,
                    'trend_ops': trend_ops,
                    'trend_avg': trend_avg,
                    'trend_iso': trend_iso,
                    'ops_3yr_avg': ops_3yr_avg,
                    'volatility_3yr': volatility_3yr,
                    'years_experience': years_experience,
                    'recent_form_weight': recent_form_weight,
                    'years_since_peak': years_since_peak,
                    'age_squared': age_squared,
                    'ops_age_interaction': ops_age_interaction,
                    'is_veteran': is_veteran,
                    'is_rookie_era': is_rookie_era,
                    'high_pa': high_pa,
                    'cluster_avg_next_ops': cluster_avg_next_ops_val
                }
                
                # Clusters one-hot
                for cluster_col in [col for col in hybrid_features if col.startswith('cluster_') and col != 'cluster_avg_next_ops']:
                    try:
                        cluster_str = cluster_col.split('_')[1]
                        if '.' in cluster_str:
                            cluster_num = int(float(cluster_str))
                        else:
                            cluster_num = int(cluster_str)
                        features_dict[cluster_col] = 1 if cluster_id == cluster_num else 0
                    except ValueError:
                        features_dict[cluster_col] = 0
                
                # Posiciones one-hot
                for pos_col in [col for col in hybrid_features if col.startswith('pos_')]:
                    pos_name = pos_col.replace('pos_', '')
                    features_dict[pos_col] = 1 if primary_position == pos_name else 0
                
                # Crear vector final
                feature_vector = []
                for feature_name in hybrid_features:
                    if feature_name in features_dict:
                        feature_vector.append(features_dict[feature_name])
                    else:
                        feature_vector.append(0)
                
                # Predicción
                prediction = best_model.predict([feature_vector])[0]
            else:
                prediction = None
        
        if prediction is not None:
            # Agregar predicción a resultados
            pesimista = max(prediction - model_mae, 0.400)
            optimista = min(prediction + model_mae, 1.400)
            
            predictions.append({
                'year': target_year,
                'age': int(target_age),
                'pesimista': pesimista,
                'realista': prediction,
                'optimista': optimista
            })
            
            # SIMULAR TEMPORADA PARA SIGUIENTE PREDICCIÓN
            if year_offset < 3:  # No simular después del último año
                # Crear temporada simulada basada en la predicción
                latest_season = simulated_career.iloc[-1].copy()
                
                # Actualizar año y edad
                latest_season['yearID'] = target_year
                latest_season['age'] = target_age
                
                # Actualizar métricas basadas en predicción
                ops_ratio = prediction / latest_season['OPS'] if latest_season['OPS'] > 0 else 1.0
                
                latest_season['OPS'] = prediction
                latest_season['AVG'] = min(latest_season['AVG'] * ops_ratio, 0.400)  # Cap realista
                latest_season['ISO'] = min(latest_season['ISO'] * ops_ratio, 0.500)  # Cap realista
                latest_season['K_PCT'] = max(latest_season['K_PCT'], 0.05)  # Floor realista
                latest_season['BB_PCT'] = min(latest_season['BB_PCT'] * (ops_ratio ** 0.5), 0.25)  # Disciplina cambia menos
                latest_season['BABIP'] = min(max(latest_season['BABIP'] * (ops_ratio ** 0.3), 0.250), 0.400)  # BABIP limitado
                
                # Mantener físico estable
                latest_season['BMI'] = latest_season['BMI']  # BMI no cambia dramáticamente
                latest_season['PA'] = min(latest_season['PA'], 650)  # PA realista
                
                # Agregar temporada simulada al historial
                simulated_career = pd.concat([simulated_career, pd.DataFrame([latest_season])], ignore_index=True)

    return {
        'player_name': player_name,
        'player_id': player_found,
        'age_2024': player_age_2024,
        'team': player_team,
        'career_ops': career_ops,
        'ops_2023': ops_latest,
        'predictions': predictions
    }


def predict_multiple_2024(player_list, df_modern, best_model, cluster_assignments, model_mae, cluster_next_ops, hybrid_features, position_feature_names):
    """
    ACTUALIZADO: Predicción múltiple usando la metodología correcta de la Sección 11
    """
    results = []
    
    for player_input in player_list:
        try:
            # Buscar jugador
            player_found = None
            if 'full_name' in df_modern.columns:
                name_matches = df_modern[df_modern['full_name'].str.contains(player_input, case=False, na=False)]
                if len(name_matches) > 0:
                    player_found = name_matches['playerID'].iloc[0]

            if player_found is None:
                id_matches = df_modern[df_modern['playerID'].str.contains(player_input, case=False, na=False)]
                if len(id_matches) > 0:
                    player_found = id_matches['playerID'].iloc[0]

            if player_found is None:
                continue

            # Obtener datos del jugador
            player_career = df_modern[df_modern['playerID'] == player_found].sort_values('yearID')
            
            if len(player_career) < 4:
                continue

            # Usar datos más recientes
            player_2023 = player_career[player_career['yearID'] == 2023]
            if len(player_2023) == 0:
                last_year = player_career['yearID'].max()
                player_latest = player_career[player_career['yearID'] == last_year]
                if len(player_latest) == 0:
                    continue
                current_season = player_latest.iloc[0]
            else:
                current_season = player_2023.iloc[0]

            player_name = current_season.get('full_name', player_found)
            player_age_2024 = int(current_season['age'] + 1)
            player_team = current_season.get('teamID', 'N/A')

            # USAR LA METODOLOGÍA CORRECTA PARA PREDECIR 2024
            target_year = 2024
            
            # Obtener los 3 años históricos para 2024: 2021, 2022, 2023
            historical_years = [2021, 2022, 2023]
            historical_data = []
            
            for year in historical_years:
                year_data = player_career[player_career['yearID'] == year]
                if len(year_data) == 1:
                    historical_data.append(year_data.iloc[0])
                else:
                    break  # No hay datos suficientes
            
            if len(historical_data) != 3:
                continue  # Saltar jugador sin suficientes datos
            
            # Calcular features como en la función original
            seasons_3yr = pd.DataFrame(historical_data)
            current_season_pred = historical_data[-1]  # 2023
            
            # Trends de 3 años
            trend_ops = (seasons_3yr.iloc[2]['OPS'] - seasons_3yr.iloc[0]['OPS']) / 2
            trend_avg = (seasons_3yr.iloc[2]['AVG'] - seasons_3yr.iloc[0]['AVG']) / 2
            trend_iso = (seasons_3yr.iloc[2]['ISO'] - seasons_3yr.iloc[0]['ISO']) / 2
            
            ops_3yr_avg = seasons_3yr['OPS'].mean()
            volatility_3yr = seasons_3yr['OPS'].std()
            
            # Recent form weight
            recent_form_weight = (seasons_3yr.iloc[0]['OPS'] + 
                                 2*seasons_3yr.iloc[1]['OPS'] + 
                                 3*seasons_3yr.iloc[2]['OPS']) / 6
            
            # Years since peak
            player_peak_year = seasons_3yr.loc[seasons_3yr['OPS'].idxmax(), 'yearID']
            years_since_peak = current_season_pred['yearID'] - player_peak_year
            
            # Features del año actual (2023)
            current_age = current_season_pred['age']
            current_ops = current_season_pred['OPS']
            current_avg = current_season_pred['AVG']
            current_iso = current_season_pred['ISO']
            current_k_pct = current_season_pred['K_PCT']
            current_bb_pct = current_season_pred['BB_PCT']
            current_pa = current_season_pred['PA']
            current_bmi = current_season_pred['BMI']
            current_babip = current_season_pred['BABIP']
            
            # Years experience
            years_experience = current_season_pred['yearID'] - player_career['yearID'].min() + 1
            
            # Features derivados
            age_squared = current_age ** 2
            ops_age_interaction = current_ops * current_age
            is_veteran = 1 if current_age >= 32 else 0
            is_rookie_era = 1 if current_age <= 25 else 0
            high_pa = 1 if current_pa >= 500 else 0
            
            # Cluster information
            cluster_data = cluster_assignments[
                (cluster_assignments['playerID'] == player_found) & 
                (cluster_assignments['yearID'] == current_season_pred['yearID'])
            ]
            
            if len(cluster_data) > 0:
                cluster_id = cluster_data.iloc[0]['cluster']
                cluster_avg_next_ops_val = cluster_next_ops.get(cluster_id, ops_3yr_avg)
            else:
                cluster_id = -1
                cluster_avg_next_ops_val = ops_3yr_avg
            
            # Posición
            primary_position = current_season_pred['primary_position']
            
            # Construir features dict
            features_dict = {
                'current_age': current_age,
                'current_OPS': current_ops,
                'current_AVG': current_avg,
                'current_ISO': current_iso,
                'current_K_PCT': current_k_pct,
                'current_BB_PCT': current_bb_pct,
                'current_PA': current_pa,
                'current_BMI': current_bmi,
                'current_BABIP': current_babip,
                'trend_ops': trend_ops,
                'trend_avg': trend_avg,
                'trend_iso': trend_iso,
                'ops_3yr_avg': ops_3yr_avg,
                'volatility_3yr': volatility_3yr,
                'years_experience': years_experience,
                'recent_form_weight': recent_form_weight,
                'years_since_peak': years_since_peak,
                'age_squared': age_squared,
                'ops_age_interaction': ops_age_interaction,
                'is_veteran': is_veteran,
                'is_rookie_era': is_rookie_era,
                'high_pa': high_pa,
                'cluster_avg_next_ops': cluster_avg_next_ops_val
            }
            
            # Clusters one-hot
            for cluster_col in [col for col in hybrid_features if col.startswith('cluster_') and col != 'cluster_avg_next_ops']:
                try:
                    cluster_str = cluster_col.split('_')[1]
                    if '.' in cluster_str:
                        cluster_num = int(float(cluster_str))
                    else:
                        cluster_num = int(cluster_str)
                    features_dict[cluster_col] = 1 if cluster_id == cluster_num else 0
                except ValueError:
                    features_dict[cluster_col] = 0
            
            # Posiciones one-hot
            for pos_col in [col for col in hybrid_features if col.startswith('pos_')]:
                pos_name = pos_col.replace('pos_', '')
                features_dict[pos_col] = 1 if primary_position == pos_name else 0
            
            # Crear vector final
            feature_vector = []
            for feature_name in hybrid_features:
                if feature_name in features_dict:
                    feature_vector.append(features_dict[feature_name])
                else:
                    feature_vector.append(0)
            
            # Verificar dimensiones
            if len(feature_vector) != len(hybrid_features):
                continue
            
            # Predicción
            prediccion = best_model.predict([feature_vector])[0]
            pesimista = max(prediccion - model_mae, 0.400)
            optimista = min(prediccion + model_mae, 1.400)

            results.append({
                'player_name': player_name,
                'age_2024': int(player_age_2024),
                'team': player_team,
                'pesimista': pesimista,
                'realista': prediccion,
                'optimista': optimista
            })
            
        except Exception as e:
            # Silenciosamente saltar jugadores con errores
            continue
    
    return results

# ========================
# 3. CARGA DE ARTIFACTS Y DATOS CON VALIDACIÓN - ACTUALIZADA
# ========================

# Cargar artifacts con manejo de errores
artifacts_loaded = cargar_artifacts()
if artifacts_loaded is None:
    st.stop()  # Detener la app si no se pueden cargar los artifacts

model_info, app_functions, cluster_assignments = artifacts_loaded

# Cargar datos con manejo de errores
df_modern, jugadores_df = cargar_datos()
if df_modern is None:
    st.stop()  # Detener la app si no se pueden cargar los datos

# ACTUALIZADO: Obtener variables del sistema de producción
model = model_info['best_model']
model_mae = model_info['best_mae']
cluster_next_ops = model_info['cluster_next_ops']
hybrid_features = model_info['hybrid_features']
position_feature_names = app_functions['position_features']



# ACTUALIZADO: Lista de jugadores desde archivos de producción
lista_jugadores = jugadores_df['full_name'].dropna().drop_duplicates().sort_values().tolist()

# VERIFICACIÓN DE DATOS CARGADOS
st.sidebar.markdown("### 📊 Información del Sistema")
st.sidebar.markdown(f"**Jugadores disponibles:** {len(lista_jugadores):,}")
st.sidebar.markdown(f"**Registros totales:** {len(df_modern):,}")
st.sidebar.markdown(f"**Años cubiertos:** {df_modern['yearID'].min()}-{df_modern['yearID'].max()}")
st.sidebar.markdown(f"**MAE del modelo:** {model_mae:.4f}")

# Verificar específicamente algunos jugadores estrella
estrellas = ['Mike Trout', 'Mookie Betts', 'Aaron Judge', 'Ronald Acuna Jr.']
estrellas_encontradas = [nombre for nombre in estrellas if nombre in lista_jugadores]
if estrellas_encontradas:
    st.sidebar.markdown(f"**Estrellas disponibles:** {len(estrellas_encontradas)}/4")

# ========================
# 4. CABECERA VISUAL (SIN CAMBIOS)
# ========================
st.markdown(
    """
    <style>
        @media only screen and (max-width: 768px) {
            .header-container {
                flex-direction: column !important;
                text-align: center !important;
            }
            .header-container img {
                margin: 10px auto !important;
            }
        }
    </style>

    <div class='header-container' style='
        background-color:#002654;
        padding:18px;
        border-radius:12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
    '>
        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Major_League_Baseball_logo.svg/1024px-Major_League_Baseball_logo.svg.png'
             width='140' style='margin: 10px; border-radius:12px; background:white; padding:6px 14px;'/>
        <div style='flex-grow:1; text-align:center; min-width: 250px;'>
            <h2 style='color:white; margin-bottom: 9px; font-size: 1.8em;'>⚾ Predicción de Rendimiento OPS en MLB ⚾</h2>
            <p style='color:white; font-size:18px; margin:0;'>Herramienta interactiva de Machine Learning para proyectar desempeño ofensivo</p>
        </div>
        <img src='https://sabr.org/sites/default/files/SABR_logo-square-700px.png'
             width='100' style='margin: 10px; border-radius:14px; background:white; padding:6px 14px;'/>
    </div>
    """,
    unsafe_allow_html=True
)


# ========================
# 5. SELECCIÓN DE MODO Y JUGADORES (SIN CAMBIOS)
# ========================
modo = st.radio(
    "Selecciona el tipo de predicción:",
    ("Predicción individual", "Predicción múltiple"),
    horizontal=True
)

if modo == "Predicción individual":
    jugador_seleccionado = st.selectbox("Selecciona el jugador:", lista_jugadores)
else:
    jugadores_seleccionados = st.multiselect(
        "Selecciona uno o más jugadores:", 
        lista_jugadores, 
        default=lista_jugadores[:2] if len(lista_jugadores) >= 2 else lista_jugadores
    )

# ========================
# 6. BOTÓN PARA LANZAR PREDICCIÓN Y MOSTRAR RESULTADOS - RESTAURADO
# ========================
if st.button("Predecir"):
    st.success("¡Predicción realizada!")

    if modo == "Predicción individual":
        player_id = get_player_id_by_name(jugador_seleccionado, jugadores_df)
        if player_id is None:
            st.error("No se encontró el jugador.")
        else:
            resultado = predict_player_multi_year(
                jugador_seleccionado, df_modern, model, cluster_assignments, model_mae, 
                cluster_next_ops, hybrid_features, position_feature_names
            )
            if 'error' in resultado:
                st.error(resultado['error'])
            else:
                team_id = resultado['team']
                logo_url = team_logos.get(team_id, None)

                st.markdown(f"""
                <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 16px;'>
                    <div style='font-size:1.38em; line-height:1.22; text-align: center;'>
                        <b>Jugador:</b> {resultado['player_name']}<br>
                        <b>Edad en 2024:</b> {resultado['age_2024']}<br>
                        <b>OPS Carrera:</b> {resultado['career_ops']:.3f}<br>
                        <b>OPS 2023:</b> {resultado['ops_2023']:.3f}
                    </div>
                    <div style='margin-left:36px; text-align:center;'>
                        {"<img src='" + logo_url + "' width='155' style='border-radius:12px; background:white; padding:6px 18px; vertical-align:middle;'/>" if logo_url else f"<span style='color:#ccc;'>Sin logo<br>({team_id})</span>"}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                for pred in resultado['predictions']:
                    year = pred['year']
                    age = pred['age'] 
                    pesimista = pred['pesimista']
                    realista = pred['realista']
                    optimista = pred['optimista']

                    if realista >= 0.900:
                        color = "#28a745"
                        emoji = "🔥"
                        text_color = "#1e7e34"
                    elif realista >= 0.800:
                        color = "#17a2b8"
                        emoji = "⭐"
                        text_color = "#117a8b"
                    elif realista >= 0.700:
                        color = "#ffc107"
                        emoji = "👍"
                        text_color = "#d39e00"
                    else:
                        color = "#dc3545"
                        emoji = "⚠️"
                        text_color = "#bd2130"

                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {color}15, {color}05); 
                                border-left: 4px solid {color}; 
                                padding: 15px 20px; 
                                margin: 8px auto; 
                                border-radius: 8px; 
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                max-width: 600px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div style='font-size: 20px; font-weight: bold; color: {text_color};'>
                                {emoji} {year} (Edad {age})
                            </div>
                            <div style='font-size: 28px; font-weight: bold; color: {text_color};'>
                                {realista:.3f}
                            </div>
                        </div>
                        <div style='margin-top: 8px; display: flex; justify-content: space-between; font-size: 14px;'>
                            <span style='color: #721c24;'>📉 Pesimista: <b>{pesimista:.3f}</b></span>
                            <span style='color: #495057;'>📊 Realista: <b>{realista:.3f}</b></span>
                            <span style='color: #155724;'>📈 Optimista: <b>{optimista:.3f}</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                if resultado['predictions'] and len(resultado['predictions']) > 0:
                    years = [pred['year'] for pred in resultado['predictions']]
                    pesimista = [pred['pesimista'] for pred in resultado['predictions']]
                    realista = [pred['realista'] for pred in resultado['predictions']]
                    optimista = [pred['optimista'] for pred in resultado['predictions']]

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.fill_between(years, pesimista, optimista, color='skyblue', alpha=0.3, label='Rango ±MAE')
                    ax.plot(years, realista, marker='o', label='Predicción central', color='blue')
                    ax.set_ylabel('OPS')
                    ax.set_ylim(min(pesimista) - 0.05, max(optimista) + 0.05)
                    ax.set_title('Proyección de OPS')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ No se pudieron generar predicciones válidas para este jugador.")
    else:
        if not jugadores_seleccionados:
            st.error("Selecciona al menos un jugador válido.")
        else:
            resultados = predict_multiple_2024(
                jugadores_seleccionados, df_modern, model, cluster_assignments, model_mae,
                cluster_next_ops, hybrid_features, position_feature_names
            )

            if not resultados:
                st.error("No se encontraron jugadores válidos.")
            else:
                tabla_multi = pd.DataFrame(resultados).reset_index(drop=True)

                st.markdown("### 📋 Resultados por Jugador - Predicción 2024")

                tabla_html = (
                    "<table style='width:100%; border-collapse:collapse; font-size:18px; text-align:center;'>"
                    "<thead>"
                    "<tr style='background-color:#003366; color:white;'>"
                    "<th style='padding:10px;'>Jugador</th>"
                    "<th style='padding:10px;'>Edad</th>"
                    "<th style='padding:10px;'>Equipo</th>"
                    "<th style='padding:10px;'>OPS Pesimista</th>"
                    "<th style='padding:10px;'>OPS Realista</th>"
                    "<th style='padding:10px;'>OPS Optimista</th>"
                    "</tr>"
                    "</thead><tbody>"
                )

                for _, row in tabla_multi.iterrows():
                    tabla_html += (
                        f"<tr>"
                        f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{row['player_name']}</td>"
                        f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{row['age_2024']}</td>"
                        f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{row['team']}</td>"
                        f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{row['pesimista']:.3f}</td>"
                        f"<td style='padding:8px; border-bottom:1px solid #ddd;'><b>{row['realista']:.3f}</b></td>"
                        f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{row['optimista']:.3f}</td>"
                        f"</tr>"
                    )

                tabla_html += "</tbody></table>"

                st.markdown(tabla_html, unsafe_allow_html=True)


                # 🎯 Gráfico estilo original: Predicciones separadas por tipo
                player_names = [result['player_name'][:15] for result in resultados]
                pesimistas = [result['pesimista'] for result in resultados]
                realistas = [result['realista'] for result in resultados]
                optimistas = [result['optimista'] for result in resultados]

                n_players = len(player_names)

                x_pesimista = np.arange(n_players)
                x_realista = x_pesimista + n_players + 0.5
                x_optimista = x_pesimista + 2 * n_players + 1

                fig, ax = plt.subplots(figsize=(16, 6))

                # Barras por tipo
                ax.bar(x_pesimista, pesimistas, color='lightcoral', alpha=0.8, label='Pesimista')
                ax.bar(x_realista, realistas, color='steelblue', alpha=0.8, label='Realista')
                ax.bar(x_optimista, optimistas, color='lightgreen', alpha=0.8, label='Optimista')

                # Valores encima de cada barra
                for i in range(n_players):
                    ax.text(x_pesimista[i], pesimistas[i] + 0.005, f'{pesimistas[i]:.3f}', ha='center', fontsize=9)
                    ax.text(x_realista[i], realistas[i] + 0.005, f'{realistas[i]:.3f}', ha='center', fontsize=9, fontweight='bold')
                    ax.text(x_optimista[i], optimistas[i] + 0.005, f'{optimistas[i]:.3f}', ha='center', fontsize=9)

                # Ejes y diseño
                all_x = np.concatenate([x_pesimista, x_realista, x_optimista])
                all_labels = player_names + player_names + player_names
                ax.set_xticks(all_x)
                ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=10)
                ax.set_xlabel('Jugadores por Tipo de Predicción', fontsize=12)
                ax.set_ylabel('OPS 2024', fontsize=12)

                ax.set_ylim([0.35, min(1.45, max(optimistas + realistas + pesimistas) * 1.15)])

                # Líneas divisorias entre grupos
                if n_players > 1:
                    ax.axvline(x=n_players - 0.5, color='gray', linestyle='--', alpha=0.5)
                    ax.axvline(x=2 * n_players + 0.5, color='gray', linestyle='--', alpha=0.5)

                # Etiquetas de grupo
                y_top = ax.get_ylim()[1]
                ax.text(np.mean(x_pesimista), y_top * 0.96, 'PESIMISTA', ha='center', fontsize=12, fontweight='bold', color='red')
                ax.text(np.mean(x_realista), y_top * 0.96, 'REALISTA', ha='center', fontsize=12, fontweight='bold', color='blue')
                ax.text(np.mean(x_optimista), y_top * 0.96, 'OPTIMISTA', ha='center', fontsize=12, fontweight='bold', color='green')

                ax.set_title(
                    f"Sistema Temporal OPS 2024: Pesimista | Realista | Optimista\n"
                    f"({type(model).__name__}, {len(hybrid_features)} features)",
                    fontsize=14, fontweight='bold'
                )

                ax.grid(axis='y', linestyle='--', alpha=0.3)
                ax.legend()
                plt.tight_layout()

                # Mostrar en Streamlit
                st.pyplot(fig)




# ========================
# 7. PIE DE PÁGINA VISUAL (SIN CAMBIOS)
# ========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:16px;color:gray;'>Proyecto TFM - Máster Data Science UCM, 2024<br>Autor: Sergio Grigorow</p>",
    unsafe_allow_html=True
)