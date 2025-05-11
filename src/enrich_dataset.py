import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

def enrich_insurance_dataset(df):
    """
    Arricchisce il dataset assicurativo con dati esterni.
    
    Args:
        df: DataFrame preprocessato
        
    Returns:
        DataFrame: Dataset arricchito
    """
    df_enriched = df.copy()
    
    # 1. DATI INTERNI ALLA MUTUA (simulati)
    
    # 1.1 Storico dei clienti (customer tenure)
    # In un caso reale, si userebbe un dataset interno con lo storico dei clienti
    # Simuliamo l'arricchimento con dati casuali
    np.random.seed(42)  # Per riproducibilità
    
    # Simula anni di anzianità cliente
    df_enriched['CUSTOMER_TENURE_YEARS'] = np.random.randint(0, 15, size=len(df_enriched))
    
    # 1.2 Storico dei sinistri precedenti
    # Simula numero di sinistri precedenti
    df_enriched['PREVIOUS_CLAIMS_COUNT'] = np.random.poisson(1.5, size=len(df_enriched))
    
    # 1.3 Valore del portafoglio cliente
    # Simula numero di polizze attive per cliente
    df_enriched['ACTIVE_POLICIES_COUNT'] = np.random.randint(1, 5, size=len(df_enriched))
    
    # Simula premio annuale totale
    base_premium = df_enriched['PREMIUM_AMOUNT_PAID']
    df_enriched['CUSTOMER_ANNUAL_PREMIUM'] = base_premium * df_enriched['ACTIVE_POLICIES_COUNT'] * np.random.uniform(0.8, 1.2, size=len(df_enriched))
    
    # 1.4 Canale di acquisizione
    channels = ['Agente', 'Online', 'Bancassurance', 'Broker', 'Call Center']
    weights = [0.45, 0.25, 0.15, 0.10, 0.05]  # Distribuzione probabilità
    df_enriched['ACQUISITION_CHANNEL'] = np.random.choice(channels, size=len(df_enriched), p=weights)
    
    # 1.5 Upsell/Cross-sell opportunities
    # Basato sul numero di polizze attive, identifichiamo opportunità di cross-selling
    df_enriched['CROSS_SELL_POTENTIAL'] = (df_enriched['ACTIVE_POLICIES_COUNT'] < 3).astype(int)
    
    # 2. DATI ESTERNI PUBBLICI
    
    # 2.1 Dati demografici ISTAT per CAP/Comune/Provincia
    # In un caso reale, si utilizzerebbero API o dataset ISTAT
    # Simuliamo l'arricchimento con dati casuali per province
    
    # Dizionario simulato province -> reddito medio
    provinces_income = {
        'TO': 28500, 'MI': 32000, 'RM': 30000, 'NA': 22000, 'BO': 29000,
        'FI': 28000, 'BA': 23000, 'PA': 21500, 'CT': 22000, 'VE': 27000,
        'GE': 27500, 'CA': 24000, 'RC': 21000, 'PG': 25000, 'AQ': 24500,
        'BZ': 31000, 'PD': 28500, 'UD': 27000, 'LE': 22500, 'PE': 24000
    }
    
    # Aggiungi reddito medio provinciale per province conosciute
    df_enriched['PROVINCE_AVG_INCOME'] = df_enriched['CLAIM_PROVINCE'].map(
        lambda x: provinces_income.get(x, 25000 + np.random.normal(0, 2000))  # Valore di default + rumore
    )
    
    # 2.2 Dati sul rischio geografico (ad es. tasso criminalità, incidenti stradali per provincia)
    # Dizionario simulato province -> tasso incidenti per 1000 veicoli
    provinces_accident_rate = {
        'TO': 8.2, 'MI': 9.1, 'RM': 10.3, 'NA': 9.7, 'BO': 7.5,
        'FI': 7.8, 'BA': 8.9, 'PA': 9.2, 'CT': 8.8, 'VE': 6.9,
        'GE': 8.5, 'CA': 7.2, 'RC': 8.1, 'PG': 6.7, 'AQ': 6.5,
        'BZ': 5.8, 'PD': 7.1, 'UD': 6.3, 'LE': 8.4, 'PE': 7.3
    }
    
    # Aggiungi tasso di incidenti per provincia
    df_enriched['PROVINCE_ACCIDENT_RATE'] = df_enriched['CLAIM_PROVINCE'].map(
        lambda x: provinces_accident_rate.get(x, 8.0 + np.random.normal(0, 1.0))  # Valore di default + rumore
    )
    
    # 2.3 Dati meteo (da servizi come il Weather Company)
    # In un caso reale, si integrerebbero i dati meteo storici nelle date dei sinistri
    # Simuliamo l'aggiunta di condizioni meteo casuali
    weather_conditions = ['Soleggiato', 'Nuvoloso', 'Pioggia', 'Pioggia intensa', 'Nebbia', 'Neve']
    weather_weights = [0.5, 0.2, 0.15, 0.07, 0.05, 0.03]  # Distribuzione probabilità
    
    # Aggiungi condizioni meteo simulate
    df_enriched['WEATHER_CONDITION'] = np.random.choice(
        weather_conditions, size=len(df_enriched), p=weather_weights
    )
    
    # 2.4 Dati sui veicoli (valore, potenza, classe di rischio)
    # Dizionario simulato marche -> valore medio del veicolo
    brand_avg_value = {
        'FIAT': 15000, 'VOLKSWAGEN': 25000, 'FORD': 22000, 'MERCEDES': 45000,
        'BMW': 48000, 'AUDI': 42000, 'RENAULT': 18000, 'TOYOTA': 25000,
        'OPEL': 19000, 'PEUGEOT': 20000, 'CITROEN': 18500, 'NISSAN': 24000,
        'ALFA ROMEO': 30000, 'LANCIA': 17000, 'SMART': 16000, 'JEEP': 35000,
        'HYUNDAI': 22000, 'KIA': 21000, 'VOLVO': 38000, 'LAND ROVER': 60000
    }
    
    # Aggiungi valore medio veicolo
    df_enriched['VEHICLE_ESTIMATED_VALUE'] = df_enriched['VEHICLE_BRAND'].map(
        lambda x: brand_avg_value.get(x, 25000 + np.random.normal(0, 5000))  # Valore di default + rumore
    )
    
    # 2.5 Dati sulle caratteristiche di rischio dei veicoli
    # Dizionario simulato marche -> potenza media CV
    brand_avg_power = {
        'FIAT': 85, 'VOLKSWAGEN': 120, 'FORD': 110, 'MERCEDES': 180,
        'BMW': 190, 'AUDI': 170, 'RENAULT': 95, 'TOYOTA': 115,
        'OPEL': 100, 'PEUGEOT': 105, 'CITROEN': 95, 'NISSAN': 120,
        'ALFA ROMEO': 150, 'LANCIA': 90, 'SMART': 70, 'JEEP': 160,
        'HYUNDAI': 110, 'KIA': 105, 'VOLVO': 150, 'LAND ROVER': 200
    }
    
    # Aggiungi potenza media veicolo
    df_enriched['VEHICLE_POWER_HP'] = df_enriched['VEHICLE_BRAND'].map(
        lambda x: brand_avg_power.get(x, 120 + np.random.normal(0, 20))  # Valore di default + rumore
    ).astype(int)
    
    # 2.6 Calcolo del rischio veicolo basato su statistiche ACI/ANIA
    # Simuliamo un indice di rischio
    # Più alto = maggior rischio
    df_enriched['VEHICLE_RISK_INDEX'] = (
        df_enriched['VEHICLE_POWER_HP'] / 100 +  # Veicoli potenti = più rischio
        df_enriched['PROVINCE_ACCIDENT_RATE'] / 10 +  # Zone con più incidenti = più rischio
        np.random.normal(0, 0.5, size=len(df_enriched))  # Componente casuale
    ).clip(1, 10)  # Normalizza tra 1 e 10
    
    # 3. FEATURE DERIVATE
    
    # 3.1 Stagionalità e festività
    df_enriched['IS_SUMMER'] = df_enriched['CLAIM_DATE'].dt.month.isin([6, 7, 8]).astype(int)
    df_enriched['IS_WINTER'] = df_enriched['CLAIM_DATE'].dt.month.isin([12, 1, 2]).astype(int)
    
    # 3.2 Rischio relativo cliente
    df_enriched['CLAIM_FREQUENCY_RATIO'] = (
        df_enriched['PREVIOUS_CLAIMS_COUNT'] / df_enriched['CUSTOMER_TENURE_YEARS'].clip(1)  # Evita divisione per zero
    )
    
    # 3.3 Indice combinato rischio cliente-veicolo (personalizzato)
    df_enriched['COMBINED_RISK_INDEX'] = (
        0.3 * df_enriched['CLAIM_FREQUENCY_RATIO'].clip(0, 5) / 5 +  # 30% basato su storico sinistri
        0.4 * df_enriched['VEHICLE_RISK_INDEX'] / 10 +  # 40% basato su rischio veicolo
        0.2 * df_enriched['PROVINCE_ACCIDENT_RATE'] / 10 +  # 20% basato su zona geografica
        0.1 * (100 - df_enriched['POLICYHOLDER_AGE'].clip(18, 90)) / 82  # 10% basato su età (inverso)
    ).clip(0, 1)
    
    # 3.4 Indicatore di valore cliente (Customer Lifetime Value)
    df_enriched['CUSTOMER_VALUE_INDEX'] = (
        df_enriched['CUSTOMER_ANNUAL_PREMIUM'] * 
        df_enriched['CUSTOMER_TENURE_YEARS'].clip(1) / 10 *  # Anni cliente
        (1 - df_enriched['COMBINED_RISK_INDEX'] * 0.5)  # Penalizza per rischio
    ) / 1000  # Scala per leggibilità
    
    # 3.5 Categorizzazione cliente
    df_enriched['CUSTOMER_SEGMENT'] = pd.qcut(
        df_enriched['CUSTOMER_VALUE_INDEX'],
        q=4,
        labels=['Bronze', 'Silver', 'Gold', 'Platinum']
    )
    
    return df_enriched

def visualize_enriched_data(df_enriched):
    """
    Visualizza le informazioni derivate dall'arricchimento dei dati.
    
    Args:
        df_enriched: DataFrame arricchito
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Distribuzione dell'indice di rischio combinato
    sns.histplot(df_enriched['COMBINED_RISK_INDEX'], kde=True, ax=axes[0, 0], color='orangered')
    axes[0, 0].set_title('Distribuzione Indice di Rischio Combinato', fontsize=14)
    axes[0, 0].set_xlabel('Indice di Rischio (0-1)')
    axes[0, 0].set_ylabel('Numero di Clienti')
    
    # 2. Customer Value Index per segmento cliente
    sns.boxplot(x='CUSTOMER_SEGMENT', y='CUSTOMER_VALUE_INDEX', 
                data=df_enriched, ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_title('Valore Cliente per Segmento', fontsize=14)
    axes[0, 1].set_xlabel('Segmento Cliente')
    axes[0, 1].set_ylabel('Indice di Valore (CLV)')
    
    # 3. Relazione tra valore veicolo e importo sinistri
    sns.scatterplot(x='VEHICLE_ESTIMATED_VALUE', y='CLAIM_AMOUNT_PAID', 
                   hue='WARRANTY', data=df_enriched.sample(1000), ax=axes[1, 0], alpha=0.6)
    axes[1, 0].set_title('Relazione tra Valore Veicolo e Importo Sinistri', fontsize=14)
    axes[1, 0].set_xlabel('Valore Stimato Veicolo (€)')
    axes[1, 0].set_ylabel('Importo Sinistro (€)')
    axes[1, 0].legend(loc='upper right', fontsize=8)
    
    # 4. Importo medio sinistri per canale di acquisizione e condizioni meteo
    weather_channel = df_enriched.groupby(['ACQUISITION_CHANNEL', 'WEATHER_CONDITION'])['CLAIM_AMOUNT_PAID'].mean().unstack()
    weather_channel.plot(kind='bar', ax=axes[1, 1], colormap='tab10')
    axes[1, 1].set_title('Importo Medio Sinistri per Canale e Condizioni Meteo', fontsize=14)
    axes[1, 1].set_xlabel('Canale di Acquisizione')
    axes[1, 1].set_ylabel('Importo Medio Sinistri (€)')
    axes[1, 1].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig

df_processed = pd.read_excel('dataset_processed.xlsx', parse_dates=['CLAIM_DATE'])
df_enriched = enrich_insurance_dataset(df_processed)
#save enrriched dataset to excel
df_enriched.to_excel('dataset_enriched.xlsx', index=False)

visualize_enriched_data(df_enriched)