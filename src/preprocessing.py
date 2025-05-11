import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Caricamento dati (simulato)
# df = pd.read_csv('dataset_assicurativo.csv')

# 1. GESTIONE OUTLIER E VALORI ANOMALI

def preprocess_insurance_dataset(df):
    """
    Pipeline completa di preprocessing per il dataset assicurativo.
    """
    print("Dimensioni del dataset originale:", df.shape)
    
    # Crea una copia del dataframe
    df_clean = df.copy()
    
    # 1.1 Correzione età anomale
    # Filtra le età negative o improbabili (>100 anni)
    age_mask = (df_clean['POLICYHOLDER_AGE'] < 0) | (df_clean['POLICYHOLDER_AGE'] > 100)
    print(f"Età anomale rilevate: {df_clean[age_mask].shape[0]} ({df_clean[age_mask].shape[0]/df_clean.shape[0]:.4%})")
    
    # Sostituisci con la mediana dell'età
    median_age = df_clean['POLICYHOLDER_AGE'].median()
    df_clean.loc[age_mask, 'POLICYHOLDER_AGE'] = median_age
    
    # 1.2 Rimozione outlier CLAIM_AMOUNT_PAID con metodo IQR
    Q1 = df_clean['CLAIM_AMOUNT_PAID'].quantile(0.25)
    Q3 = df_clean['CLAIM_AMOUNT_PAID'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identifica gli outlier
    outliers_mask = (df_clean['CLAIM_AMOUNT_PAID'] < lower_bound) | (df_clean['CLAIM_AMOUNT_PAID'] > upper_bound)
    print(f"Outlier importo sinistri: {df_clean[outliers_mask].shape[0]} ({df_clean[outliers_mask].shape[0]/df_clean.shape[0]:.4%})")
    
    # Non rimuoviamo gli outlier ma li sostituiamo con valori limite per preservare la distribuzione
    df_clean.loc[df_clean['CLAIM_AMOUNT_PAID'] > upper_bound, 'CLAIM_AMOUNT_PAID'] = upper_bound
    df_clean.loc[df_clean['CLAIM_AMOUNT_PAID'] < lower_bound, 'CLAIM_AMOUNT_PAID'] = lower_bound
    
    # 1.3 Verifica date anomale
    # Rileva date future (consideriamo la data attuale come 2025-05-10)
    current_date = pd.to_datetime('2025-05-10')
    future_dates_mask = df_clean['CLAIM_DATE'] > current_date
    print(f"Date future rilevate: {df_clean[future_dates_mask].shape[0]} ({df_clean[future_dates_mask].shape[0]/df_clean.shape[0]:.4%})")
    
    # Tratta le date future (sostituisci con la mediana delle date)
    if sum(future_dates_mask) > 0:
        median_date = df_clean['CLAIM_DATE'].median()
        df_clean.loc[future_dates_mask, 'CLAIM_DATE'] = median_date
    
    # 2. GESTIONE VALORI NULLI
    
    # 2.1 POLICYHOLDER_GENDER (6 valori nulli)
    # Data la bassa percentuale, imputiamo con il valore più frequente
    most_frequent_gender = df_clean['POLICYHOLDER_GENDER'].mode()[0]
    df_clean['POLICYHOLDER_GENDER'].fillna(most_frequent_gender, inplace=True)
    
    # 2.2 CLAIM_REGION (1289 valori nulli) - Circa 0.54%
    # Per le regioni, utilizziamo l'imputazione basata sulla provincia se disponibile
    for idx in df_clean[df_clean['CLAIM_REGION'].isna()].index:
        province = df_clean.loc[idx, 'CLAIM_PROVINCE']
        if pd.notna(province):
            # Trova la regione più comune per quella provincia
            common_region = df_clean[df_clean['CLAIM_PROVINCE'] == province]['CLAIM_REGION'].mode()
            if not common_region.empty:
                df_clean.loc[idx, 'CLAIM_REGION'] = common_region[0]
    
    # Per i restanti valori nulli, utilizziamo la moda
    df_clean['CLAIM_REGION'].fillna(df_clean['CLAIM_REGION'].mode()[0], inplace=True)
    
    # 2.3 CLAIM_PROVINCE (7610 valori nulli) - Circa 3.2%
    # Imputiamo con la provincia più frequente della regione corrispondente
    for idx in df_clean[df_clean['CLAIM_PROVINCE'].isna()].index:
        region = df_clean.loc[idx, 'CLAIM_REGION']
        if pd.notna(region):
            # Trova la provincia più comune per quella regione
            common_province = df_clean[df_clean['CLAIM_REGION'] == region]['CLAIM_PROVINCE'].mode()
            if not common_province.empty:
                df_clean.loc[idx, 'CLAIM_PROVINCE'] = common_province[0]
    
    # Per i rimanenti, utilizziamo la moda generale
    df_clean['CLAIM_PROVINCE'].fillna(df_clean['CLAIM_PROVINCE'].mode()[0], inplace=True)
    
    # 2.4 VEHICLE_BRAND e VEHICLE_MODEL (471 e 412 valori nulli rispettivamente)
    # Strategia: se uno dei due è presente, imputiamo il valore mancante in base alle associazioni
    
    # Completa VEHICLE_BRAND se VEHICLE_MODEL è presente
    for idx in df_clean[df_clean['VEHICLE_BRAND'].isna() & df_clean['VEHICLE_MODEL'].notna()].index:
        model = df_clean.loc[idx, 'VEHICLE_MODEL']
        # Trova la marca più comune per quel modello
        common_brand = df_clean[df_clean['VEHICLE_MODEL'] == model]['VEHICLE_BRAND'].mode()
        if not common_brand.empty:
            df_clean.loc[idx, 'VEHICLE_BRAND'] = common_brand[0]
    
    # Completa VEHICLE_MODEL se VEHICLE_BRAND è presente
    for idx in df_clean[df_clean['VEHICLE_MODEL'].isna() & df_clean['VEHICLE_BRAND'].notna()].index:
        brand = df_clean.loc[idx, 'VEHICLE_BRAND']
        # Trova il modello più comune per quella marca
        common_model = df_clean[df_clean['VEHICLE_BRAND'] == brand]['VEHICLE_MODEL'].mode()
        if not common_model.empty:
            df_clean.loc[idx, 'VEHICLE_MODEL'] = common_model[0]
    
    # Per i rimanenti valori nulli, utilizziamo la moda
    df_clean['VEHICLE_BRAND'].fillna(df_clean['VEHICLE_BRAND'].mode()[0], inplace=True)
    df_clean['VEHICLE_MODEL'].fillna(df_clean['VEHICLE_MODEL'].mode()[0], inplace=True)
    
    # 3. FEATURE ENGINEERING
    
    # 3.1 Estrazione caratteristiche temporali
    df_clean['CLAIM_YEAR'] = df_clean['CLAIM_DATE'].dt.year
    df_clean['CLAIM_MONTH'] = df_clean['CLAIM_DATE'].dt.month
    df_clean['CLAIM_DAY'] = df_clean['CLAIM_DATE'].dt.day
    df_clean['CLAIM_WEEKDAY'] = df_clean['CLAIM_DATE'].dt.dayofweek
    df_clean['CLAIM_QUARTER'] = df_clean['CLAIM_DATE'].dt.quarter
    df_clean['IS_WEEKEND'] = df_clean['CLAIM_WEEKDAY'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 3.2 Calcolo loss ratio (rapporto tra sinistri pagati e premi incassati)
    df_clean['LOSS_RATIO'] = df_clean['CLAIM_AMOUNT_PAID'] / df_clean['PREMIUM_AMOUNT_PAID']
    
    # 3.3 Categorizzazione dell'età
    df_clean['AGE_GROUP'] = pd.cut(df_clean['POLICYHOLDER_AGE'], 
                                  bins=[0, 25, 35, 45, 55, 65, 75, 100],
                                  labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76+'])
    
    # 3.4 Categorizzazione importo sinistri
    df_clean['CLAIM_CATEGORY'] = pd.qcut(df_clean['CLAIM_AMOUNT_PAID'], 
                                        q=5, 
                                        labels=['Molto basso', 'Basso', 'Medio', 'Alto', 'Molto alto'])
    
    # 3.5 Flag di claim di alto valore
    df_clean['HIGH_VALUE_CLAIM'] = (df_clean['CLAIM_AMOUNT_PAID'] > df_clean['CLAIM_AMOUNT_PAID'].quantile(0.75)).astype(int)
    
    # 4. STANDARDIZZAZIONE/NORMALIZZAZIONE VARIABILI NUMERICHE
    numeric_features = ['POLICYHOLDER_AGE', 'CLAIM_AMOUNT_PAID', 'PREMIUM_AMOUNT_PAID', 'LOSS_RATIO']
    
    # Per dimostrare il processo, creiamo colonne standardizzate
    scaler = StandardScaler()
    df_clean[['AGE_SCALED', 'CLAIM_AMOUNT_SCALED', 'PREMIUM_SCALED', 'LOSS_RATIO_SCALED']] = scaler.fit_transform(df_clean[numeric_features])
    
    # 5. VERIFICA FINALE DEL DATASET
    print("\nVerifica finale dataset processato:")
    print("Dimensioni:", df_clean.shape)
    print("Valori nulli rimasti:", df_clean.isnull().sum().sum())
    
    return df_clean

# Funzione di esempio per visualizzare le distribuzioni prima e dopo il preprocessing
def plot_distributions_comparison(df_original, df_processed):
    """
    Visualizza le distribuzioni delle variabili chiave prima e dopo il preprocessing.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Distribuzione età
    sns.histplot(df_original['POLICYHOLDER_AGE'], kde=True, ax=axes[0, 0], color='blue', alpha=0.5)
    sns.histplot(df_processed['POLICYHOLDER_AGE'], kde=True, ax=axes[0, 0], color='red', alpha=0.5)
    axes[0, 0].set_title('Distribuzione Età (Blu: Originale, Rosso: Processato)')
    
    # Distribuzione importo sinistri
    sns.histplot(df_original['CLAIM_AMOUNT_PAID'], kde=True, ax=axes[0, 1], color='blue', alpha=0.5)
    sns.histplot(df_processed['CLAIM_AMOUNT_PAID'], kde=True, ax=axes[0, 1], color='red', alpha=0.5)
    axes[0, 1].set_title('Distribuzione Importo Sinistri (Blu: Originale, Rosso: Processato)')
    
    # Loss Ratio
    sns.histplot(df_processed['LOSS_RATIO'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Distribuzione Loss Ratio (Processato)')
    
    # Distribuzione per tipo di garanzia
    warranty_counts = df_processed['WARRANTY'].value_counts().head(10)
    sns.barplot(x=warranty_counts.values, y=warranty_counts.index, ax=axes[1, 1])
    axes[1, 1].set_title('Top 10 Tipi di Garanzia')
    
    plt.tight_layout()
    return fig

# Esempio di utilizzo completo
df = pd.read_excel('dataset.xlsx')
df_processed = preprocess_insurance_dataset(df)
# write df_processed to excel
df_processed.to_excel('dataset_processed.xlsx', index=False)
#plot_distributions_comparison(df, df_processed)