import pandas as pd
import numpy as np
from datetime import datetime

def extract_key_kpis(df):
    kpis = {}
    
    # 1. Loss Ratio per tipo di garanzia
    warranty_loss_ratio = df.groupby('WARRANTY').apply(
        lambda x: x['CLAIM_AMOUNT_PAID'].sum() / x['PREMIUM_AMOUNT_PAID'].sum()
    ).sort_values(ascending=False)
    kpis['warranty_loss_ratio'] = warranty_loss_ratio
    
    # 2. Importo medio sinistri per fascia d'età
    if 'AGE_GROUP' not in df.columns:
        df['AGE_GROUP'] = pd.cut(df['POLICYHOLDER_AGE'], 
                              bins=[0, 25, 35, 45, 55, 65, 100],
                              labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    age_group_claims = df.groupby('AGE_GROUP')['CLAIM_AMOUNT_PAID'].agg(['mean'])
    kpis['age_group_claims'] = age_group_claims
    
    # 3. Top 10 regioni per numero di sinistri
    region_claims = df.groupby('CLAIM_REGION').agg(
        claim_count=('CLAIM_ID', 'count')
    ).sort_values('claim_count', ascending=False)
    kpis['region_claims'] = region_claims
    
    # 4. Stagionalità dei sinistri (mensile)
    df['MONTH'] = df['CLAIM_DATE'].dt.month
    monthly_seasonality = df.groupby('MONTH').agg(
        claim_count=('CLAIM_ID', 'count')
    )
    kpis['monthly_seasonality'] = monthly_seasonality
    
    # 5. Top 10 marche veicoli per sinistri
    brand_claims = df.groupby('VEHICLE_BRAND').agg(
        claim_count=('CLAIM_ID', 'count')
    ).sort_values('claim_count', ascending=False)
    kpis['top_brand_claims'] = brand_claims
    
    # 6. Distribuzione sinistri per genere
    gender_claims = df.groupby('POLICYHOLDER_GENDER').agg(
        count=('CLAIM_ID', 'count')
    )
    kpis['gender_claims'] = gender_claims
        
    # Overall loss ratio
    total_claims = df['CLAIM_AMOUNT_PAID'].sum()
    total_premiums = df['PREMIUM_AMOUNT_PAID'].sum()
    loss_ratio = total_claims / total_premiums
    kpis['overall_loss_ratio'] = loss_ratio
    
    # Costo medio dei sinistri
    avg_claim_amount = df['CLAIM_AMOUNT_PAID'].mean()
    kpis['average_claim_amount'] = avg_claim_amount
    
    # Premio medio
    avg_premium = df['PREMIUM_AMOUNT_PAID'].mean()
    kpis['average_premium'] = avg_premium
    
    return kpis

df = pd.read_excel('dataset_processed.xlsx')
kpis = extract_key_kpis(df)