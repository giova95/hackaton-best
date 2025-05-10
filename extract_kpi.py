import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def extract_key_kpis(df):
    """
    Estrae i KPI chiave dal dataset assicurativo.
    
    Args:
        df: Il DataFrame pulito e preprocessato
        
    Returns:
        dict: Un dizionario contenente i KPI calcolati
    """
    kpis = {}
    
    # 1. KPI FINANZIARI
    
    # 1.1 Loss Ratio complessivo (rapporto sinistri/premi)
    total_claims = df['CLAIM_AMOUNT_PAID'].sum()
    total_premiums = df['PREMIUM_AMOUNT_PAID'].sum()
    loss_ratio = total_claims / total_premiums
    kpis['overall_loss_ratio'] = loss_ratio
    
    # 1.2 Loss Ratio per tipo di garanzia
    warranty_loss_ratio = df.groupby('WARRANTY').apply(
        lambda x: x['CLAIM_AMOUNT_PAID'].sum() / x['PREMIUM_AMOUNT_PAID'].sum()
    ).sort_values(ascending=False)
    kpis['warranty_loss_ratio'] = warranty_loss_ratio
    
    # 1.3 Costo medio dei sinistri
    avg_claim_amount = df['CLAIM_AMOUNT_PAID'].mean()
    kpis['average_claim_amount'] = avg_claim_amount
    
    # 1.4 Costo medio dei sinistri per regione
    regional_avg_claim = df.groupby('CLAIM_REGION')['CLAIM_AMOUNT_PAID'].mean().sort_values(ascending=False)
    kpis['regional_avg_claim'] = regional_avg_claim
    
    # 1.5 Premio medio
    avg_premium = df['PREMIUM_AMOUNT_PAID'].mean()
    kpis['average_premium'] = avg_premium
    
    # 2. KPI DEMOGRAFICI
    
    # 2.1 Distribuzione sinistri per fascia d'età
    if 'AGE_GROUP' not in df.columns:
        df['AGE_GROUP'] = pd.cut(df['POLICYHOLDER_AGE'], 
                              bins=[0, 25, 35, 45, 55, 65, 75, 100],
                              labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76+'])
    
    age_group_claims = df.groupby('AGE_GROUP')['CLAIM_AMOUNT_PAID'].agg(['count', 'mean', 'sum'])
    age_group_claims['frequency'] = age_group_claims['count'] / age_group_claims['count'].sum()
    age_group_claims['severity'] = age_group_claims['mean']
    kpis['age_group_claims'] = age_group_claims
    
    # 2.2 Distribuzione sinistri per genere
    gender_claims = df.groupby('POLICYHOLDER_GENDER')['CLAIM_AMOUNT_PAID'].agg(['count', 'mean', 'sum'])
    gender_claims['frequency'] = gender_claims['count'] / gender_claims['count'].sum()
    gender_claims['severity'] = gender_claims['mean']
    kpis['gender_claims'] = gender_claims
    
    # 3. KPI TEMPORALI
    
    # 3.1 Trend temporale (mensile) dei sinistri
    df['YEAR_MONTH'] = df['CLAIM_DATE'].dt.strftime('%Y-%m')
    monthly_claims = df.groupby('YEAR_MONTH').agg(
        claim_count=('CLAIM_ID', 'count'),
        claim_amount=('CLAIM_AMOUNT_PAID', 'sum'),
        avg_claim=('CLAIM_AMOUNT_PAID', 'mean')
    ).sort_index()
    kpis['monthly_claims_trend'] = monthly_claims
    
    # 3.2 Stagionalità dei sinistri (per mese)
    df['MONTH'] = df['CLAIM_DATE'].dt.month
    monthly_seasonality = df.groupby('MONTH').agg(
        claim_count=('CLAIM_ID', 'count'),
        claim_amount=('CLAIM_AMOUNT_PAID', 'sum'),
        avg_claim=('CLAIM_AMOUNT_PAID', 'mean')
    ).sort_index()
    kpis['monthly_seasonality'] = monthly_seasonality
    
    # 3.3 Distribuzione sinistri per giorno della settimana
    df['WEEKDAY'] = df['CLAIM_DATE'].dt.dayofweek
    weekday_claims = df.groupby('WEEKDAY').agg(
        claim_count=('CLAIM_ID', 'count'),
        claim_amount=('CLAIM_AMOUNT_PAID', 'sum'),
        avg_claim=('CLAIM_AMOUNT_PAID', 'mean')
    ).sort_index()
    kpis['weekday_claims'] = weekday_claims
    
    # 4. KPI PRODOTTO
    
    # 4.1 Distribuzione sinistri per tipo di garanzia
    warranty_claims = df.groupby('WARRANTY').agg(
        claim_count=('CLAIM_ID', 'count'),
        claim_amount=('CLAIM_AMOUNT_PAID', 'sum'),
        avg_claim=('CLAIM_AMOUNT_PAID', 'mean'),
        total_premium=('PREMIUM_AMOUNT_PAID', 'sum')
    ).sort_values('claim_count', ascending=False)
    warranty_claims['frequency'] = warranty_claims['claim_count'] / warranty_claims['claim_count'].sum()
    warranty_claims['loss_ratio'] = warranty_claims['claim_amount'] / warranty_claims['total_premium']
    kpis['warranty_claims'] = warranty_claims
    
    # 5. KPI VEICOLI
    
    # 5.1 Top 10 marche per numero di sinistri
    brand_claims = df.groupby('VEHICLE_BRAND').agg(
        claim_count=('CLAIM_ID', 'count'),
        claim_amount=('CLAIM_AMOUNT_PAID', 'sum'),
        avg_claim=('CLAIM_AMOUNT_PAID', 'mean')
    ).sort_values('claim_count', ascending=False).head(10)
    kpis['top_brand_claims'] = brand_claims
    
    # 5.2 Top 10 modelli per numero di sinistri
    model_claims = df.groupby('VEHICLE_MODEL').agg(
        claim_count=('CLAIM_ID', 'count'),
        claim_amount=('CLAIM_AMOUNT_PAID', 'sum'),
        avg_claim=('CLAIM_AMOUNT_PAID', 'mean')
    ).sort_values('claim_count', ascending=False).head(10)
    kpis['top_model_claims'] = model_claims
    
    # 5.3 Loss ratio per marca di veicolo
    brand_loss_ratio = df.groupby('VEHICLE_BRAND').apply(
        lambda x: x['CLAIM_AMOUNT_PAID'].sum() / x['PREMIUM_AMOUNT_PAID'].sum()
    ).sort_values(ascending=False)
    kpis['brand_loss_ratio'] = brand_loss_ratio
    
    # 6. KPI GEOGRAFICI
    
    # 6.1 Distribuzione sinistri per regione
    region_claims = df.groupby('CLAIM_REGION').agg(
        claim_count=('CLAIM_ID', 'count'),
        claim_amount=('CLAIM_AMOUNT_PAID', 'sum'),
        avg_claim=('CLAIM_AMOUNT_PAID', 'mean')
    ).sort_values('claim_count', ascending=False)
    region_claims['frequency'] = region_claims['claim_count'] / region_claims['claim_count'].sum()
    kpis['region_claims'] = region_claims
    
    # 6.2 Top 10 province per numero di sinistri
    province_claims = df.groupby('CLAIM_PROVINCE').agg(
        claim_count=('CLAIM_ID', 'count'),
        claim_amount=('CLAIM_AMOUNT_PAID', 'sum'),
        avg_claim=('CLAIM_AMOUNT_PAID', 'mean')
    ).sort_values('claim_count', ascending=False).head(10)
    kpis['top_province_claims'] = province_claims
    
    # 6.3 Loss ratio per regione
    region_loss_ratio = df.groupby('CLAIM_REGION').apply(
        lambda x: x['CLAIM_AMOUNT_PAID'].sum() / x['PREMIUM_AMOUNT_PAID'].sum()
    ).sort_values(ascending=False)
    kpis['region_loss_ratio'] = region_loss_ratio
    
    # 7. KPI DI RISCHIO E COSTO

    # 7.1 Distribuzione della frequenza dei sinistri 
    # (quanti assicurati hanno 1, 2, 3... sinistri)
    claim_frequency = df.groupby('POLICYHOLDER_AGE').size().value_counts().sort_index()
    kpis['claim_frequency_distribution'] = claim_frequency
    
    # 7.2 Segmentazione sinistri per importo
    df['CLAIM_SEGMENT'] = pd.qcut(df['CLAIM_AMOUNT_PAID'], 4, 
                                 labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    claim_segments = df.groupby('CLAIM_SEGMENT').agg(
        claim_count=('CLAIM_ID', 'count'),
        claim_amount=('CLAIM_AMOUNT_PAID', 'sum'),
        avg_claim=('CLAIM_AMOUNT_PAID', 'mean')
    )
    kpis['claim_segments'] = claim_segments
    
    # 7.3 Rapporto tra importo premio e importo sinistro
    df['PREMIUM_TO_CLAIM_RATIO'] = df['PREMIUM_AMOUNT_PAID'] / df['CLAIM_AMOUNT_PAID']
    premium_claim_ratio_stats = df['PREMIUM_TO_CLAIM_RATIO'].describe()
    kpis['premium_claim_ratio_stats'] = premium_claim_ratio_stats
    
    # 8. INDICI COMBINATI
    
    # 8.1 Matrice età-genere per importo medio sinistri
    age_gender_matrix = df.pivot_table(
        values='CLAIM_AMOUNT_PAID', 
        index='AGE_GROUP',
        columns='POLICYHOLDER_GENDER',
        aggfunc='mean'
    )
    kpis['age_gender_avg_claim'] = age_gender_matrix
    
    # 8.2 Matrice regione-tipo garanzia per frequenza sinistri
    region_warranty_matrix = df.pivot_table(
        values='CLAIM_ID', 
        index='CLAIM_REGION',
        columns='WARRANTY',
        aggfunc='count',
        fill_value=0
    )
    # Normalizza per ottenere percentuali
    region_warranty_pct = region_warranty_matrix.div(region_warranty_matrix.sum(axis=1), axis=0)
    kpis['region_warranty_frequency'] = region_warranty_pct
    
    # 8.3 Analisi sinistri per fascia d'età e tipo di garanzia
    age_warranty_matrix = df.pivot_table(
        values='CLAIM_AMOUNT_PAID',
        index='AGE_GROUP',
        columns='WARRANTY',
        aggfunc='mean',
        fill_value=0
    )
    kpis['age_warranty_avg_claim'] = age_warranty_matrix

    # 8.4 Top combinazioni di marca-modello per numero di sinistri
    df['BRAND_MODEL'] = df['VEHICLE_BRAND'] + ' - ' + df['VEHICLE_MODEL']
    brand_model_claims = df.groupby('BRAND_MODEL').agg(
        claim_count=('CLAIM_ID', 'count'),
        claim_amount=('CLAIM_AMOUNT_PAID', 'sum'),
        avg_claim=('CLAIM_AMOUNT_PAID', 'mean')
    ).sort_values('claim_count', ascending=False).head(15)
    kpis['top_brand_model_claims'] = brand_model_claims
    
    # 9. KPI DI TENDENZA TEMPORALE
    
    # 9.1 Crescita anno su anno dei sinistri
    df['YEAR'] = df['CLAIM_DATE'].dt.year
    yearly_claims = df.groupby('YEAR').agg(
        claim_count=('CLAIM_ID', 'count'),
        claim_amount=('CLAIM_AMOUNT_PAID', 'sum'),
        avg_claim=('CLAIM_AMOUNT_PAID', 'mean')
    )
    # Calcola la crescita percentuale
    yearly_claims['count_growth'] = yearly_claims['claim_count'].pct_change() * 100
    yearly_claims['amount_growth'] = yearly_claims['claim_amount'].pct_change() * 100
    kpis['yearly_claims_growth'] = yearly_claims
    
    # 9.2 Stagionalità trimestrale
    df['QUARTER'] = df['CLAIM_DATE'].dt.quarter
    quarterly_claims = df.groupby(['YEAR', 'QUARTER']).agg(
        claim_count=('CLAIM_ID', 'count'),
        claim_amount=('CLAIM_AMOUNT_PAID', 'sum'),
        avg_claim=('CLAIM_AMOUNT_PAID', 'mean')
    )
    kpis['quarterly_claims'] = quarterly_claims
    
    return kpis

# Funzione per generare report dei KPI principali
def generate_kpi_report(kpis):
    """
    Genera un report testuale dei KPI principali.
    
    Args:
        kpis: Il dizionario contenente i KPI calcolati
        
    Returns:
        str: Report testuale formattato
    """
    report = "=============================================\n"
    report += "REPORT KPI ASSICURATIVI\n"
    report += "=============================================\n\n"
    
    # KPI finanziari
    report += "1. KPI FINANZIARI\n"
    report += "-----------------\n"
    report += f"Loss Ratio complessivo: {kpis['overall_loss_ratio']:.4f}\n"
    report += f"Costo medio sinistri: {kpis['average_claim_amount']:.2f} €\n"
    report += f"Premio medio: {kpis['average_premium']:.2f} €\n\n"
    
    # Top 5 tipi di garanzia per loss ratio
    report += "Top 5 tipi di garanzia per Loss Ratio:\n"
    for warranty, lr in kpis['warranty_loss_ratio'].head(5).items():
        report += f"- {warranty}: {lr:.4f}\n"
    report += "\n"
    
    # KPI demografici
    report += "2. KPI DEMOGRAFICI\n"
    report += "------------------\n"
    report += "Sinistri per fascia d'età (importo medio):\n"
    for age_group, row in kpis['age_group_claims'].iterrows():
        report += f"- {age_group}: {row['mean']:.2f} € (freq: {row['frequency']*100:.1f}%)\n"
    report += "\n"
    
    # KPI temporali
    report += "3. KPI TEMPORALI\n"
    report += "---------------\n"
    report += "Stagionalità mensile (top 3 mesi):\n"
    month_names = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno', 'Luglio', 'Agosto', 'Settembre', 'Ottobre', 'Novembre', 'Dicembre']
    top_months = kpis['monthly_seasonality'].sort_values('claim_count', ascending=False).head(3)
    for i, (idx, row) in enumerate(top_months.iterrows()):
        report += f"- {month_names[idx-1]}: {row['claim_count']} sinistri (media: {row['avg_claim']:.2f} €)\n"
    report += "\n"
    
    # KPI prodotto
    report += "4. KPI PRODOTTO\n"
    report += "--------------\n"
    report += "Top 5 tipi di garanzia per numero di sinistri:\n"
    top_warranty = kpis['warranty_claims'].head(5)
    for warranty, row in top_warranty.iterrows():
        report += f"- {warranty}: {row['claim_count']} sinistri ({row['frequency']*100:.1f}%)\n"
    report += "\n"
    
    # KPI veicoli
    report += "5. KPI VEICOLI\n"
    report += "-------------\n"
    report += "Top 5 marche per numero di sinistri:\n"
    for brand, row in kpis['top_brand_claims'].head(5).iterrows():
        report += f"- {brand}: {row['claim_count']} sinistri (media: {row['avg_claim']:.2f} €)\n"
    report += "\n"
    
    # KPI geografici
    report += "6. KPI GEOGRAFICI\n"
    report += "----------------\n"
    report += "Top 5 regioni per frequenza sinistri:\n"
    for region, row in kpis['region_claims'].head(5).iterrows():
        report += f"- {region}: {row['claim_count']} sinistri ({row['frequency']*100:.1f}%)\n"
    report += "\n"
    
    # Segmentazione sinistri
    report += "7. SEGMENTAZIONE SINISTRI\n"
    report += "------------------------\n"
    report += "Distribuzione sinistri per segmento di importo:\n"
    for segment, row in kpis['claim_segments'].iterrows():
        report += f"- {segment}: {row['claim_count']} sinistri (media: {row['avg_claim']:.2f} €)\n"
    report += "\n"
    
    # Tendenze annuali
    report += "8. TENDENZE ANNUALI\n"
    report += "------------------\n"
    report += "Crescita anno su anno del numero di sinistri:\n"
    for year, row in kpis['yearly_claims_growth'].iterrows():
        if pd.notna(row['count_growth']):
            report += f"- {year}: {row['count_growth']:.1f}% ({row['claim_count']} sinistri)\n"
    report += "\n"
    
    return report


# use the function to extract kpis
df = pd.read_excel('dataset_processed.xlsx')
kpis = extract_key_kpis(df)
# generate the report
report = generate_kpi_report(kpis)
print(report)
# save the report to a text file
with open('kpi_report.txt', 'w') as f:
    f.write(report)
# save the kpis to a pickle file
import pickle
with open('kpis.pkl', 'wb') as f:
    pickle.dump(kpis, f)
# save the kpis to an excel file
kpis_df = pd.DataFrame(kpis)
kpis_df.to_excel('kpis.xlsx', index=False)

