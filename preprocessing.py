import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Impostazioni generali
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)

# 1. Caricamento dati
df = pd.read_excel("2025 - BEST Hackathon - dataset.xlsx")

# 2. Informazioni base
print(">>> Informazioni sul dataset:")
df.info()

print("\n>>> Statistiche descrittive (numeriche e categoriche):")
print(df.describe(include='all'))

print("\n>>> Prime righe:")
print(df.head())

# 3. Valori mancanti
print("\n>>> Valori nulli per colonna:")
print(df.isnull().sum())

# 4. Valori unici per capire variabilità e possibili errori di digitazione
print("\n>>> Valori unici per ogni colonna:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} valori unici")

# 5. Distribuzioni e anomalie

## Età
plt.figure(figsize=(10, 5))
df[df['POLICYHOLDER_AGE'].between(0, 200)]['POLICYHOLDER_AGE'].hist(
    bins=30, color='skyblue', edgecolor='black'
)
plt.title("Distribuzione Età Assicurati (solo età 10-100)")
plt.xlabel("Età")
plt.ylabel("Frequenza")
plt.grid(True)
plt.show()


## Genere
plt.figure(figsize=(5,4))
df['POLICYHOLDER_GENDER'].value_counts(dropna=False).plot(kind='bar', color='orange')
plt.title("Distribuzione genere")
plt.ylabel("Conteggio")
plt.show()

## Importi pagati (claim e premio)
for col in ['CLAIM_AMOUNT_PAID', 'PREMIUM_AMOUNT_PAID']:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[col])
    plt.title(f"Distribuzione {col}")
    plt.xlabel("Importo")
    plt.show()

# 6. Variabili temporali

## Parsing date e verifica anomalie
df['CLAIM_DATE'] = pd.to_datetime(df['CLAIM_DATE'], errors='coerce')
print("\n>>> Intervallo date:")
print(f"{df['CLAIM_DATE'].min()} --> {df['CLAIM_DATE'].max()}")

## Distribuzione per anno/mese
plt.figure(figsize=(10,4))
df['CLAIM_DATE'].dt.to_period('M').value_counts().sort_index().plot(kind='bar')
plt.title("Numero di claim per mese")
plt.xlabel("Mese")
plt.ylabel("Numero di claim")
plt.show()

# 7. Variabili geografiche

## Regione
plt.figure(figsize=(10,4))
df['CLAIM_REGION'].value_counts().plot(kind='bar', color='green')
plt.title("Distribuzione claim per regione")
plt.ylabel("Numero di claim")
plt.show()

## Provincia
top_provinces = df['CLAIM_PROVINCE'].value_counts().head(10)
plt.figure(figsize=(10,4))
top_provinces.plot(kind='bar', color='teal')
plt.title("Top 10 province per numero di claim")
plt.ylabel("Numero di claim")
plt.show()

# 8. Veicoli

## Brand
plt.figure(figsize=(10,4))
df['VEHICLE_BRAND'].value_counts().head(10).plot(kind='bar', color='purple')
plt.title("Top 10 marche di veicoli")
plt.ylabel("Numero di claim")
plt.show()

## Model (solo i più frequenti)
top_models = df['VEHICLE_MODEL'].value_counts().head(10)
plt.figure(figsize=(10,4))
top_models.plot(kind='bar', color='violet')
plt.title("Top 10 modelli di veicoli")
plt.ylabel("Numero di claim")
plt.show()

# 9. Correlazioni numeriche
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice di correlazione")
plt.show()

# 10. Relazioni interessanti

## Età vs importo claim
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='POLICYHOLDER_AGE', y='CLAIM_AMOUNT_PAID', hue='POLICYHOLDER_GENDER')
plt.title("Età vs importo del claim")
plt.xlabel("Età")
plt.ylabel("Importo claim")
plt.show()

## Premium vs Claim
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PREMIUM_AMOUNT_PAID', y='CLAIM_AMOUNT_PAID')
plt.title("Importo premio vs importo claim")
plt.xlabel("Premio pagato")
plt.ylabel("Claim pagato")
plt.show()


import matplotlib.pyplot as plt

# Definizione del range realistico
MIN_ETA = 16
MAX_ETA = 100

# Conteggi
totali = len(df)
eta_valide = df[(df['POLICYHOLDER_AGE'] >= MIN_ETA) & (df['POLICYHOLDER_AGE'] <= MAX_ETA)]
eta_anomale = df[(df['POLICYHOLDER_AGE'] < MIN_ETA) | (df['POLICYHOLDER_AGE'] > MAX_ETA)]

# Percentuali
pct_valide = len(eta_valide) / totali * 100
pct_anomale = len(eta_anomale) / totali * 100

# Plot distribuzione
plt.figure(figsize=(10, 5))
df['POLICYHOLDER_AGE'].hist(bins=30, color='skyblue', edgecolor='black')
plt.axvspan(0, MIN_ETA, color='red', alpha=0.2, label=f"Età < {MIN_ETA}")
plt.axvspan(MAX_ETA, df['POLICYHOLDER_AGE'].max(), color='red', alpha=0.2, label=f"Età > {MAX_ETA}")
plt.title("Distribuzione Età Assicurati con Evidenza Anomalie")
plt.xlabel("Età")
plt.ylabel("Frequenza")
plt.legend()
plt.grid(True)
plt.show()

# Output numerico
print(f"Totale record: {totali}")
print(f"Età valide: {len(eta_valide)} ({pct_valide:.2f}%)")
print(f"Età sospette: {len(eta_anomale)} ({pct_anomale:.2f}%)")


columns_to_check = ['POLICYHOLDER_AGE', 'WARRANTY', 'VEHICLE_BRAND', 'VEHICLE_MODEL', 'CLAIM_AMOUNT_PAID', 'PREMIUM_AMOUNT_PAID', 'CLAIM_DATE']

for col in columns_to_check:
    print(f"\nValori unici per {col}:")
    print(df[col].value_counts(dropna=False))

