import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import time

print("Script di clustering ottimizzato avviato...")
print(f"Ora di inizio: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Carica i dati
print("\nCaricamento dataset da 'dataset_enriched.xlsx'...")
start_load = time.time()
df_enriched = pd.read_excel('dataset_enriched.xlsx')
load_time = time.time() - start_load
print(f"Dataset caricato in {load_time:.2f} secondi. Shape: {df_enriched.shape}")

# OTTIMIZZAZIONE 1: Selezione di un sottoinsieme di dati se il dataset è grande
print("\nValutazione dimensione dataset...")
sample_size = 5000  # Dimensione massima del campione
if len(df_enriched) > sample_size:
    print(f"Dataset grande rilevato ({len(df_enriched)} righe). Campionamento di {sample_size} righe per analisi iniziale...")
    df_sample = df_enriched.sample(n=sample_size, random_state=42)
else:
    print(f"Dataset di dimensioni gestibili ({len(df_enriched)} righe). Utilizzo dell'intero dataset.")
    df_sample = df_enriched.copy()

# OTTIMIZZAZIONE 2: Riduzione delle feature utilizzate
print("\nPreparazione dati con feature essenziali...")
start_time = time.time()

# Selezione di feature ridotte ma rappresentative
essential_features = [
    'POLICYHOLDER_AGE',
    'CLAIM_AMOUNT_PAID',
    'PREMIUM_AMOUNT_PAID',
    'CUSTOMER_TENURE_YEARS',
    'VEHICLE_RISK_INDEX',
    'CLAIM_FREQUENCY_RATIO',
    'COMBINED_RISK_INDEX',
    'CUSTOMER_VALUE_INDEX'
]

# Verifica quali feature sono effettivamente disponibili
available_features = [f for f in essential_features if f in df_sample.columns]
print(f"Utilizzando {len(available_features)}/{len(essential_features)} feature essenziali: {available_features}")

# Prepara i dati
df_features = df_sample[available_features].copy()

# Gestione valori nulli veloce
for col in df_features.columns:
    null_count = df_features[col].isnull().sum()
    if null_count > 0:
        print(f"  - Riempiendo {null_count} valori nulli in {col}")
        df_features[col].fillna(df_features[col].median(), inplace=True)

# Standardizzazione
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_features),
    columns=df_features.columns
)

print(f"Preparazione dati completata in {time.time() - start_time:.2f} secondi.")

# OTTIMIZZAZIONE 3: Analisi rapida del numero ottimale di cluster
print("\nRicerca rapida del numero ottimale di cluster...")
start_time = time.time()

# MODIFICA: Testare più valori per il numero di cluster
n_clusters_to_test = [2, 3, 4, 5, 6, 7, 8, 9, 10]
results = {'n_clusters': [], 'inertia': [], 'silhouette': []}

# MODIFICA: Implementazione parallela per K-means
from joblib import Parallel, delayed

def evaluate_kmeans(n, X):
    print(f"  - Testando {n} cluster...")
    # Parametri più rilassati per velocizzare
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=3, max_iter=100, tol=1e-3)
    kmeans.fit(X)
    
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, labels)
    print(f"    * Silhouette score: {silhouette:.4f}")
    return n, inertia, silhouette

# Esegui in parallelo con un numero adeguato di jobs
n_jobs = min(4, len(n_clusters_to_test))  # Limita a max 4 processi paralleli
cluster_results = Parallel(n_jobs=n_jobs)(
    delayed(evaluate_kmeans)(n, df_scaled) for n in n_clusters_to_test
)

# Elabora i risultati
for n, inertia, silhouette in cluster_results:
    results['n_clusters'].append(n)
    results['inertia'].append(inertia)
    results['silhouette'].append(silhouette)

print(f"Ricerca completata in {time.time() - start_time:.2f} secondi.")

# MODIFICA: Selezione più sofisticata del numero di cluster
# Trova il miglior silhouette score
best_silhouette_idx = np.argmax(results['silhouette'])
best_n_clusters_silhouette = results['n_clusters'][best_silhouette_idx]
best_silhouette = results['silhouette'][best_silhouette_idx]

# Trova il punto di "gomito" dell'inertia (metodo del gomito)
from kneed import KneeLocator
if len(results['n_clusters']) >= 4:  # Almeno 4 punti per trovare il gomito
    try:
        kneedle = KneeLocator(
            results['n_clusters'], 
            results['inertia'],
            curve='convex', 
            direction='decreasing'
        )
        elbow_point = kneedle.elbow
        print(f"Punto di gomito nell'inertia: {elbow_point}")
    except:
        elbow_point = None
else:
    elbow_point = None

# Scegli il numero di cluster in base a entrambi i criteri
if elbow_point and results['silhouette'][results['n_clusters'].index(elbow_point)] > 0.7 * best_silhouette:
    best_n_clusters = elbow_point
    print(f"\nNumero di cluster selezionato dal metodo del gomito: {best_n_clusters}")
    print(f"Silhouette: {results['silhouette'][results['n_clusters'].index(elbow_point)]:.4f}")
else:
    best_n_clusters = best_n_clusters_silhouette
    print(f"\nMiglior numero di cluster selezionato dal silhouette score: {best_n_clusters} (silhouette: {best_silhouette:.4f})")

# Visualizzazione rapida dei risultati
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(results['n_clusters'], results['inertia'], 'o-')
plt.title('Metodo del Gomito')
plt.xlabel('Numero di Cluster')
plt.ylabel('Inertia')
if elbow_point:
    plt.axvline(x=elbow_point, color='r', linestyle='--')

plt.subplot(1, 2, 2)
plt.plot(results['n_clusters'], results['silhouette'], 'o-')
plt.title('Silhouette Score')
plt.xlabel('Numero di Cluster')
plt.ylabel('Silhouette Score')
plt.axvline(x=best_n_clusters_silhouette, color='g', linestyle='--')
plt.tight_layout()
plt.savefig('cluster_selection.png')
print("Grafico di selezione dei cluster salvato in 'cluster_selection.png'")

# OTTIMIZZAZIONE 4: Esecuzione K-means rapida con parametri di convergenza più rilassati
print(f"\nEseguendo K-means veloce con {best_n_clusters} cluster...")
start_time = time.time()

# MODIFICA: Equilibrio tra velocità e precisione
kmeans = KMeans(
    n_clusters=best_n_clusters, 
    random_state=42, 
    n_init=5,  # Aumentato leggermente per migliore stabilità con più cluster
    max_iter=150,  # Aumentato per consentire una migliore convergenza con più cluster
    tol=1e-3,  # Tolleranza più elevata per convergenza più rapida
    algorithm='elkan'  # Generalmente più veloce per dataset di piccole/medie dimensioni
)
cluster_labels = kmeans.fit_predict(df_scaled)

# Distribuzione dei cluster
counts = np.bincount(cluster_labels)
for i, count in enumerate(counts):
    print(f"  - Cluster {i}: {count} elementi ({count/len(cluster_labels)*100:.1f}%)")

print(f"Clustering completato in {time.time() - start_time:.2f} secondi.")

# OTTIMIZZAZIONE 5: PCA semplificata
print("\nApplicazione PCA veloce per visualizzazione...")
start_time = time.time()

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)

pca_df = pd.DataFrame(
    data=principal_components,
    columns=['PC1', 'PC2']
)
pca_df['Cluster'] = cluster_labels

# Stampa la varianza spiegata
var_explained = pca.explained_variance_ratio_
print(f"  - Varianza spiegata: PC1={var_explained[0]:.2%}, PC2={var_explained[1]:.2%}")
print(f"  - Varianza totale spiegata: {sum(var_explained):.2%}")

print(f"PCA completata in {time.time() - start_time:.2f} secondi.")

# MODIFICA: Visualizzazione migliorata dei cluster
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    pca_df['PC1'], 
    pca_df['PC2'], 
    c=pca_df['Cluster'], 
    cmap='tab10', 
    alpha=0.7, 
    s=30
)
plt.title(f'Visualizzazione Cluster (PCA) - {best_n_clusters} cluster')
plt.xlabel(f'PC1 ({var_explained[0]:.2%})')
plt.ylabel(f'PC2 ({var_explained[1]:.2%})')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('cluster_visualization.png', dpi=300)
print("Visualizzazione dei cluster salvata in 'cluster_visualization.png'")

# OTTIMIZZAZIONE 6: Analisi semplificata delle caratteristiche dei cluster
print("\nCalcolo statistiche essenziali per cluster...")
start_time = time.time()

# Aggiungi le etichette al dataset originale
df_with_clusters = df_sample.copy()
df_with_clusters['CLUSTER'] = cluster_labels

# MODIFICA: Ampliato il set di metriche analizzate
key_metrics = ['POLICYHOLDER_AGE', 'COMBINED_RISK_INDEX', 'CUSTOMER_VALUE_INDEX', 
               'CLAIM_AMOUNT_PAID', 'PREMIUM_AMOUNT_PAID', 'CUSTOMER_TENURE_YEARS']
available_metrics = [m for m in key_metrics if m in df_with_clusters.columns]

# Calcola statistiche per ogni cluster
if available_metrics:
    cluster_stats = df_with_clusters.groupby('CLUSTER')[available_metrics].agg(['mean', 'median', 'std', 'min', 'max'])
    print("\nStatistiche dei cluster:")
    print(cluster_stats)
else:
    print("Nessuna metrica disponibile per l'analisi dei cluster.")

print(f"Analisi cluster completata in {time.time() - start_time:.2f} secondi.")

# OTTIMIZZAZIONE 7: Generazione rapida delle strategie di marketing
print("\nGenerazione rapida delle strategie di marketing...")
start_time = time.time()

# Creare strategie semplici basate sui cluster
marketing_strategies = []

for cluster_id in range(best_n_clusters):
    cluster_df = df_with_clusters[df_with_clusters['CLUSTER'] == cluster_id]
    
    # Calcola caratteristiche chiave
    profile = {}
    for metric in available_metrics:
        profile[metric] = cluster_df[metric].mean()
    
    # Determina il tipo di cluster in base ai valori
    strategy = {
        'Cluster': f'Cluster {cluster_id}',
        'Size': len(cluster_df),
        'Percentage': f"{len(cluster_df) / len(df_with_clusters) * 100:.1f}%",
        'Profile': '',
        'Marketing_Strategy': '',
        'Product_Recommendations': '',
        'Communication_Channel': ''
    }
    
    # Logica per la caratterizzazione
    if 'POLICYHOLDER_AGE' in profile:
        age = profile['POLICYHOLDER_AGE']
        if age < 30:
            age_profile = "molto giovani"
        elif age < 40:
            age_profile = "giovani"
        elif age < 55:
            age_profile = "adulti"
        elif age < 70:
            age_profile = "senior"
        else:
            age_profile = "anziani"
    else:
        age_profile = "vari"
        
    if 'COMBINED_RISK_INDEX' in profile:
        risk = profile['COMBINED_RISK_INDEX']
        if risk < 0.3:
            risk_profile = "bassissimo rischio"
        elif risk < 0.5:
            risk_profile = "basso rischio"
        elif risk < 0.7:
            risk_profile = "medio rischio"
        elif risk < 0.85:
            risk_profile = "alto rischio"
        else:
            risk_profile = "altissimo rischio"
    else:
        risk_profile = "rischio vario"
        
    if 'CUSTOMER_VALUE_INDEX' in profile:
        value = profile['CUSTOMER_VALUE_INDEX']
        if value < 5:
            value_profile = "basso valore"
        elif value < 10:
            value_profile = "medio-basso valore"
        elif value < 15:
            value_profile = "medio valore"
        elif value < 20:
            value_profile = "medio-alto valore"
        else:
            value_profile = "alto valore"
    else:
        value_profile = "valore vario"
    
    # MODIFICA: Aggiungi informazioni su fedeltà/retention
    if 'CUSTOMER_TENURE_YEARS' in profile:
        tenure = profile['CUSTOMER_TENURE_YEARS']
        if tenure < 1:
            tenure_profile = "nuovi clienti"
        elif tenure < 3:
            tenure_profile = "clienti recenti"
        elif tenure < 7:
            tenure_profile = "clienti consolidati" 
        else:
            tenure_profile = "clienti di lunga data"
    else:
        tenure_profile = ""
    
    # Imposta il profilo
    if tenure_profile:
        strategy['Profile'] = f"Clienti {age_profile}, {risk_profile}, {value_profile}, {tenure_profile}"
    else:
        strategy['Profile'] = f"Clienti {age_profile}, {risk_profile}, {value_profile}"
    
    # MODIFICA: Strategia più granulare basata sui profili identificati
    # Matrice di decisione semplificata
    if "basso rischio" in risk_profile and "alto valore" in value_profile:
        strategy['Marketing_Strategy'] = "Premium loyalty program"
        strategy['Product_Recommendations'] = "Premium add-ons, family protection plan"
        strategy['Communication_Channel'] = "Personalized, app, direct contact"
    elif "medio rischio" in risk_profile and "alto valore" in value_profile:
        strategy['Marketing_Strategy'] = "Value protection + premium services"
        strategy['Product_Recommendations'] = "Extended coverage, concierge service"
        strategy['Communication_Channel'] = "App notifications, email, phone"
    elif "alto rischio" in risk_profile and "alto valore" in value_profile:
        strategy['Marketing_Strategy'] = "Risk prevention + premium retention"
        strategy['Product_Recommendations'] = "Additional protection, smart devices"
        strategy['Communication_Channel'] = "Direct contact, personalized app"
    elif "basso rischio" in risk_profile and "medio valore" in value_profile:
        strategy['Marketing_Strategy'] = "Value enhancement"
        strategy['Product_Recommendations'] = "Mid-tier products, bundle offers"
        strategy['Communication_Channel'] = "Email, app notifications"
    elif "medio rischio" in risk_profile and "medio valore" in value_profile:
        strategy['Marketing_Strategy'] = "Balanced service improvement"
        strategy['Product_Recommendations'] = "Customized packages"
        strategy['Communication_Channel'] = "Email, SMS, app"
    elif "alto rischio" in risk_profile and "medio valore" in value_profile:
        strategy['Marketing_Strategy'] = "Risk management education"
        strategy['Product_Recommendations'] = "Protection bundles, monitoring"
        strategy['Communication_Channel'] = "Educational content, email"
    elif "basso rischio" in risk_profile and "basso valore" in value_profile:
        strategy['Marketing_Strategy'] = "Cross-selling focus"
        strategy['Product_Recommendations'] = "Basic packages, complementary products"
        strategy['Communication_Channel'] = "Email, SMS, social media"
    elif "medio rischio" in risk_profile and "basso valore" in value_profile:
        strategy['Marketing_Strategy'] = "Risk-aware value growth"
        strategy['Product_Recommendations'] = "Basic protection, starter packages"
        strategy['Communication_Channel'] = "Email, educational content"
    elif "alto rischio" in risk_profile and "basso valore" in value_profile:
        strategy['Marketing_Strategy'] = "Essential risk education"
        strategy['Product_Recommendations'] = "Basic protection, discounts for safe behavior"
        strategy['Communication_Channel'] = "Educational content, social media"
    else:
        strategy['Marketing_Strategy'] = "Balanced approach"
        strategy['Product_Recommendations'] = "Mixed products"
        strategy['Communication_Channel'] = "Multi-channel"
    
    marketing_strategies.append(strategy)

# Converti in DataFrame
marketing_strategies_df = pd.DataFrame(marketing_strategies)
print("\nStrategie di marketing generate:")
print(marketing_strategies_df)

print(f"Generazione strategie completata in {time.time() - start_time:.2f} secondi.")

# OTTIMIZZAZIONE 8: Risultati applicati all'intero dataset (opzionale, solo se necessario)
if len(df_sample) < len(df_enriched):
    print("\nApplicazione dei risultati all'intero dataset...")
    start_time = time.time()
    
    # Addestra un modello finale con i parametri ottimali
    final_model = KMeans(
        n_clusters=best_n_clusters, 
        random_state=42, 
        n_init=1,  # Usa una sola inizializzazione per velocità
        max_iter=100,
        algorithm='elkan'  # Generalmente più veloce per dataset di piccole/medie dimensioni
    )
    
    # Prepara i dati completi
    full_features = df_enriched[available_features].copy()
    for col in full_features.columns:
        if full_features[col].isnull().sum() > 0:
            full_features[col].fillna(full_features[col].median(), inplace=True)
    
    full_scaled = scaler.transform(full_features)
    
    # Predici i cluster
    full_labels = final_model.fit_predict(full_scaled)
    
    # Aggiungi al dataset completo
    df_enriched['CLUSTER'] = full_labels
    
    # Mostra la distribuzione
    full_counts = np.bincount(full_labels)
    for i, count in enumerate(full_counts):
        print(f"  - Cluster {i} (dataset completo): {count} elementi ({count/len(full_labels)*100:.1f}%)")
    
    print(f"Applicazione all'intero dataset completata in {time.time() - start_time:.2f} secondi.")

# Salva il modello e i risultati principali
print("\nSalvataggio dei risultati principali...")

try:
    # Salva le strategie di marketing
    marketing_strategies_df.to_csv('marketing_strategies.csv', index=False)
    print("Strategie di marketing salvate in 'marketing_strategies.csv'")
    
    # Salva statistiche cluster
    if isinstance(cluster_stats, pd.DataFrame):
        cluster_stats.to_csv('cluster_statistics.csv')
        print("Statistiche dei cluster salvate in 'cluster_statistics.csv'")
    
    # Salva i risultati dell'analisi del numero ottimale di cluster
    pd.DataFrame(results).to_csv('cluster_analysis_results.csv', index=False)
    print("Risultati dell'analisi dei cluster salvati in 'cluster_analysis_results.csv'")
    
    # Salva il dataset con le etichette
    df_with_clusters.to_csv('clustered_sample_data.csv', index=False)
    print("Dataset con cluster salvato in 'clustered_sample_data.csv'")
    
    # Se è stato elaborato l'intero dataset
    if 'CLUSTER' in df_enriched.columns:
        # Salva solo le colonne essenziali e il cluster
        save_cols = ['CLUSTER'] + [c for c in available_features if c in df_enriched.columns]
        df_enriched[save_cols].to_csv('full_dataset_clusters.csv', index=False)
        print("Dataset completo con cluster salvato in 'full_dataset_clusters.csv'")
    
except Exception as e:
    print(f"Errore durante il salvataggio dei risultati: {e}")

# Tempo totale
total_runtime = time.time() - start_load
print("\n" + "="*80)
print(f"PIPELINE ACCELERATA COMPLETATA in {total_runtime:.2f} secondi ({total_runtime/60:.2f} minuti)")
print("="*80)
print(f"Script completato alle {time.strftime('%Y-%m-%d %H:%M:%S')}")