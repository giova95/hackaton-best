import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def prepare_data_for_clustering(df_enriched):
    """
    Prepara i dati per la clusterizzazione, selezionando e preprocessando le features.
    
    Args:
        df_enriched: DataFrame arricchito
        
    Returns:
        tuple: (DataFrame con features selezionate e preprocessate, nomi delle features)
    """
    # 1. Seleziona le features per la clusterizzazione
    features = [
        'POLICYHOLDER_AGE',
        'CLAIM_AMOUNT_PAID',
        'PREMIUM_AMOUNT_PAID',
        'CUSTOMER_TENURE_YEARS',
        'PREVIOUS_CLAIMS_COUNT',
        'ACTIVE_POLICIES_COUNT',
        'CUSTOMER_ANNUAL_PREMIUM',
        'PROVINCE_AVG_INCOME',
        'PROVINCE_ACCIDENT_RATE',
        'VEHICLE_ESTIMATED_VALUE',
        'VEHICLE_POWER_HP',
        'VEHICLE_RISK_INDEX',
        'CLAIM_FREQUENCY_RATIO',
        'COMBINED_RISK_INDEX',
        'CUSTOMER_VALUE_INDEX'
    ]
    
    # 2. Crea una copia e gestisci valori nulli
    df_features = df_enriched[features].copy()
    
    # Riempi eventuali valori nulli nelle features selezionate
    for col in df_features.columns:
        if df_features[col].isnull().sum() > 0:
            df_features[col].fillna(df_features[col].median(), inplace=True)
    
    # 3. Standardizza le features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_features),
        columns=df_features.columns
    )
    
    return df_scaled, features

def find_optimal_clusters(df_scaled, max_clusters=10):
    """
    Determina il numero ottimale di cluster usando il metodo dell'elbow e silhouette score.
    
    Args:
        df_scaled: DataFrame con features standardizzate
        max_clusters: Numero massimo di cluster da testare
        
    Returns:
        dict: Risultati delle metriche per ogni numero di cluster
    """
    results = {
        'n_clusters': [],
        'inertia': [],
        'silhouette': []
    }
    
    for n in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        
        # Salva i risultati
        results['n_clusters'].append(n)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(df_scaled, kmeans.labels_))
    
    return results

def visualize_cluster_metrics(results):
    """
    Visualizza le metriche dei cluster per determinare il numero ottimale.
    
    Args:
        results: Dizionario con i risultati delle metriche
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot dell'inerzia (elbow method)
    color = 'tab:blue'
    ax1.set_xlabel('Numero di Cluster')
    ax1.set_ylabel('Inerzia', color=color)
    ax1.plot(results['n_clusters'], results['inertia'], 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Plot del silhouette score
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(results['n_clusters'], results['silhouette'], 'o-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Metodo dell\'Elbow e Silhouette Score per K-Means')
    plt.tight_layout()
    return fig

def perform_clustering(df_scaled, n_clusters=4):
    """
    Esegue la clusterizzazione K-Means con il numero ottimale di cluster.
    
    Args:
        df_scaled: DataFrame con features standardizzate
        n_clusters: Numero di cluster da utilizzare
        
    Returns:
        tuple: (labels dei cluster, modello K-Means)
    """
    # Applica K-Means con il numero ottimale di cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_scaled)
    
    return cluster_labels, kmeans

def apply_pca_for_visualization(df_scaled, n_components=2):
    """
    Applica PCA per ridurre le dimensioni e visualizzare i cluster.
    
    Args:
        df_scaled: DataFrame con features standardizzate
        n_components: Numero di componenti principali
        
    Returns:
        tuple: (DataFrame con le componenti principali, modello PCA)
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_scaled)
    
    # Crea un DataFrame con le componenti principali
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return pca_df, pca

def visualize_clusters_2d(pca_df, cluster_labels, features, pca):
    """
    Visualizza i cluster in 2D usando le componenti principali.
    
    Args:
        pca_df: DataFrame con le componenti principali
        cluster_labels: Etichette dei cluster
        features: Nomi delle features originali
        pca: Modello PCA
    """
    # Aggiungi le etichette dei cluster al DataFrame PCA
    pca_df['Cluster'] = cluster_labels
    
    # Converti in categorie per legenda
    pca_df['Cluster'] = pca_df['Cluster'].astype('category')
    
    # Crea un grafico scatter con Plotly
    fig = px.scatter(
        pca_df, x='PC1', y='PC2',
        color='Cluster',
        title='Visualizzazione 2D dei Cluster con PCA',
        color_continuous_scale=px.colors.qualitative.G10,
        width=800, height=600
    )
    
    # Aggiungi informazioni sulle componenti principali
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Aggiungi frecce per rappresentare le feature loadings
    for i, feature in enumerate(features):
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1],
            line=dict(color='black', width=1, dash='dot'),
            name=feature
        )
        
        # Aggiungi etichette per le feature
        fig.add_annotation(
            x=loadings[i, 0] * 1.1,
            y=loadings[i, 1] * 1.1,
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
            arrowhead=0,
            font=dict(size=10, color='darkblue')
        )
    
    # Aggiungi titoli degli assi con la varianza spiegata
    fig.update_layout(
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza spiegata)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza spiegata)",
        legend_title="Cluster"
    )
    
    return fig

def analyze_clusters(df_enriched, cluster_labels, features):
    """
    Analizza le caratteristiche di ciascun cluster.
    
    Args:
        df_enriched: DataFrame originale arricchito
        cluster_labels: Etichette dei cluster
        features: Lista delle features utilizzate per la clusterizzazione
        
    Returns:
        DataFrame: Statistiche descrittive per ogni cluster
    """
    # Aggiungi le etichette dei cluster al DataFrame originale
    df_with_clusters = df_enriched.copy()
    df_with_clusters['CLUSTER'] = cluster_labels
    
    # Calcola le statistiche per ogni cluster
    cluster_stats = []
    
    for cluster_id in range(len(set(cluster_labels))):
        # Ottieni sottoinsiemi per cluster
        cluster_df = df_with_clusters[df_with_clusters['CLUSTER'] == cluster_id]
        
        # Calcola statistiche per le features numeriche
        stats = {}
        stats['Cluster'] = f'Cluster {cluster_id}'
        stats['Size'] = len(cluster_df)
        stats['Size_Pct'] = len(cluster_df) / len(df_with_clusters) * 100
        
        # Calcola media per ogni feature
        for feature in features:
            stats[f'{feature}_mean'] = cluster_df[feature].mean()
        
        # Calcola mediana per alcune feature importanti
        for feature in ['POLICYHOLDER_AGE', 'CUSTOMER_VALUE_INDEX', 'COMBINED_RISK_INDEX']:
            stats[f'{feature}_median'] = cluster_df[feature].median()
        
        # Aggiungi statistiche sulle feature categoriche
        for cat_col in ['POLICYHOLDER_GENDER', 'WARRANTY', 'ACQUISITION_CHANNEL']:
            if cat_col in df_with_clusters.columns:
                # Top 3 valori
                top3 = cluster_df[cat_col].value_counts(normalize=True).nlargest(3)
                for i, (val, pct) in enumerate(top3.items()):
                    stats[f'{cat_col}_top{i+1}'] = val
                    stats[f'{cat_col}_top{i+1}_pct'] = pct * 100
        
        cluster_stats.append(stats)
    
    # Converti in DataFrame
    cluster_stats_df = pd.DataFrame(cluster_stats)
    
    return cluster_stats_df

def visualize_cluster_profiles(df_with_clusters, features, n_clusters):
    """
    Visualizza i profili dei cluster per le principali features.
    
    Args:
        df_with_clusters: DataFrame con le etichette dei cluster
        features: Lista delle features da visualizzare
        n_clusters: Numero di cluster
    """
    # Seleziona le feature più significative da visualizzare
    key_features = [
        'POLICYHOLDER_AGE', 
        'CUSTOMER_VALUE_INDEX', 
        'COMBINED_RISK_INDEX',
        'CLAIM_FREQUENCY_RATIO',
        'VEHICLE_RISK_INDEX',
        'CUSTOMER_ANNUAL_PREMIUM'
    ]
    
    # Calcola valori medi per ogni feature in ogni cluster
    cluster_means = df_with_clusters.groupby('CLUSTER')[key_features].mean()
    
    # Standardizza i valori per la visualizzazione radar
    scaler = StandardScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        columns=cluster_means.columns,
        index=cluster_means.index
    )
    
    # Crea un grafico radar per ogni cluster
    fig = make_subplots(
        rows=2, cols=2, 
        specs=[[{'type': 'polar'}, {'type': 'polar'}], [{'type': 'polar'}, {'type': 'polar'}]],
        subplot_titles=[f'Cluster {i}' for i in range(n_clusters)]
    )
    
    # Aggiunge i dati per ogni cluster
    for i in range(n_clusters):
        row, col = i // 2 + 1, i % 2 + 1
        
        # Prepara i dati per il grafico radar
        cluster_data = cluster_means_scaled.iloc[i].tolist()
        # Chiudi il cerchio ripetendo il primo valore
        cluster_data.append(cluster_data[0])
        
        feature_names = list(key_features)
        # Chiudi il cerchio ripetendo il primo nome
        feature_names.append(feature_names[0])
        
        # Aggiungi il grafico radar
        fig.add_trace(
            go.Scatterpolar(
                r=cluster_data,
                theta=feature_names,
                fill='toself',
                name=f'Cluster {i}'
            ),
            row=row, col=col
        )
    
    # Aggiorna il layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text="Profili dei Cluster per le Principali Caratteristiche",
        showlegend=False
    )
    
    return fig

def create_marketing_strategy(cluster_stats_df, df_with_clusters):
    """
    Definisce strategie di marketing personalizzate per ciascun cluster.
    
    Args:
        cluster_stats_df: DataFrame con le statistiche dei cluster
        df_with_clusters: DataFrame con le etichette dei cluster
        
    Returns:
        DataFrame: Strategie di marketing per ogni cluster
    """
    strategies = []
    
    for i, row in cluster_stats_df.iterrows():
        cluster_id = i
        cluster_size = row['Size']
        cluster_pct = row['Size_Pct']
        
        strategy = {
            'Cluster': f'Cluster {cluster_id}',
            'Size': cluster_size,
            'Percentage': f"{cluster_pct:.1f}%",
            'Profile': '',
            'Risk_Level': '',
            'Value_Level': '',
            'Marketing_Strategy': '',
            'Product_Recommendations': '',
            'Communication_Channel': '',
            'Pricing_Strategy': '',
            'Retention_Strategy': ''
        }
        
        # Determina il profilo del cluster in base alle caratteristiche
        age_mean = row['POLICYHOLDER_AGE_mean']
        value_index = row['CUSTOMER_VALUE_INDEX_mean']
        risk_index = row['COMBINED_RISK_INDEX_mean']
        tenure = row['CUSTOMER_TENURE_YEARS_mean']
        policies = row['ACTIVE_POLICIES_COUNT_mean']
        claim_freq = row['CLAIM_FREQUENCY_RATIO_mean']
        
        # Profilo demografico
        if age_mean < 35:
            age_profile = "giovani"
        elif age_mean < 55:
            age_profile = "adulti"
        elif age_mean < 70:
            age_profile = "senior"
        else:
            age_profile = "anziani"
        
        # Livello di rischio
        if risk_index < 0.4:
            strategy['Risk_Level'] = "Basso"
            risk_profile = "a basso rischio"
        elif risk_index < 0.7:
            strategy['Risk_Level'] = "Medio"
            risk_profile = "a rischio medio"
        else:
            strategy['Risk_Level'] = "Alto"
            risk_profile = "ad alto rischio"
        
        # Livello di valore
        if value_index < 5:
            strategy['Value_Level'] = "Basso"
            value_profile = "a basso valore"
        elif value_index < 15:
            strategy['Value_Level'] = "Medio"
            value_profile = "a valore medio"
        else:
            strategy['Value_Level'] = "Alto"
            value_profile = "ad alto valore"
        
        # Fedeltà cliente
        if tenure < 3:
            loyalty_profile = "nuovi"
        elif tenure < 7:
            loyalty_profile = "consolidati"
        else:
            loyalty_profile = "fedeli di lunga data"
        
        # Combina le caratteristiche per creare un profilo
        strategy['Profile'] = f"Clienti {age_profile} {risk_profile} {value_profile}, {loyalty_profile}"
        
        # Definisci strategie in base al profilo
        
        # 1. CLUSTER A BASSO RISCHIO, ALTO VALORE
        if risk_index < 0.4 and value_index > 15:
            strategy['Marketing_Strategy'] = "Programma di fidelizzazione premium e cross-selling"
            strategy['Product_Recommendations'] = "Garanzie aggiuntive premium, assicurazioni casa/salute/viaggi"
            strategy['Communication_Channel'] = "Comunicazione personalizzata, contatto diretto dell'agente"
            strategy['Pricing_Strategy'] = "Prezzi premium con sconti di fedeltà, sconti multi-polizza"
            strategy['Retention_Strategy'] = "Servizi esclusivi, assistenza dedicata 24/7, inviti ad eventi"
        
        # 2. CLUSTER AD ALTO RISCHIO, ALTO VALORE
        elif risk_index >= 0.7 and value_index > 15:
            strategy['Marketing_Strategy'] = "Programma di prevenzione e riduzione rischi con incentivi"
            strategy['Product_Recommendations'] = "Protezioni aggiuntive, servizi di monitoraggio e assistenza"
            strategy['Communication_Channel'] = "App dedicata con notifiche, contatto proattivo"
            strategy['Pricing_Strategy'] = "Prezzi premium con sconti basati su comportamenti virtuosi"
            strategy['Retention_Strategy'] = "Programma punti per comportamenti sicuri, assistenza prioritaria"
        
        # 3. CLUSTER A BASSO RISCHIO, BASSO VALORE
        elif risk_index < 0.4 and value_index < 5:
            strategy['Marketing_Strategy'] = "Crescita del valore cliente con cross-selling mirato"
            strategy['Product_Recommendations'] = "Pacchetti base con opportunità di upgrade"
            strategy['Communication_Channel'] = "Email, SMS, app self-service"
            strategy['Pricing_Strategy'] = "Prezzi competitivi con incentivi per polizze aggiuntive"
            strategy['Retention_Strategy'] = "Rinnovi automatici con sconti incrementali, processi digitalizzati"
        
        # 4. CLUSTER AD ALTO RISCHIO, BASSO VALORE
        elif risk_index >= 0.7 and value_index < 5:
            strategy['Marketing_Strategy'] = "Ritenzione selettiva con mitigazione rischi"
            strategy['Product_Recommendations'] = "Prodotti base con franchigie elevate, servizi di assistenza"
            strategy['Communication_Channel'] = "Comunicazione educativa, formazione sulla prevenzione"
            strategy['Pricing_Strategy'] = "Pricing dinamico basato sul rischio, incentivi per comportamenti virtuosi"
            strategy['Retention_Strategy'] = "Rivalutazione periodica del profilo di rischio, programmi educativi"
        
        # 5. CLIENTI GIOVANI (STRATEGIE SPECIFICHE)
        elif age_mean < 35:
            strategy['Marketing_Strategy'] = "Acquisizione digitale e costruzione relazione"
            strategy['Product_Recommendations'] = "Prodotti entry-level, assicurazioni pay-per-use, garanzie su misura"
            strategy['Communication_Channel'] = "Social media, app mobile, messaggistica istantanea"
            strategy['Pricing_Strategy'] = "Prezzi competitivi, piani rateizzati, sconti per comportamenti virtuosi"
            strategy['Retention_Strategy'] = "Programma loyalty digitale, gamification, community online"
        
        # 6. CLIENTI SENIOR (STRATEGIE SPECIFICHE)
        elif age_mean >= 65:
            strategy['Marketing_Strategy'] = "Approccio tradizionale, servizi premium di assistenza"
            strategy['Product_Recommendations'] = "Pacchetti completi, assistenza medica, tutela patrimonio"
            strategy['Communication_Channel'] = "Consulenti dedicati, materiale cartaceo, call center dedicato"
            strategy['Pricing_Strategy'] = "Prezzi tutto-incluso, facilità di pagamento, tariffe stabili"
            strategy['Retention_Strategy'] = "Contatto periodico personale, eventi dedicati, servizi aggiuntivi gratuiti"
        
        # FALLBACK PER ALTRI CLUSTER
        else:
            strategy['Marketing_Strategy'] = "Approccio bilanciato di cross-selling e up-selling"
            strategy['Product_Recommendations'] = "Mix di prodotti base e premium in base al profilo"
            strategy['Communication_Channel'] = "Mix di canali digitali e tradizionali"
            strategy['Pricing_Strategy'] = "Prezzi competitivi con personalizzazioni"
            strategy['Retention_Strategy'] = "Programma loyalty standard, comunicazioni periodiche"
        
        strategies.append(strategy)
    
    return pd.DataFrame(strategies)

# Funzione principale per eseguire l'intero flusso di clusterizzazione
def run_customer_clustering_pipeline(df_enriched):
    """
    Esegue l'intero flusso di clusterizzazione e generazione di strategie.
    
    Args:
        df_enriched: DataFrame arricchito
        
    Returns:
        tuple: (DataFrame con cluster, statistiche dei cluster, strategie di marketing)
    """
    # 1. Prepara i dati
    df_scaled, features = prepare_data_for_clustering(df_enriched)
    
    # 2. Trova il numero ottimale di cluster
    clustering_metrics = find_optimal_clusters(df_scaled)
    visualize_cluster_metrics(clustering_metrics)
    
    # 3. Seleziona il numero di cluster (in base alle metriche)
    # Per questo esempio usiamo 4 cluster
    n_clusters = 4
    
    # 4. Esegui la clusterizzazione
    cluster_labels, kmeans_model = perform_clustering(df_scaled, n_clusters)
    
    # 5. Applica PCA per visualizzazione
    pca_df, pca_model = apply_pca_for_visualization(df_scaled)
    
    # 6. Visualizza i cluster
    cluster_viz = visualize_clusters_2d(pca_df, cluster_labels, features, pca_model)
    
    # 7. Aggiungi etichette cluster al dataset originale
    df_with_clusters = df_enriched.copy()
    df_with_clusters['CLUSTER'] = cluster_labels
    
    # 8. Analizza i cluster
    cluster_stats = analyze_clusters(df_enriched, cluster_labels, features)
    
    # 9. Visualizza profili dei cluster
    cluster_profiles = visualize_cluster_profiles(df_with_clusters, features, n_clusters)
    
    # 10. Definisci strategie di marketing
    marketing_strategies = create_marketing_strategy(cluster_stats, df_with_clusters)
    
    return df_with_clusters, cluster_stats, marketing_strategies


df_enriched = pd.read_excel('dataset_enriched.xlsx')
df_with_clusters, cluster_stats, marketing_strategies = run_customer_clustering_pipeline(df_enriched)
print(marketing_strategies)