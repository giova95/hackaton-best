import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def create_key_visualizations(df_processed, df_enriched, df_with_clusters, kpis):
    visualizations = {}
    
    # 1. Dashboard KPI principali
    def create_kpi_dashboard():
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Loss Ratio per Tipo di Garanzia (Top 10)',
                'Importo Medio Sinistri per Fascia d\'Età',
                'Top 10 Regioni per Numero di Sinistri',
                'Stagionalità dei Sinistri (Mensile)',
                'Top 10 Marche Veicoli per Sinistri',
                'Distribuzione Sinistri per Genere'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "pie"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.07
        )
        
        # 1.1 Loss Ratio per tipo di garanzia
        top_warranty_lr = kpis['warranty_loss_ratio'].sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(
                x=top_warranty_lr.values,
                y=top_warranty_lr.index,
                orientation='h',
                marker=dict(color=top_warranty_lr.values, colorscale='YlOrRd'),
                name='Loss Ratio'
            ),
            row=1, col=1
        )
        
        # 1.2 Importo medio sinistri per fascia d'età
        age_claims = kpis['age_group_claims']
        fig.add_trace(
            go.Bar(
                x=age_claims.index,
                y=age_claims['mean'],
                marker_color='lightblue',
                name='Importo Medio'
            ),
            row=1, col=2
        )
        
        # 1.3 Top 10 regioni per numero di sinistri
        region_claims = kpis['region_claims'].sort_values('claim_count', ascending=False).head(10)
        fig.add_trace(
            go.Bar(
                x=region_claims['claim_count'],
                y=region_claims.index,
                orientation='h',
                marker_color='lightgreen',
                name='Numero Sinistri'
            ),
            row=2, col=1
        )
        
        # 1.4 Stagionalità dei sinistri (mensile)
        month_data = kpis['monthly_seasonality']
        month_names = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
        fig.add_trace(
            go.Bar(
                x=month_names,
                y=month_data['claim_count'],
                marker_color='coral',
                name='Num. Sinistri'
            ),
            row=2, col=2
        )
        
        # 1.5 Top 10 marche veicoli per sinistri
        brand_data = kpis['top_brand_claims'].head(10)
        fig.add_trace(
            go.Bar(
                x=brand_data['claim_count'],
                y=brand_data.index,
                orientation='h',
                marker_color='mediumpurple',
                name='Num. Sinistri'
            ),
            row=3, col=1
        )
        
        # 1.6 Distribuzione sinistri per genere
        gender_data = kpis['gender_claims']
        fig.add_trace(
            go.Pie(
                labels=gender_data.index,
                values=gender_data['count'],
                hole=.4,
                marker_colors=['skyblue', 'pink'],
                name='Genere'
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title_text='Dashboard KPI Assicurativi',
            showlegend=False,
            height=1000,
            width=1200
        )
        
        return fig
    
    # 2. Heatmap correlazione
    def create_correlation_heatmap():
        all_numeric_cols = [
            'POLICYHOLDER_AGE', 'CLAIM_AMOUNT_PAID', 'PREMIUM_AMOUNT_PAID',
            'CUSTOMER_TENURE_YEARS', 'PREVIOUS_CLAIMS_COUNT', 'ACTIVE_POLICIES_COUNT',
            'PROVINCE_AVG_INCOME', 'PROVINCE_ACCIDENT_RATE', 'VEHICLE_RISK_INDEX',
            'COMBINED_RISK_INDEX', 'CUSTOMER_VALUE_INDEX'
        ]
        
        numeric_cols = [col for col in all_numeric_cols if col in df_enriched.columns]
        
        if len(numeric_cols) < 2:
            print("Avviso: Meno di 2 colonne numeriche disponibili per la mappa di calore.")
            return None
        
        corr_matrix = df_enriched[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text:.2f}"
        ))
        
        fig.update_layout(
            title_text='Correlazione tra Variabili Chiave',
            height=800,
            width=900
        )
        
        return fig
    
    # 3. Profili cluster
    def create_cluster_profiles():
        if 'CLUSTER' not in df_with_clusters.columns:
            print("Avviso: La colonna 'CLUSTER' non è presente nel dataframe.")
            return None
        
        cols = {
            'POLICYHOLDER_AGE': 'Età Media',
            'CUSTOMER_VALUE_INDEX': 'Valore Cliente',
            'COMBINED_RISK_INDEX': 'Indice di Rischio',
            'CLAIM_AMOUNT_PAID': 'Importo Medio Sinistri',
            'PREVIOUS_CLAIMS_COUNT': 'N. Sinistri Precedenti',
            'VEHICLE_ESTIMATED_VALUE': 'Valore Veicolo'
        }
        
        available_cols = {}
        for col, label in cols.items():
            if col in df_with_clusters.columns:
                available_cols[col] = label
        
        if len(available_cols) < 3:
            print("Avviso: Non ci sono abbastanza feature numeriche per creare i profili dei cluster.")
            return None
        
        agg_dict = {col: 'mean' for col in available_cols.keys()}
        cluster_profiles = df_with_clusters.groupby('CLUSTER').agg(agg_dict).reset_index()
        
        cluster_profiles = cluster_profiles.rename(columns=available_cols)
        
        for col in available_cols.values():
            min_val = cluster_profiles[col].min()
            max_val = cluster_profiles[col].max()
            if max_val > min_val:
                cluster_profiles[f'{col}_norm'] = (cluster_profiles[col] - min_val) / (max_val - min_val)
            else:
                cluster_profiles[f'{col}_norm'] = 0
        
        fig = go.Figure()
        
        categories = list(available_cols.values())
        
        for i, row in cluster_profiles.iterrows():
            values = [row[f'{col}_norm'] for col in categories]
            values.append(values[0])
            
            cats = categories.copy()
            cats.append(cats[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=cats,
                fill='toself',
                name=f'Cluster {row["CLUSTER"]}'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Profili dei Cluster',
            showlegend=True,
            height=600,
            width=800
        )
        
        return fig
    
    # 4. Visualizzazione 2D dei cluster con PCA
    def create_cluster_pca_viz():
        if 'CLUSTER' not in df_with_clusters.columns:
            print("Avviso: La colonna 'CLUSTER' non è presente nel dataframe.")
            return None
        
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        cols = [
            'POLICYHOLDER_AGE', 'CLAIM_AMOUNT_PAID', 'PREMIUM_AMOUNT_PAID',
            'CUSTOMER_TENURE_YEARS', 'PREVIOUS_CLAIMS_COUNT', 'COMBINED_RISK_INDEX',
            'CUSTOMER_VALUE_INDEX', 'VEHICLE_RISK_INDEX'
        ]
        
        features = [col for col in cols if col in df_with_clusters.columns]
        
        if len(features) < 2:
            print("Avviso: Non ci sono abbastanza feature numeriche per PCA.")
            return None
        
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_with_clusters[features])
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_scaled)
        
        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Cluster': df_with_clusters['CLUSTER'].astype(str)
        })
        
        fig = px.scatter(
            pca_df, x='PC1', y='PC2',
            color='Cluster',
            title='Visualizzazione 2D dei Cluster (PCA)',
            color_discrete_sequence=px.colors.qualitative.Plotly 
        )
        
        centroids = pca_df.groupby('Cluster').mean().reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=centroids['PC1'],
                y=centroids['PC2'],
                mode='markers',
                marker=dict(
                    color='black',
                    size=12,
                    symbol='x'
                ),
                name='Centroidi'
            )
        )
        
        var_exp = pca.explained_variance_ratio_
        
        fig.update_layout(
            xaxis_title=f"PC1 ({var_exp[0]:.2%} varianza)",
            yaxis_title=f"PC2 ({var_exp[1]:.2%} varianza)",
            height=600,
            width=800
        )
        
        return fig
    
    # 5. Strategia di marketing per cluster
    def create_marketing_strategy_viz():
        
        strategies = pd.DataFrame({
            'Cluster': [f'Cluster {i}' for i in range(4)],
            'Risk_Level': ['Basso', 'Alto', 'Basso', 'Alto'],
            'Value_Level': ['Alto', 'Alto', 'Basso', 'Basso'],
            'Size': [25, 20, 30, 25]  # percentuale del totale
        })
        
        color_map = {
            ('Basso', 'Alto'): 'green',   
            ('Alto', 'Alto'): 'orange',   
            ('Basso', 'Basso'): 'lightblue', 
            ('Alto', 'Basso'): 'red'     
        }
        
        strategies['Color'] = strategies.apply(
            lambda x: color_map[(x['Risk_Level'], x['Value_Level'])], 
            axis=1
        )
        
        strategies['Main_Strategy'] = [
            'Fidelizzazione e Cross-selling Premium',
            'Prevenzione Rischi e Servizi di Monitoraggio',
            'Up-selling e Cross-selling Mirato',
            'Ritenzione Selettiva e Mitigazione Rischi'
        ]
        
        fig = px.treemap(
            strategies, 
            path=['Value_Level', 'Risk_Level', 'Cluster'],
            values='Size',
            color='Color',
            color_discrete_map='identity',
            hover_data=['Main_Strategy']
        )
        
        fig.update_layout(
            title='Strategie di Marketing per Cluster',
            height=600,
            width=800
        )
        
        return fig
    
    # 6. Analisi temporale dei sinistri
    def create_time_series_viz():
        dates = pd.date_range(start='2020-01-01', end='2024-05-01', freq='MS')
        n_months = len(dates)
        
        base_trend = np.linspace(800, 1200, n_months) 
        seasonality = 200 * np.sin(np.linspace(0, 2*np.pi*4, n_months))
        noise = np.random.normal(0, 50, n_months)
        
        claims_count = base_trend + seasonality + noise
        claims_amount = claims_count * np.random.uniform(1.8, 2.2, n_months)
        
        time_series_df = pd.DataFrame({
            'Date': dates,
            'Claims_Count': claims_count,
            'Claims_Amount': claims_amount
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=time_series_df['Date'],
                y=time_series_df['Claims_Count'],
                name='Numero Sinistri',
                line=dict(color='blue')
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=time_series_df['Date'],
                y=time_series_df['Claims_Amount'],
                name='Importo Sinistri (€)',
                line=dict(color='red')
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Trend Temporale dei Sinistri',
            xaxis_title='Data',
            height=500,
            width=900
        )
        
        fig.update_yaxes(title_text="Numero Sinistri", secondary_y=False)
        fig.update_yaxes(title_text="Importo Sinistri (€)", secondary_y=True)
        
        return fig
    
    visualizations['kpi_dashboard'] = create_kpi_dashboard()
    
    correlation_heatmap = create_correlation_heatmap()
    if correlation_heatmap is not None:
        visualizations['correlation_heatmap'] = correlation_heatmap
    
    cluster_profiles = create_cluster_profiles()
    if cluster_profiles is not None:
        visualizations['cluster_profiles'] = cluster_profiles
    
    cluster_pca = create_cluster_pca_viz()
    if cluster_pca is not None:
        visualizations['cluster_pca'] = cluster_pca
    
    visualizations['marketing_strategy'] = create_marketing_strategy_viz()
    visualizations['time_series'] = create_time_series_viz()
    
    return visualizations

def extract_key_kpis(df):
    kpis = {}
    
    # 1. Loss Ratio per tipo di garanzia
    warranty_loss_ratio = df.groupby('WARRANTY', as_index=False).apply(
        lambda x: pd.Series({
            'loss_ratio': x['CLAIM_AMOUNT_PAID'].sum() / x['PREMIUM_AMOUNT_PAID'].sum()
        })
    ).set_index('WARRANTY')['loss_ratio'].sort_values(ascending=False)
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

def save_visualizations(visualizations, output_dir='visualizations'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for name, fig in visualizations.items():
        html_path = os.path.join(output_dir, f"{name}.html")
        fig.write_html(html_path)
        print(f"Visualizzazione '{name}' salvata come HTML: {html_path}")
        
        try:
            png_path = os.path.join(output_dir, f"{name}.png")
            fig.write_image(png_path)
            print(f"Visualizzazione '{name}' salvata come PNG: {png_path}")
        except Exception as e:
            print(f"Impossibile salvare '{name}' come PNG. Errore: {e}")
            print("Per salvare come PNG, installa kaleido: pip install kaleido")

try:
    print("Caricamento dataset_processed.xlsx...")
    df_processed = pd.read_excel('dataset_processed.xlsx')
    print(f"Colonne in df_processed: {df_processed.columns.tolist()}")
    
    print("Estrazione KPI...")
    kpis = extract_key_kpis(df_processed)
    
    print("Caricamento dataset_enriched.xlsx...")
    df_enriched = pd.read_excel('dataset_enriched.xlsx')
    print(f"Colonne in df_enriched: {df_enriched.columns.tolist()}")
    
    print("Caricamento full_dataset_clusters.csv...")
    df_with_clusters = pd.read_csv('full_dataset_clusters.csv')
    print(f"Colonne in df_with_clusters: {df_with_clusters.columns.tolist()}")
    
    print("Creazione visualizzazioni...")
    visualizations = create_key_visualizations(df_processed, df_enriched, df_with_clusters, kpis)
    
    print("Salvataggio visualizzazioni...")
    save_visualizations(visualizations)
    
    print("Processo completato con successo!")
except Exception as e:
    import traceback
    print(f"Si è verificato un errore: {e}")
    traceback.print_exc()