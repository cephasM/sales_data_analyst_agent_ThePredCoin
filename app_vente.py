#!/usr/bin/env python
# coding: utf-8

# In[8]:


###############################################
### 1. CONFIGURATION INITIALE ET IMPORTS ###
###############################################

import warnings
import logging
import os
import time
import base64
from io import BytesIO
from fpdf import FPDF
import tempfile
from datetime import datetime

# Désactivation des warnings et configuration des logs
warnings.filterwarnings("ignore")
os.environ['STREAMLIT_HIDE_SCRIPT_RUN_CONTEXT'] = 'true'
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Imports principaux
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse Prédictive des Ventes", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

#######################################
### 2. FONCTIONS UTILITAIRES ###
#######################################

@st.cache_data
def load_data(uploaded_file):
    """Charge le fichier Excel/CSV"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ Fichier chargé avec succès")
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return None

@st.cache_data
def clean_data(df):
    """Nettoie les données"""
    df_cleaned = df.copy()
    
    # Conversion des dates
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'jour' in col.lower()]
    for col in date_cols:
        try:
            df_cleaned[col] = pd.to_datetime(df_cleaned[col]).dt.normalize()
        except:
            pass
    
    # Conversion des valeurs numériques
    for col in df_cleaned.columns:
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    # Suppression des lignes avec valeurs manquantes
    df_cleaned = df_cleaned.dropna()
    
    st.success("🧹 Données nettoyées")
    return df_cleaned

def generate_pdf_report(analysis_results, figures, filtered_df, value_col):
    """Génère un rapport PDF professionnel"""
    pdf = FPDF()
    pdf.add_page()
    
    # Style du document
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    
    # En-tête avec logo
    pdf.image("https://via.placeholder.com/150x50?text=LOGO", x=10, y=8, w=40)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Rapport d'Analyse Commerciale", ln=1, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=1, align='C')
    pdf.ln(15)
    
    # Section Résumé
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Résumé Analytique", ln=1)
    pdf.set_font("Arial", '', 12)
    
    summary_text = [
        f"Période analysée: {analysis_results['period']}",
        f"Chiffre d'affaires total: {analysis_results['total_sales']:,.2f} €",
        f"Nombre de transactions: {len(filtered_df):,}",
        f"Valeur moyenne: {filtered_df[value_col].mean():,.2f} €"
    ]
    
    for line in summary_text:
        pdf.multi_cell(0, 10, txt=line)
    
    pdf.ln(10)
    
    # Graphiques
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Visualisations Clés", ln=1)
    
    temp_files = []
    for name, fig in figures.items():
        # Sauvegarde temporaire des images
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.write_image(temp_file.name, width=800, height=500)
        temp_files.append(temp_file)
        
        # Ajout au PDF
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Graphique: {name.replace('_', ' ').title()}", ln=1)
        pdf.image(temp_file.name, w=180)
        pdf.ln(5)
    
    # Pied de page
    pdf.set_y(-15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, f"Page {pdf.page_no()}", 0, 0, 'C')
    
    # Nettoyage des fichiers temporaires
    for temp_file in temp_files:
        temp_file.close()
    
    return pdf

##############################################
### 3. FONCTIONS D'ANALYSE DE DONNÉES ###
##############################################

def detect_columns(df):
    """Détecte automatiquement les types de colonnes"""
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if not date_cols:
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'jour' in col.lower()]
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    product_cols = [col for col in df.columns if 'produit' in col.lower() or 'product' in col.lower()]
    region_cols = [col for col in df.columns if 'région' in col.lower() or 'region' in col.lower()]
    
    return {
        'date_cols': date_cols,
        'numeric_cols': numeric_cols,
        'product_cols': product_cols,
        'region_cols': region_cols
    }

def apply_filters(df, date_col, date_range, region_col, regions, product_col, products):
    """Applique les filtres aux données"""
    filtered_df = df.copy()
    
    # Filtre temporel
    if pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]) and date_range[0] is not None:
        filtered_df = filtered_df[
            (filtered_df[date_col] >= pd.to_datetime(date_range[0])) & 
            (filtered_df[date_col] <= pd.to_datetime(date_range[1]))
        ]
    
    # Filtres par région et produit
    if regions and products:
        filtered_df = filtered_df[
            filtered_df[region_col].isin(regions) &
            filtered_df[product_col].isin(products)
        ]
    
    return filtered_df

def generate_plots(filtered_df, date_col, value_col, region_col, product_col):
    """Génère les visualisations Plotly"""
    figures = {}
    
    # 1. Graphique des ventes par région
    region_sales = filtered_df.groupby(region_col)[value_col].sum().sort_values(ascending=False)
    figures['region_sales'] = px.bar(
        region_sales,
        title="<b>Chiffre d'affaires par région</b>",
        labels={'value': 'CA (€)', 'index': 'Région'},
        color=region_sales.values,
        color_continuous_scale='Bluered'
    )
    
    # 2. Top produits
    top_products = filtered_df.groupby(product_col)[value_col].sum().nlargest(10)
    figures['top_products'] = px.bar(
        top_products,
        title="<b>Top 10 produits par CA</b>",
        labels={'value': 'CA (€)', 'index': 'Produit'},
        color=top_products.values,
        color_continuous_scale='Greens'
    )
    
    # 3. Évolution temporelle
    if pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
        time_series = filtered_df.groupby(pd.Grouper(key=date_col, freq='M'))[value_col].sum().reset_index()
        time_series[date_col] = time_series[date_col].dt.strftime('%Y-%m-%d')
        figures['time_series'] = px.line(
            time_series,
            x=date_col,
            y=value_col,
            title="<b>Évolution mensuelle du CA</b>",
            labels={value_col: 'CA (€)', date_col: 'Mois'},
            markers=True
        ).update_layout(hovermode="x unified")
    
    return figures

########################################
### 4. INTERFACE UTILISATEUR ###
########################################

def setup_sidebar():
    """Configure la barre latérale"""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Sales+Analytics", width=150)
        st.title("Paramètres")
        
        if 'df' in st.session_state:
            st.success("Données chargées")
            st.info(f"Enregistrements: {len(st.session_state['df']):,}")

def display_main_header():
    """Affiche l'en-tête principal"""
    st.title("📈 Analyse Prédictive des Ventes")
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1>Plateforme d'Analyse Commerciale Avancée</h1>
        <p>Chargez vos données de vente pour obtenir des analyses détaillées</p>
    </div>
    """, unsafe_allow_html=True)

def data_loading_section():
    """Section de chargement des données"""
    with st.expander("📤 Chargement des données", expanded=True):
        uploaded_file = st.file_uploader("Téléchargez votre fichier Excel ou CSV", 
                                       type=["xlsx", "csv"], 
                                       key="file_uploader")
        
        if uploaded_file is not None:
            with st.spinner('Chargement en cours...'):
                df = load_data(uploaded_file)
                time.sleep(1)
                
                if df is not None:
                    df_clean = clean_data(df)
                    st.session_state['df'] = df_clean
                    
                    # Aperçu des données
                    st.subheader("Aperçu des données")
                    st.dataframe(df_clean.head(), use_container_width=True)

def data_analysis_section():
    """Section d'analyse des données"""
    if 'df' not in st.session_state:
        return
    
    with st.expander("🔍 Analyse exploratoire", expanded=True):
        df_clean = st.session_state['df']
        col_types = detect_columns(df_clean)
        
        # Sélection des colonnes
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            date_col = st.selectbox(
                "Colonne de date",
                options=df_clean.columns,
                index=0 if not col_types['date_cols'] else df_clean.columns.get_loc(col_types['date_cols'][0])
            )
        with col2:
            value_col = st.selectbox(
                "Colonne de valeur",
                options=col_types['numeric_cols'],
                index=0
            )
        with col3:
            product_col = st.selectbox(
                "Colonne de produit",
                options=df_clean.columns,
                index=0 if not col_types['product_cols'] else df_clean.columns.get_loc(col_types['product_cols'][0])
            )
        with col4:
            region_col = st.selectbox(
                "Colonne de région",
                options=df_clean.columns,
                index=0 if not col_types['region_cols'] else df_clean.columns.get_loc(col_types['region_cols'][0])
            )
        
        # Filtres interactifs
        st.markdown("<h3 style='color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 10px;'>Filtres interactifs</h3>", 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Filtre temporel
            if pd.api.types.is_datetime64_any_dtype(df_clean[date_col]):
                try:
                    min_date = df_clean[date_col].min().to_pydatetime()
                    max_date = df_clean[date_col].max().to_pydatetime()
                    date_range = st.slider(
                        "Période",
                        min_value=min_date,
                        max_value=max_date,
                        value=(min_date, max_date),
                        format="DD/MM/YYYY"
                    )
                except Exception as e:
                    st.error(f"Erreur avec les dates : {str(e)}")
                    date_range = (df_clean[date_col].min(), df_clean[date_col].max())
        
        with col2:
            # Filtres par région et produit
            regions = st.multiselect(
                "Régions à inclure",
                options=df_clean[region_col].unique(),
                default=df_clean[region_col].unique()
            )
            products = st.multiselect(
                "Produits à inclure",
                options=df_clean[product_col].unique(),
                default=df_clean[product_col].unique()
            )
        
        # Application des filtres
        filtered_df = apply_filters(df_clean, date_col, date_range, region_col, regions, product_col, products)
        
        # Analyse et visualisation
        with st.spinner('Analyse en cours...'):
            # KPI principaux
            st.markdown("<h3 style='color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 10px;'>Indicateurs Clés</h3>", 
                       unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chiffre d'affaires total", f"{filtered_df[value_col].sum():,.2f} €")
            with col2:
                st.metric("Nombre de transactions", f"{len(filtered_df):,}")
            with col3:
                st.metric("Valeur moyenne", f"{filtered_df[value_col].mean():,.2f} €")
            
            # Génération des graphiques
            figures = generate_plots(filtered_df, date_col, value_col, region_col, product_col)
            
            # Affichage des graphiques
            for fig in figures.values():
                st.plotly_chart(fig, use_container_width=True)
            
            # Sauvegarde des résultats
            st.session_state['figures'] = figures
            st.session_state['filtered_df'] = filtered_df
            st.session_state['analysis_results'] = {
                'period': f"{date_range[0].strftime('%d/%m/%Y')} au {date_range[1].strftime('%d/%m/%Y')}",
                'total_sales': filtered_df[value_col].sum()
            }
            st.session_state['value_col'] = value_col

        # Section Export PDF
        st.markdown("---")
        st.subheader("📊 Export du Rapport")
        
        if st.button("🔄 Générer le Rapport PDF", key="generate_pdf"):
            with st.spinner("Création du rapport PDF..."):
                st.session_state['pdf_report'] = generate_pdf_report(
                    st.session_state['analysis_results'],
                    st.session_state['figures'],
                    st.session_state['filtered_df'],
                    st.session_state['value_col']
                )
                st.success("Rapport PDF prêt !")
        
        if 'pdf_report' in st.session_state:
            # Style CSS personnalisé pour le bouton
            st.markdown("""
            <style>
                .download-btn {
                    display: inline-block;
                    padding: 12px 24px;
                    background-color: #3498db;
                    color: white;
                    text-decoration: none;
                    border-radius: 6px;
                    font-weight: bold;
                    transition: all 0.3s;
                    border: none;
                    cursor: pointer;
                    text-align: center;
                    margin-top: 10px;
                }
                .download-btn:hover {
                    background-color: #2980b9;
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                }
            </style>
            """, unsafe_allow_html=True)
            
            pdf_output = st.session_state['pdf_report'].output(dest='S').encode('latin-1')
            st.markdown(
                f'<a href="data:application/pdf;base64,{base64.b64encode(pdf_output).decode()}" '
                f'download="rapport_ventes_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf" '
                'class="download-btn">📥 Télécharger le Rapport PDF</a>',
                unsafe_allow_html=True
            )

#######################################
### 5. FONCTION PRINCIPALE ###
#######################################

def main():
    # Configuration de l'interface
    setup_sidebar()
    display_main_header()
    
    # Sections principales
    data_loading_section()
    data_analysis_section()

if __name__ == "__main__":
    main()


# In[ ]:




