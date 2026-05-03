import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Parcl - Buyer Segmentation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DARK THEME CSS
# ============================================================
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .title-text { color: #00d4ff; font-size: 36px; font-weight: bold; }
    .subtitle-text { color: #a0a0a0; font-size: 16px; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD & PREPARE DATA
# ============================================================
@st.cache_data
def load_data():
    clients = pd.read_csv("clients.csv")
    properties = pd.read_csv("properties.csv")

    # Clean sale price
    properties['sale_price'] = properties['sale_price'].str.replace(
        '[\$,]', '', regex=True).astype(float)

    # Remove duplicates
    clients = clients.drop_duplicates()
    properties = properties.drop_duplicates()

    # Handle missing values
    clients = clients.fillna(clients.mode().iloc[0])
    properties = properties.fillna(properties.mode().iloc[0])

    # Normalize categorical labels
    clients['gender'] = clients['gender'].str.strip().str.upper()
    clients['client_type'] = clients['client_type'].str.strip().str.title()
    clients['acquisition_purpose'] = clients['acquisition_purpose'].str.strip().str.title()
    clients['loan_applied'] = clients['loan_applied'].str.strip().str.title()
    clients['referral_channel'] = clients['referral_channel'].str.strip().str.title()
    clients['country'] = clients['country'].str.strip().str.title()
    clients['region'] = clients['region'].str.strip().str.title()

    # Calculate age
    clients['date_of_birth'] = pd.to_datetime(
        clients['date_of_birth'], dayfirst=True, errors='coerce')
    clients['age'] = (pd.Timestamp('2024-01-01') -
                      clients['date_of_birth']).dt.days // 365

    # Merge datasets
    merged = clients.merge(
        properties, left_on='client_id',
        right_on='client_ref', how='left')

    return clients, properties, merged

clients, properties, merged = load_data()

# ============================================================
# CLUSTERING ENGINE
# ============================================================
@st.cache_data
def run_clustering(n_clusters=4):
    df = clients.copy()

    # ── Label Encoding (for low-cardinality columns) ──
    le = LabelEncoder()
    label_encode_cols = ['gender', 'loan_applied',
                         'acquisition_purpose', 'client_type',
                         'referral_channel']
    for col in label_encode_cols:
        df[col + '_le'] = le.fit_transform(df[col].astype(str))

    # ── One-Hot Encoding (for country & region) ──
    ohe_cols = ['country', 'region']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_array = ohe.fit_transform(df[ohe_cols])
    ohe_feature_names = ohe.get_feature_names_out(ohe_cols)
    ohe_df = pd.DataFrame(ohe_array,
                          columns=ohe_feature_names,
                          index=df.index)
    df = pd.concat([df, ohe_df], axis=1)

    # ── Feature Matrix ──
    label_features = ['age', 'satisfaction_score',
                      'gender_le', 'loan_applied_le',
                      'acquisition_purpose_le',
                      'client_type_le', 'referral_channel_le']
    ohe_features = list(ohe_feature_names)
    all_features = label_features + ohe_features

    X = df[all_features].dropna()

    # ── Feature Scaling ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── K-Means Clustering ──
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=42, n_init=10)
    df.loc[X.index, 'KMeans_Cluster'] = kmeans.fit_predict(X_scaled)
    df['KMeans_Cluster'] = df['KMeans_Cluster'].fillna(-1).astype(int)

    # ── Hierarchical Clustering ──
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    df.loc[X.index, 'Hierarchical_Cluster'] = hierarchical.fit_predict(X_scaled)
    df['Hierarchical_Cluster'] = df['Hierarchical_Cluster'].fillna(-1).astype(int)

    # ── Silhouette Scores ──
    kmeans_sil = silhouette_score(X_scaled, kmeans.labels_)
    hier_sil = silhouette_score(X_scaled, hierarchical.labels_)

    # ── Elbow Method ──
    inertias = []
    k_range = range(2, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # ── Linkage Matrix for Dendrogram ──
    sample = X_scaled[:200]
    linkage_matrix = linkage(sample, method='ward')

    return (df, kmeans_sil, hier_sil,
            list(k_range), inertias,
            linkage_matrix, X_scaled)

# Cluster labels
cluster_labels = {
    0: "🌍 Global Investors",
    1: "🏠 First-Time Buyers",
    2: "🏢 Corporate Buyers",
    3: "💎 Luxury Investors"
}

cluster_colors = {
    "🌍 Global Investors": "#00d4ff",
    "🏠 First-Time Buyers": "#00ff88",
    "🏢 Corporate Buyers": "#ff9900",
    "💎 Luxury Investors": "#ff00ff"
}

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image(
    "https://img.icons8.com/emoji/96/house-emoji.png", width=80)
st.sidebar.title("🏠 Parcl Co.")
st.sidebar.markdown("**Buyer Segmentation System**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Home Dashboard",
    "👥 Buyer Segmentation",
    "🔬 Hierarchical Clustering",
    "📈 Investor Behavior",
    "🗺️ Geographic Analysis",
    "🔍 Segment Insights"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Filters")

selected_country = st.sidebar.selectbox(
    "Country", ["All"] + sorted(clients['country'].unique().tolist()))
selected_region = st.sidebar.selectbox(
    "Region", ["All"] + sorted(clients['region'].unique().tolist()))
selected_purpose = st.sidebar.selectbox(
    "Acquisition Purpose",
    ["All"] + clients['acquisition_purpose'].unique().tolist())
selected_type = st.sidebar.selectbox(
    "Client Type",
    ["All"] + clients['client_type'].unique().tolist())
n_clusters = st.sidebar.slider(
    "Number of Clusters", min_value=2, max_value=8, value=4)

# Apply filters
filtered = clients.copy()
if selected_country != "All":
    filtered = filtered[filtered['country'] == selected_country]
if selected_region != "All":
    filtered = filtered[filtered['region'] == selected_region]
if selected_purpose != "All":
    filtered = filtered[
        filtered['acquisition_purpose'] == selected_purpose]
if selected_type != "All":
    filtered = filtered[filtered['client_type'] == selected_type]

# Run clustering
(clustered_df, kmeans_sil, hier_sil,
 k_range, inertias, linkage_matrix, X_scaled) = run_clustering(n_clusters)

clustered_df['Cluster Label'] = clustered_df['KMeans_Cluster'].map(
    lambda x: cluster_labels.get(x, f"Cluster {x}"))
clustered_df['Hier Label'] = clustered_df['Hierarchical_Cluster'].map(
    lambda x: cluster_labels.get(x, f"Cluster {x}"))

# ============================================================
# PAGE 1 - HOME DASHBOARD
# ============================================================
if page == "🏠 Home Dashboard":
    st.markdown(
        '<p class="title-text">🏠 Parcl Buyer Segmentation System</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle-text">ML-Based Buyer Segmentation & Investment Profiling for Real Estate Market Intelligence</p>',
        unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Total Clients", f"{len(clients):,}")
    with col2:
        st.metric("🏢 Properties", f"{len(properties):,}")
    with col3:
        st.metric("🌍 Countries", clients['country'].nunique())
    with col4:
        st.metric("🎯 KMeans Silhouette", f"{kmeans_sil:.3f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👥 Client Type Distribution")
        ct = filtered['client_type'].value_counts().reset_index()
        ct.columns = ['Client Type', 'Count']
        fig = px.pie(ct, names='Client Type', values='Count',
                     template='plotly_dark',
                     color_discrete_sequence=['#00d4ff', '#ff9900'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Acquisition Purpose")
        ap = filtered['acquisition_purpose'].value_counts().reset_index()
        ap.columns = ['Purpose', 'Count']
        fig = px.bar(ap, x='Purpose', y='Count',
                     color='Count',
                     color_continuous_scale='blues',
                     template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💳 Loan Applied Distribution")
        loan = filtered['loan_applied'].value_counts().reset_index()
        loan.columns = ['Loan Applied', 'Count']
        fig = px.pie(loan, names='Loan Applied', values='Count',
                     template='plotly_dark',
                     color_discrete_sequence=['#00ff88', '#ff4444'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📣 Referral Channel")
        ref = filtered['referral_channel'].value_counts().reset_index()
        ref.columns = ['Channel', 'Count']
        fig = px.bar(ref, x='Channel', y='Count',
                     color='Count',
                     color_continuous_scale='teal',
                     template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2 - BUYER SEGMENTATION (KMEANS)
# ============================================================
elif page == "👥 Buyer Segmentation":
    st.title("👥 Buyer Segmentation — K-Means Clustering")
    st.markdown("K-Means clustering reveals hidden buyer segments")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Clusters", n_clusters)
    with col2:
        st.metric("📊 Silhouette Score", f"{kmeans_sil:.3f}")
    with col3:
        st.metric("👥 Total Buyers", f"{len(clustered_df):,}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Cluster Distribution")
        cluster_counts = clustered_df['Cluster Label'].value_counts().reset_index()
        cluster_counts.columns = ['Segment', 'Count']
        fig = px.pie(cluster_counts,
                     names='Segment', values='Count',
                     template='plotly_dark',
                     color_discrete_sequence=list(
                         cluster_colors.values()))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📉 Elbow Method")
        elbow_df = pd.DataFrame(
            {'K': k_range, 'Inertia': inertias})
        fig = px.line(elbow_df, x='K', y='Inertia',
                      markers=True,
                      template='plotly_dark',
                      title='Optimal Number of Clusters')
        fig.update_traces(line_color='#00d4ff')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Age Distribution by Segment")
        fig = px.box(clustered_df,
                     x='Cluster Label', y='age',
                     color='Cluster Label',
                     template='plotly_dark',
                     color_discrete_map=cluster_colors)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⭐ Satisfaction by Segment")
        fig = px.box(clustered_df,
                     x='Cluster Label',
                     y='satisfaction_score',
                     color='Cluster Label',
                     template='plotly_dark',
                     color_discrete_map=cluster_colors)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Segment Summary")
    summary = clustered_df.groupby('Cluster Label').agg(
        Count=('client_id', 'count'),
        Avg_Age=('age', 'mean'),
        Avg_Satisfaction=('satisfaction_score', 'mean'),
        Loan_Yes=('loan_applied',
                  lambda x: (x == 'Yes').sum()),
        Investment_Purpose=('acquisition_purpose',
                            lambda x: (x == 'Investment').sum())
    ).round(2).reset_index()
    st.dataframe(summary, use_container_width=True)

# ============================================================
# PAGE 3 - HIERARCHICAL CLUSTERING
# ============================================================
elif page == "🔬 Hierarchical Clustering":
    st.title("🔬 Hierarchical Clustering Analysis")
    st.markdown(
        "Reveals nested cluster relationships & validates K-Means")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 K-Means Silhouette", f"{kmeans_sil:.3f}")
    with col2:
        st.metric("🔬 Hierarchical Silhouette", f"{hier_sil:.3f}")
    with col3:
        better = "K-Means ✅" if kmeans_sil > hier_sil else "Hierarchical ✅"
        st.metric("🏆 Better Model", better)

    st.markdown("---")

    st.subheader("🌳 Dendrogram — Hierarchical Cluster Tree")
    st.markdown(
        "Shows how buyers are grouped — sample of 200 buyers")

    fig_dend, ax = plt.subplots(figsize=(12, 5))
    fig_dend.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    dendrogram(
        linkage_matrix, ax=ax,
        truncate_mode='level', p=5,
        color_threshold=0.7 * max(linkage_matrix[:, 2]),
        above_threshold_color='#888888'
    )
    ax.set_title('Hierarchical Clustering Dendrogram',
                 color='white', fontsize=14)
    ax.set_xlabel('Buyer Samples', color='white')
    ax.set_ylabel('Distance', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    st.pyplot(fig_dend)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Hierarchical Distribution")
        hier_counts = clustered_df['Hier Label'].value_counts(
        ).reset_index()
        hier_counts.columns = ['Segment', 'Count']
        fig = px.pie(hier_counts,
                     names='Segment', values='Count',
                     template='plotly_dark',
                     color_discrete_sequence=list(
                         cluster_colors.values()))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Model Quality Comparison")
        comparison_data = pd.DataFrame({
            'Model': ['K-Means', 'Hierarchical'],
            'Silhouette Score': [kmeans_sil, hier_sil]
        })
        fig = px.bar(comparison_data,
                     x='Model', y='Silhouette Score',
                     color='Model',
                     template='plotly_dark',
                     color_discrete_sequence=[
                         '#00d4ff', '#ff9900'],
                     title='Model Quality Comparison')
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 K-Means vs Hierarchical Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**K-Means Segments:**")
        kmeans_summary = clustered_df.groupby(
            'Cluster Label').agg(
            Count=('client_id', 'count'),
            Avg_Age=('age', 'mean'),
            Avg_Satisfaction=('satisfaction_score', 'mean')
        ).round(2).reset_index()
        st.dataframe(kmeans_summary, use_container_width=True)

    with col2:
        st.markdown("**Hierarchical Segments:**")
        hier_summary = clustered_df.groupby('Hier Label').agg(
            Count=('client_id', 'count'),
            Avg_Age=('age', 'mean'),
            Avg_Satisfaction=('satisfaction_score', 'mean')
        ).round(2).reset_index()
        st.dataframe(hier_summary, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Age Distribution — Hierarchical Segments")
    fig = px.box(clustered_df,
                 x='Hier Label', y='age',
                 color='Hier Label',
                 template='plotly_dark',
                 color_discrete_map=cluster_colors)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 4 - INVESTOR BEHAVIOR
# ============================================================
elif page == "📈 Investor Behavior":
    st.title("📈 Investor Behavior Dashboard")
    st.markdown(
        "Investment patterns and financing behavior by segment")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Acquisition Purpose by Segment")
        purpose_cluster = clustered_df.groupby(
            ['Cluster Label', 'acquisition_purpose']
        ).size().reset_index(name='Count')
        fig = px.bar(purpose_cluster,
                     x='Cluster Label', y='Count',
                     color='acquisition_purpose',
                     barmode='group',
                     template='plotly_dark',
                     color_discrete_sequence=[
                         '#00d4ff', '#ff9900'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💳 Loan Behavior by Segment")
        loan_cluster = clustered_df.groupby(
            ['Cluster Label', 'loan_applied']
        ).size().reset_index(name='Count')
        fig = px.bar(loan_cluster,
                     x='Cluster Label', y='Count',
                     color='loan_applied',
                     barmode='group',
                     template='plotly_dark',
                     color_discrete_sequence=[
                         '#00ff88', '#ff4444'])
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📣 Referral Channel by Segment")
        ref_cluster = clustered_df.groupby(
            ['Cluster Label', 'referral_channel']
        ).size().reset_index(name='Count')
        fig = px.bar(ref_cluster,
                     x='Cluster Label', y='Count',
                     color='referral_channel',
                     barmode='stack',
                     template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("👥 Client Type by Segment")
        type_cluster = clustered_df.groupby(
            ['Cluster Label', 'client_type']
        ).size().reset_index(name='Count')
        fig = px.bar(type_cluster,
                     x='Cluster Label', y='Count',
                     color='client_type',
                     barmode='group',
                     template='plotly_dark',
                     color_discrete_sequence=[
                         '#00d4ff', '#ff9900'])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🏠 Property Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Sale Price Distribution")
        fig = px.histogram(properties,
                           x='sale_price', nbins=30,
                           template='plotly_dark',
                           color_discrete_sequence=['#00d4ff'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏢 Unit Category Distribution")
        uc = properties['unit_category'].value_counts(
        ).reset_index()
        uc.columns = ['Category', 'Count']
        fig = px.pie(uc, names='Category', values='Count',
                     template='plotly_dark',
                     color_discrete_sequence=[
                         '#00d4ff', '#ff9900'])
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 5 - GEOGRAPHIC ANALYSIS
# ============================================================
elif page == "🗺️ Geographic Analysis":
    st.title("🗺️ Geographic Buyer Analysis")
    st.markdown("Buyer segments mapped by country and region")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌍 Buyers by Country")
        country_counts = filtered['country'].value_counts(
        ).reset_index()
        country_counts.columns = ['Country', 'Count']
        fig = px.bar(country_counts,
                     x='Country', y='Count',
                     color='Count',
                     color_continuous_scale='blues',
                     template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🗺️ Buyers by Country Map")
        fig = px.choropleth(country_counts,
                            locations='Country',
                            locationmode='country names',
                            color='Count',
                            color_continuous_scale='Blues',
                            template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Segments by Country")
        country_cluster = clustered_df.groupby(
            ['country', 'Cluster Label']
        ).size().reset_index(name='Count')
        fig = px.bar(country_cluster,
                     x='country', y='Count',
                     color='Cluster Label',
                     barmode='stack',
                     template='plotly_dark',
                     color_discrete_map=cluster_colors)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Purpose by Country")
        purpose_country = filtered.groupby(
            ['country', 'acquisition_purpose']
        ).size().reset_index(name='Count')
        fig = px.bar(purpose_country,
                     x='country', y='Count',
                     color='acquisition_purpose',
                     barmode='group',
                     template='plotly_dark',
                     color_discrete_sequence=[
                         '#00d4ff', '#ff9900'])
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Top 15 Regions by Buyer Count")
    top_regions = filtered['region'].value_counts().head(
        15).reset_index()
    top_regions.columns = ['Region', 'Count']
    fig = px.bar(top_regions,
                 x='Region', y='Count',
                 color='Count',
                 color_continuous_scale='teal',
                 template='plotly_dark')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 6 - SEGMENT INSIGHTS
# ============================================================
elif page == "🔍 Segment Insights":
    st.title("🔍 Segment Insights Panel")
    st.markdown(
        "Detailed descriptive statistics per buyer segment")
    st.markdown("---")

    selected_segment = st.selectbox(
        "Select Segment to Analyze",
        options=list(cluster_labels.values()))

    segment_df = clustered_df[
        clustered_df['Cluster Label'] == selected_segment]

    st.markdown(f"### {selected_segment}")
    st.markdown(
        f"**Total buyers in this segment: {len(segment_df):,}**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Age",
                  f"{segment_df['age'].mean():.0f} years")
    with col2:
        st.metric("Avg Satisfaction",
                  f"{segment_df['satisfaction_score'].mean():.1f}/5")
    with col3:
        loan_pct = (segment_df['loan_applied'] == 'Yes').mean() * 100
        st.metric("Loan Applied", f"{loan_pct:.0f}%")
    with col4:
        inv_pct = (segment_df['acquisition_purpose'] ==
                   'Investment').mean() * 100
        st.metric("Investment Purpose", f"{inv_pct:.0f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌍 Countries in Segment")
        seg_country = segment_df['country'].value_counts(
        ).reset_index()
        seg_country.columns = ['Country', 'Count']
        fig = px.pie(seg_country,
                     names='Country', values='Count',
                     template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📣 Referral Channels")
        seg_ref = segment_df['referral_channel'].value_counts(
        ).reset_index()
        seg_ref.columns = ['Channel', 'Count']
        fig = px.bar(seg_ref,
                     x='Channel', y='Count',
                     color='Count',
                     color_continuous_scale='blues',
                     template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👥 Gender Distribution")
        seg_gender = segment_df['gender'].value_counts(
        ).reset_index()
        seg_gender.columns = ['Gender', 'Count']
        fig = px.pie(seg_gender,
                     names='Gender', values='Count',
                     template='plotly_dark',
                     color_discrete_sequence=[
                         '#00d4ff', '#ff9900'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Acquisition Purpose")
        seg_purpose = segment_df['acquisition_purpose'].value_counts(
        ).reset_index()
        seg_purpose.columns = ['Purpose', 'Count']
        fig = px.bar(seg_purpose,
                     x='Purpose', y='Count',
                     color='Count',
                     color_continuous_scale='teal',
                     template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader(f"📋 Sample Buyers from {selected_segment}")
    st.dataframe(
        segment_df[['client_id', 'client_type', 'gender',
                    'country', 'region', 'age',
                    'acquisition_purpose', 'loan_applied',
                    'satisfaction_score',
                    'referral_channel']].head(10),
        use_container_width=True)
