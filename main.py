import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

# --- Page Config & Styling ---
st.set_page_config(
    page_title="Cyber-Sec Data Quality Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Cyber-Dark Custom CSS
st.markdown("""
<style>
    :root {
        --cyber-green: #00ff66;
        --cyber-blue: #00ccff;
        --cyber-dark: #0a0a12;
        --cyber-alert: #ff0055;
    }
    
    /* Background and typography */
    .stApp {
        background-color: var(--cyber-dark);
        color: #e0e0e0;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Titles and Headers */
    h1, h2, h3 {
        color: var(--cyber-blue) !important;
        text-shadow: 0 0 10px rgba(0, 204, 255, 0.3);
        font-weight: 700;
    }
    
    /* Upload Box */
    .stFileUploader > div > div > div > div {
        background-color: rgba(0, 204, 255, 0.05);
        border: 1px dashed var(--cyber-blue);
        border-radius: 8px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--cyber-green) !important;
        text-shadow: 0 0 8px rgba(0, 255, 102, 0.3);
    }
    
    /* Dataframes/Tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #1f1f3a;
    }
    
    /* Success/Error Alerts */
    .stAlert {
        border-left: 4px solid;
    }
    .st-emotion-cache-12m21k8 { /* Success */
        border-color: var(--cyber-green);
        background-color: rgba(0, 255, 102, 0.05);
    }
    .st-emotion-cache-16k1w6a { /* Error */
        border-color: var(--cyber-alert);
        background-color: rgba(255, 0, 85, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def detect_pii(df):
    """Detects potential PII / Security Leakage (IPs, Emails)."""
    pii_report = {}
    ip_pattern = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
    email_pattern = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check a sample to save time, or all if dataset is small
            sample = df[col].dropna().astype(str).head(1000)
            if sample.str.match(ip_pattern).any():
                pii_report[col] = "Potential IPv4 Addresses Detected"
            elif sample.str.match(email_pattern).any():
                pii_report[col] = "Potential Email Addresses Detected"
    return pii_report


# --- Main App ---
st.title("🛡️ Cyber-Sec Data Quality Auditor")
st.markdown("_Automated Data Quality Engineering & ML Readiness Reporting for Cybersecurity Data_")

# Sidebar
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Security Log (CSV/Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
        st.sidebar.metric("Total Rows", f"{len(df):,}")
        st.sidebar.metric("Total Columns", f"{len(df.columns):,}")
        
        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧹 Data Quality Engineering", "🤖 ML Readiness & Leakage"])
        
        # TAB 1: Overview
        with tab1:
            st.subheader("Dataset Snapshot")
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Data Types")
                st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={'index': 'Column', 0: 'Type'}), use_container_width=True)
            with col2:
                st.subheader("Statistical Summary")
                st.dataframe(df.describe(), use_container_width=True)
                
        # TAB 2: Data Quality Engineering
        with tab2:
            st.header("Data Quality Checks")
            
            # 1. Missing Values
            st.subheader("1. Missing Values (Nulls)")
            missing_stats = df.isnull().sum()
            missing_percent = (missing_stats / len(df)) * 100
            missing_df = pd.DataFrame({'Missing Count': missing_stats, 'Percentage (%)': missing_percent})
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
            
            if not missing_df.empty:
                col_m1, col_m2 = st.columns([1, 2])
                with col_m1:
                    st.warning(f"Found missing values in {len(missing_df)} columns.")
                    st.dataframe(missing_df.style.format({'Percentage (%)': '{:.2f}%'}), use_container_width=True)
                with col_m2:
                    st.write("**Missing Value Heatmap**")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax)
                    fig.patch.set_facecolor('#0a0a12')
                    ax.tick_params(colors='white')
                    st.pyplot(fig)
            else:
                st.success("✅ No missing values detected! Perfect completeness.")

            st.divider()

            # 2. Duplicates
            st.subheader("2. Duplicate Records")
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                st.warning(f"⚠️ Found {duplicate_count:,} completely duplicate rows ({(duplicate_count/len(df))*100:.2f}% of data).")
                st.write("In cybersecurity datasets (like network flows), some exact duplicates *can* be legitimate repeated traffic, but large numbers often indicate data collection errors.")
                if st.button("Show Sample of Duplicates"):
                    st.dataframe(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head(10), use_container_width=True)
            else:
                st.success("✅ No duplicate records found. Data is unique.")
                
        # TAB 3: ML Readiness & Leakage
        with tab3:
            st.header("Machine Learning Readiness")
            
            # Target Selection
            st.markdown("Select your **Target Label** (e.g., 'attack_type', 'label', 'is_malicious') to analyze class imbalance and target leakage.")
            target_col = st.selectbox("Target Variable", options=['<Select Target>'] + list(df.columns))
            
            if target_col != '<Select Target>':
                st.divider()
                
                # 1. Class Imbalance
                st.subheader("⚖️ Class Imbalance Analyzer")
                class_counts = df[target_col].value_counts()
                class_percentages = df[target_col].value_counts(normalize=True) * 100
                
                col_i1, col_i2 = st.columns([1, 2])
                with col_i1:
                    st.dataframe(pd.DataFrame({'Count': class_counts, 'Percentage': class_percentages}).style.format({'Percentage': '{:.2f}%'}))
                    
                    # Imbalance logic
                    max_pct = class_percentages.iloc[0]
                    if max_pct > 90:
                        st.error("🚨 **Extreme Imbalance Detected:** The majority class dominates >90% of the dataset. ML models will likely overfit to this class. Consider SMOTE, ADASYN, or under-sampling.")
                    elif max_pct > 75:
                        st.warning("⚠️ **Moderate Imbalance:** The majority class is quite large. Models might struggle with minority attacks. Consider cost-sensitive learning.")
                    else:
                        st.success("✅ **Balanced:** The classes are reasonably balanced.")
                
                with col_i2:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, palette="mako")
                    fig.patch.set_facecolor('#0a0a12')
                    ax.tick_params(colors='white')
                    ax.set_ylabel("Count", color='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    st.pyplot(fig)
                
                st.divider()
                
                # 2. Target Leakage (Correlation)
                st.subheader("🎯 Target Leakage Scanner")
                st.markdown("Identifies features that are highly correlated with the target. While high correlation is good for predictive power, a correlation near 1.0 often indicates **Target Leakage** (a feature that won't be available at prediction time in the real world).")
                
                # Convert target to numeric temporarily if it's categorical for correlation
                df_corr = df.copy()
                if df_corr[target_col].dtype == 'object':
                    df_corr[target_col] = df_corr[target_col].astype('category').cat.codes
                
                # Select only numeric columns for correlation
                numeric_cols = df_corr.select_dtypes(include=[np.number]).columns
                if target_col in df_corr.columns and target_col not in numeric_cols:
                    # If target is string but we converted it, ensure it's in the list
                    numeric_cols = list(numeric_cols) + [target_col]
                    
                if len(numeric_cols) > 1:
                    correlations = df_corr[numeric_cols].corr()[target_col].sort_values(ascending=False).drop(target_col, errors='ignore')
                    
                    high_corr = correlations[abs(correlations) > 0.85]
                    if not high_corr.empty:
                        st.error(f"🚨 **Potential Target Leakage:** Found features with > 0.85 correlation to '{target_col}'.")
                        st.dataframe(pd.DataFrame({'Correlation': high_corr}).style.background_gradient(cmap='Reds'))
                    else:
                        st.success("✅ No obvious target leakage detected (all numeric correlations < 0.85).")
                else:
                    st.info("Not enough numeric columns to calculate target correlation.")
            
            st.divider()
            
            # 3. Security PII Leakage
            st.subheader("🕵️ Security Data Leakage (PII Scanner)")
            st.markdown("Scans text columns for potential IP addresses or Emails that shouldn't be exposed in public ML datasets.")
            
            with st.spinner("Scanning for PII..."):
                pii_results = detect_pii(df)
            
            if pii_results:
                st.error("🚨 **Potential Sensitive Data Found!**")
                for col, issue in pii_results.items():
                    st.markdown(f"- **{col}**: {issue}")
            else:
                st.success("✅ No obvious plaintext IPs or Emails detected in string columns.")

else:
    # Landing page instructions
    st.info("Please upload a dataset using the sidebar to begin the audit.")
    st.markdown("""
    ### Why use this Auditor?
    In cybersecurity, ML models are only as good as their training data. Bad data leads to false positives (alert fatigue) and false negatives (breaches).
    
    **This tool automatically checks for:**
    1. 🛡️ **Missing Values**: Which logs dropped packets?
    2. 🛡️ **Duplicates**: Are we over-representing specific attacks?
    3. ⚖️ **Imbalance**: Are normal flows drowning out the 1% of malicious traffic?
    4. 🕵️ **Leakage**: Are we leaking the answer (Target Leakage) or leaking PII (Security Leakage)?
    """)
