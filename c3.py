import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.anomaly import AnomalyExperiment

# Set up page layout
st.set_page_config(layout="wide")
with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


st.title("Machine Learning-Classification,Regression Tool")
# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {background-color: #f5f5f5;}
    .css-10trblm {font-size: 32px; font-weight: 700; color: #4A90E2; text-align: center;}
    .stButton > button {background-color: #4A90E2; color: white; font-size: 16px; padding: 10px 24px; border-radius: 8px; border: none; transition: background-color 0.3s ease;}
    .stButton > button:hover {background-color: #357ABD;}
    .css-1s2e8r0 h2 {color: #357ABD; font-weight: bold; border-bottom: 2px solid #357ABD; padding-bottom: 5px;}
    .css-1a3v6xh {background-color: #f0f0f0; border: 2px dashed #4A90E2; color: #4A90E2;}
    .stDataFrame {border: 1px solid #ddd; border-radius: 8px; overflow: hidden;}
    .stProgress > div > div > div {background-color: #4A90E2;}
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
#st.title('Machine Learning Tool- Classification,Regression')

# Step 1: Select analysis type
st.write('<p style="font-size:24px; font-weight:bold;">Choose Analysis Type</p>', unsafe_allow_html=True)
analysis_type = st.selectbox("Select Analysis Type", ['Classification', 'Regression'])

# Step 2: Upload the dataset (CSV)
st.write('<p style="font-size:24px; font-weight:bold;">Upload your CSV file[Note:must contain a column named "Class variable"[Predicted Column]]</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=['csv'])

if uploaded_file is not None:
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.text("Loading dataset...")
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
        
    st.table(data=data.head())
    progress_bar.progress(20)
    
    # Ensure 'Class variable' column exists
    if 'Class variable' not in data.columns:
        st.error("The dataset must contain a column named 'Class variable'.")
    else:
        # Determine PyCaret experiment based on analysis type
        if analysis_type == 'Classification':
            exp = ClassificationExperiment()
            exp.setup(data, target='Class variable', session_id=123)
        
        elif analysis_type == 'Regression':
            exp = RegressionExperiment()
            exp.setup(data, target='Class variable', session_id=123)
        
    
        progress_bar.progress(40)
        
        # Step 3: Model selection
        st.subheader("Model Performance Metrics")
        status_text.text("Comparing models...")
        
        all_models = exp.compare_models(n_select=20)  # List top 20 models
        metrics = exp.pull()  # Pull the comparison metrics table
        st.table(data=metrics)
        progress_bar.progress(60)

       
