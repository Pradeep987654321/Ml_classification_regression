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

st.title("Machine Learning-Classification, Regression Tool")

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

# Step 1: Select analysis type
st.write('<p style="font-size:24px; font-weight:bold;">Choose Analysis Type</p>', unsafe_allow_html=True)
analysis_type = st.selectbox("Select Analysis Type", ['Classification', 'Regression'])

# Step 2: Upload the dataset (CSV)
st.write('<p style="font-size:24px; font-weight:bold;">Upload your CSV file [Note: must contain a column named "Class variable"]</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=['csv'])

@st.cache_data
def load_and_setup_model(data, target_column, analysis_type):
    if analysis_type == 'Classification':
        exp = ClassificationExperiment()
    elif analysis_type == 'Regression':
        exp = RegressionExperiment()
    
    exp.setup(data, target=target_column, session_id=123, silent=True)
    return exp

if uploaded_file is not None:
    status_text = st.empty()
    progress_bar = st.progress(0)

    status_text.text("Loading dataset...")
    try:
        data = pd.read_csv(uploaded_file)
        st.table(data.head())
        progress_bar.progress(20)
        
        if 'Class variable' not in data.columns:
            st.error("The dataset must contain a column named 'Class variable'.")
        else:
            exp = load_and_setup_model(data, 'Class variable', analysis_type)
            progress_bar.progress(40)

            st.subheader("Model Performance Metrics")
            status_text.text("Comparing models...")

            all_models = exp.compare_models(n_select=5)  # Compare only top 5 models to reduce load
            metrics = exp.pull()
            st.table(data=metrics)
            progress_bar.progress(60)

            st.subheader("Choose the Best Model")
            model_options = [str(model).split('(')[0] for model in all_models]
            st.write('<p style="font-size:24px; font-weight:bold;">Choose the model you want to use</p>', unsafe_allow_html=True)
            chosen_model = st.selectbox("", model_options)
            selected_model = all_models[model_options.index(chosen_model)]
            progress_bar.progress(70)

            st.subheader(f"Evaluating the Model: {chosen_model}")
            status_text.text("Evaluating the selected model...")
            exp.evaluate_model(selected_model)
            progress_bar.progress(80)

            st.subheader("Predictions on Holdout Set")
            status_text.text("Generating predictions on holdout set...")
            holdout_pred = exp.predict_model(selected_model)
            st.table(data=holdout_pred.head())
            progress_bar.progress(90)

            st.write('<p style="font-size:24px; font-weight:bold;">Download Predictions CSV</p>', unsafe_allow_html=True)
            holdout_pred.to_csv('predictions.csv', index=False)
            with open('predictions.csv', 'rb') as f:
                st.download_button('Download Predictions', f, file_name='predictions.csv')

            st.subheader("Original vs Predicted Comparison")
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=range(len(data['Class variable'].values[:50])), y=data['Class variable'].values[:50], label='Original', marker='o')
            sns.lineplot(x=range(len(holdout_pred['prediction_label'].values[:50])), y=holdout_pred['prediction_label'].values[:50], label='Predicted', marker='x')
            plt.legend()
            plt.title('Original vs Predicted - Comparison')
            st.pyplot(plt)
            progress_bar.progress(100)
            status_text.text("Analysis complete.")
            progress_bar.empty()

            st.write('<p style="font-size:24px; font-weight:bold;">Download Comparison Chart</p>', unsafe_allow_html=True)
            plt.savefig('comparison_chart.png')
            with open('comparison_chart.png', 'rb') as f:
                st.download_button('Download Chart', f, file_name='comparison_chart.png')

            st.write('<p style="font-size:24px; font-weight:bold;">Upload Test Dataset</p>', unsafe_allow_html=True)
            test_file = st.file_uploader("Upload test dataset", type=['csv', 'xlsx'])

            if test_file is not None:
                status_text.text("Loading test dataset...")
                test_data = pd.read_csv(test_file) if test_file.name.endswith('.csv') else pd.read_excel(test_file)
                st.table(test_data.head())

                st.subheader("Predictions on New Test Dataset")
                status_text.text("Generating predictions...")
                test_predictions = exp.predict_model(selected_model, data=test_data)
                st.table(test_predictions.head())

                st.write('<p style="font-size:24px; font-weight:bold;">Download Predictions</p>', unsafe_allow_html=True)
                test_predictions.to_csv('test_predictions.csv', index=False)
                with open('test_predictions.csv', 'rb') as f:
                    st.download_button('Download Predictions', f, file_name='test_predictions.csv')

            if st.button("Save the Trained Model"):
                status_text.text("Saving the trained model...")
                progress_bar = st.progress(0)
                
                exp.save_model(selected_model, 'trained_model')
                progress_bar.progress(80)
                
                with open('trained_model.pkl', 'rb') as f:
                    st.download_button('Download Model', f, file_name='trained_model.pkl')
                progress_bar.progress(100)
                status_text.text("Model saved successfully.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file to proceed.")
