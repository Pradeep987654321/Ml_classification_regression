import streamlit as st
import pandas as pd
from pycaret.classification import ClassificationExperiment

# Set up page layout
st.set_page_config(layout="wide")
st.title("PyCaret Model Comparison App")

# Step 1: Upload the dataset (CSV)
st.write("### Upload your CSV file")
uploaded_file = st.file_uploader("", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Ensure 'Class variable' column exists
    if 'Class variable' not in data.columns:
        st.error("The dataset must contain a column named 'Class variable'.")
    else:
        # Step 2: Setup PyCaret Classification Experiment
        exp = ClassificationExperiment()
        exp.setup(data, target='Class variable', session_id=123)

        # Step 3: Compare models
        st.write("### Comparing Models...")
        best_model = exp.compare_models(n_select=2)

        # Step 4: Display the comparison metrics
        st.write("### Model Comparison Metrics")
        comparison_df = exp.pull()  # Pull the comparison metrics table
        st.write(comparison_df)

        # Step 5: Display the best model
        st.write("### Best Model")
        st.write(best_model)
else:
    st.write("Please upload a CSV file to proceed.")
