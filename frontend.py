import streamlit as st
import requests
import io
import pandas as pd
from io import StringIO

# Set page config at the very beginning
st.set_page_config(page_title="💲")

# Backend URL (FastAPI backend running on Render)
backend_url = "https://anomaly-detection-using-credit-card.onrender.com"

# Set up a simple gradient background for the app
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display title for the app in the sidebar (Dollar sign)
# st.sidebar.title("💲")

# Display header
st.title("Credit Card Transactions Fraud Detection")

# File upload for training
st.subheader("Upload Training CSV File")
train_file = st.file_uploader("Choose the training file", type="csv")

if train_file is not None:
    # Send file to the backend to train the model
    files = {'file': train_file}
    response = requests.post(f"{backend_url}/train/", files=files)
    if response.status_code == 200:
        st.success("Training completed successfully!")
    else:
        st.error(f"Training failed with status code {response.status_code}")

# File upload for prediction
st.subheader("Upload CSV File for Prediction")
predict_file = st.file_uploader("Choose the prediction file", type="csv")

if predict_file is not None:
    # Send file to the backend to make predictions
    files = {'file': predict_file}
    response = requests.post(f"{backend_url}/predict", files=files)

    if response.status_code == 200:
        # Display predictions
        st.success("Predictions generated successfully!")

        # Convert the predictions CSV data to a Pandas DataFrame
        prediction_data = pd.read_csv(io.StringIO(response.text))
        st.subheader("Predictions")
        st.write(prediction_data)

        # Provide download option for predictions
        csv = prediction_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        # Show plot (Decision Tree)
        st.subheader("Decision Tree Plot")
        plot_response = requests.post(f"{backend_url}/plot/")
        if plot_response.status_code == 200:
            # Display the image of the decision tree
            st.image(plot_response.content, caption="Decision Tree Plot", use_container_width=True)

            # Provide download option for the plot
            st.download_button(
                label="Download Decision Tree Plot",
                data=plot_response.content,
                file_name="plot.png",
                mime="image/png"
            )
        else:
            st.error(f"Failed to fetch the plot with status code {plot_response.status_code}")
