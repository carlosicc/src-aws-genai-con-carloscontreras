# Image Workshop credits: https://catalog.workshops.aws/building-with-amazon-bedrock/en-US

import streamlit as st
import boto3
from boto3.session import Session
import image_lib as glib

# Set the page width wider to accommodate columns
st.set_page_config(layout="wide", page_title="Image Generation")

# Page title
st.title("📸Image Generation")

# Sidebar
st.sidebar.title("Model Configuration")
with st.sidebar:
    st.write("-----")

    # Region selection
    regions = Session().get_available_regions('bedrock')
    selected_region = st.selectbox(
        "Select AWS Region",
        options=regions,
        index=regions.index('us-west-2')
    )

    # Model selection — only show models available in the selected region
    available_models = glib.get_image_models(selected_region)
    if available_models:
        selected_model_name = st.selectbox("Select Model", options=list(available_models.keys()))
        selected_model_id = available_models[selected_model_name]
    else:
        st.warning("No supported image models available in this region.")
        selected_model_id = None
    st.info("**Model availability:**\n\n"
            "- **Stability AI** — us-west-2 only\n"
            "- **Nova Canvas** — ap-northeast-1, eu-west-1, us-east-1")

    # Create the Bedrock client for the selected region
    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=selected_region)

# Create 2 columns
col1, col2 = st.columns(2)

# Everything in this with block will be placed in column 1
with col1:
    st.subheader("Prompt text:")
    st.markdown("---")

    # Display a multiline text box and Run Button
    prompt_text = st.text_area("Prompt text", height=200, label_visibility="collapsed")
    process_button = st.button("Run", type="primary")

with col2:
    st.subheader("Result")

    if process_button and selected_model_id:
        with st.spinner("Drawing..."):

            # Call the model through the supporting library
            generated_image = glib.get_image_response(
                prompt_content=prompt_text,
                bedrock_client=bedrock_client,
                model_id=selected_model_id
            )

        st.image(generated_image)  # display the generated image
