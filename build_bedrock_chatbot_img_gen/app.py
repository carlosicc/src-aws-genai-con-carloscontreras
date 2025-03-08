# Image Workshop credits: https://catalog.workshops.aws/building-with-amazon-bedrock/en-US

import streamlit as st #all streamlit commands will be available through the "st" alias
import image_lib as glib #reference to local lib script

# Set the page width wider to accommodate columns
st.set_page_config(layout="wide", page_title="Image Generation") 

# Page title
st.title("📸Image Generation")

# Create 2 columns
col1, col2 = st.columns(2) 

# Everything in this with block will be placed in column 1
with col1: 
    st.subheader("Prompt") 
    
    # Display a multiline text box and Run Button
    prompt_text = st.text_area("Prompt text", height=200, label_visibility="collapsed") 
    process_button = st.button("Run", type="primary")

with col2:
    st.subheader("Result")
    
    if process_button: 
        with st.spinner("Drawing..."): 

            # Call the model through the supporting library
            generated_image = glib.get_image_response(prompt_content=prompt_text)
        
        st.image(generated_image) #display the generated image
