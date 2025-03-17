"""
@ Credits:
  Some sections of this Streamlit app have been obtained from: https://github.com/aws-samples/genai-quickstart-pocs
"""
import streamlit as st
import boto3
from utils import *
from invoke_model_converse_stream_api import stream_conversation

# Title displayed on the streamlit web app
st.title(f"""        🚀 :rainbow[ChatLab]""")

# configuring values for session state
# - See: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for default values
if 'region' not in st.session_state:
    st.session_state['region'] = 'us-west-2'
if 'provider' not in st.session_state:
    st.session_state['provider'] = 'Anthropic'


# writing the message that is stored in session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar
st.sidebar.title("Model Configuration")
with st.sidebar:
    st.write("-----")
    # Region selection
    regions = get_available_regions()
    selected_region = st.selectbox(
        "Select AWS Region",
        options=regions,
        index=regions.index(st.session_state['region'])
    )

    # Get all model summaries for the selected region
    # --> IMPORTANT!! We're limiting to Anthropic models, as some models may not support streaming or System Messages
    #                 Add your own logic to add more Providers; e.g. Amazon, Meta, AI21, etc.
    all_model_summaries = get_model_summaries(region=selected_region, provider='Anthropic')
    
    # Provider selection
    providers = get_unique_providers(all_model_summaries)
    selected_provider = st.selectbox(
        "Select Model Provider",
        options=providers,
        help="IMPORTANT: We're limiting to Anthropic models, as some models may not support streaming or System Messages. Add your own logic to add more Providers; e.g. Amazon, Meta, AI21, etc.",
        index=providers.index(st.session_state['provider']) if st.session_state['provider'] in providers else 0
    )

    # Filter models for selected provider
    provider_models = [
        model for model in filter_models(all_model_summaries)
        if model['providerName'] == selected_provider
        and model['responseStreamingSupported'] == True
    ]

    # Determine default model selection
    default_model_index = 0
    for idx, model in enumerate(provider_models):
        if 'haiku' in model['modelId'].lower():
            default_model_index = idx
            break

    # Model selection
    if provider_models:
        selected_model = st.selectbox(
            "Select Model",
            options=[model['modelName'] for model in provider_models],
            index=default_model_index
        )
        
        # Store the selected modelId
        selected_model_id = next(
            model['modelId'] for model in provider_models
            if model['modelName'] == selected_model
        )
    else:
        st.warning("No compatible models found for the selected provider.")
        selected_model_id = None

    # Create a Bedrock agent runtime client
    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=selected_region)

    # Temperature selection
    temperature = st.slider("**Temperature**", 0.1, 1.0, 0.5, step=0.1)
    
    # Top-K selection
    top_k = st.slider("**Top-K:**", 5, 250, 150, step=25)

    # System prompt as text input:
    st.markdown('### System Prompt:')
    system_prompt = st.text_area(
        "Introduce el system prompt aquí (opcional)",
        "Eres un asistente virtual amigable"
    )

# evaluating st.chat_input and determining if a question has been input
if question := st.chat_input("Message Claude"):

    # with the user icon, write the question to the front end
    with st.chat_message("user"):
        st.markdown(question)

    # append the question and the role (user) as a message to the session state
    st.session_state.messages.append({"role": "user",
                                      "content": question})
    
    # respond as the assistant with the answer
    with st.chat_message("assistant"):
        # making sure there are no messages present when generating the answer
        message_placeholder = st.empty()
        
        # calling the invoke_llm_with_streaming to generate the answer as a generator object, and using
        answer = st.write_stream(stream_conversation(bedrock_client=bedrock_client, 
                                                     question=question, 
                                                     system_prompt=system_prompt, 
                                                     input_model_id=selected_model_id, 
                                                     input_temperature=temperature, 
                                                     input_top_k=top_k))
    
    # appending the final answer to the session state
    st.session_state.messages.append({"role": "assistant",
                                      "content": answer})
