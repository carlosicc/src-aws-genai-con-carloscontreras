import streamlit as st
import boto3
from utils import *
from invoke_model_converse_stream_api import stream_conversation

# Page configuration
st.set_page_config(
    page_title="ChatLab",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Added chat container positioning
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        font-size: 3rem !important;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    .stSelectbox {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    .stSlider {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    /* New styles for chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 100px);
    }
    .chat-messages {
        flex-grow: 1;
        overflow-y: auto;
        padding-bottom: 100px;
    }
    .chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: white;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

# Main title with animation
st.title("🚀 ChatLab")
st.markdown("<div style='text-align: center; color: #666; margin-bottom: 2rem;'>Your AI Conversation Partner</div>", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'region' not in st.session_state:
    st.session_state['region'] = 'us-west-2'
if 'provider' not in st.session_state:
    st.session_state['provider'] = 'Anthropic'

# Enhanced sidebar
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    st.markdown("---")
    
    # Region selection with better styling
    regions = get_available_regions()
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_region = st.selectbox(
            label="AWS Region Selection",  # Added explicit label
            options=regions,
            index=regions.index(st.session_state['region'])
        )

    # Get all model summaries for the selected region
    # --> IMPORTANT!! We're limiting to Anthropic models, as some models may not support streaming or System Messages
    #                 Add your own logic to add more Providers; e.g. Amazon, Meta, AI21, etc.
    all_model_summaries = get_model_summaries(region=selected_region, provider='Anthropic')
    providers = get_unique_providers(all_model_summaries)
    
    st.markdown("### 🤖 Model Selection")
    selected_provider = st.selectbox(
        label="Model Provider Selection",  # Added explicit label
        options=providers,
        help="IMPORTANT: We're limiting to Anthropic models, as some models may not support streaming or System Messages. Add your own logic to add more Providers; e.g. Amazon, Meta, AI21, etc.",
        index=providers.index(st.session_state['provider']) if st.session_state['provider'] in providers else 0
    )

    provider_models = [
        model for model in filter_models(all_model_summaries)
        if model['providerName'] == selected_provider
        and model['responseStreamingSupported'] == True
    ]

    default_model_index = next((idx for idx, model in enumerate(provider_models) 
                              if 'haiku' in model['modelId'].lower()), 0)

    if provider_models:
        selected_model = st.selectbox(
            label="Model Selection",  # Added explicit label
            options=[model['modelName'] for model in provider_models],
            index=default_model_index
        )
        selected_model_id = next(
            model['modelId'] for model in provider_models
            if model['modelName'] == selected_model
        )
    else:
        st.error("❌ No compatible models found")
        selected_model_id = None

    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=selected_region)

    # Parameters section with better organization
    st.markdown("### 🎛️ Parameters")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.5, step=0.1,
                          help="Controls randomness in responses")
    top_k = st.slider("Top-K", 5, 250, 150, step=25,
                      help="Number of tokens to consider for sampling")

    # System prompt with better styling
    st.markdown("### 💭 System Prompt")
    system_prompt = st.text_area(
        "",
        "Eres un asistente virtual amigable",
        height=100
    )

# Main chat area
messages_container = st.container()

# Display existing messages
with messages_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input at the bottom
if question := st.chat_input("Type your message here..."):
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        answer = st.write_stream(stream_conversation(
            bedrock_client=bedrock_client,
            question=question,
            system_prompt=system_prompt,
            input_model_id=selected_model_id,
            input_temperature=temperature,
            input_top_k=top_k
        ))
    
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Auto-scroll to bottom (JavaScript)
st.markdown("""
    <script>
        var element = document.getElementById('chat-container');
        element.scrollTop = element.scrollHeight;
    </script>
    """, unsafe_allow_html=True)
