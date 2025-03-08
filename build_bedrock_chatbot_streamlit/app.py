"""
@ Credits:
  Some sections of this Streamlit app have been obtained from: https://github.com/aws-samples/genai-quickstart-pocs
"""
import streamlit as st
import boto3
from invoke_model_converse_stream_api import stream_conversation

# Title displayed on the streamlit web app
st.title(f"""        🚀 :rainbow[ChatLab]""")

# configuring values for session state
# - See: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# writing the message that is stored in session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar
st.sidebar.title("Model Configuration")
with st.sidebar:
    st.write("-----")

    aws_regions = [
        "us-west-2",       # US West (Oregon) // Default on chatbot screen
        "us-east-2",       # US East (Ohio)
        "us-east-1",       # US East (N. Virginia)
        "us-west-1",       # US West (N. California)
        "af-south-1",      # Africa (Cape Town)
        "ap-east-1",       # Asia Pacific (Hong Kong)
        "ap-south-2",      # Asia Pacific (Hyderabad)
        "ap-southeast-3",  # Asia Pacific (Jakarta)
        "ap-southeast-4",  # Asia Pacific (Melbourne)
        "ap-south-1",      # Asia Pacific (Mumbai)
        "ap-northeast-3",  # Asia Pacific (Osaka)
        "ap-northeast-2",  # Asia Pacific (Seoul)
        "ap-southeast-1",  # Asia Pacific (Singapore)
        "ap-southeast-2",  # Asia Pacific (Sydney)
        "ap-northeast-1",  # Asia Pacific (Tokyo)
        "ca-central-1",    # Canada (Central)
        "eu-central-1",    # Europe (Frankfurt)
        "eu-west-1",       # Europe (Ireland)
        "eu-west-2",       # Europe (London)
        "eu-south-1",      # Europe (Milan)
        "eu-west-3",       # Europe (Paris)
        "eu-south-2",      # Europe (Spain)
        "eu-north-1",      # Europe (Stockholm)
        "eu-central-2",    # Europe (Zurich)
        "me-south-1",      # Middle East (Bahrain)
        "me-central-1",    # Middle East (UAE)
        "sa-east-1"        # South America (São Paulo)
    ]

    # Create a Bedrock agent runtime client
    aws_region=st.selectbox('**AWS Region - Confirm Bedrock support**', aws_regions)
    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=aws_region)
    
    # Model Selection
    models=[
        'anthropic.claude-3-haiku-20240307-v1:0',
        'anthropic.claude-3-5-sonnet-20240620-v1:0',
        'anthropic.claude-3-sonnet-20240229-v1:0',
        'anthropic.claude-instant-v1'
        ]
    model=st.selectbox('**Model**', models)

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
        answer = st.write_stream(stream_conversation(bedrock_client, question, system_prompt, model, temperature, top_k))
    
    # appending the final answer to the session state
    st.session_state.messages.append({"role": "assistant",
                                      "content": answer})
