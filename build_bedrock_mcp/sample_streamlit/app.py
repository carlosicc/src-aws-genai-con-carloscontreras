import streamlit as st
import boto3
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client

# Configuration
region = boto3.session.Session().region_name

NOVA_PRO_MODEL_ID = "us.amazon.nova-pro-v1:0"
if region.startswith("eu"):
    NOVA_PRO_MODEL_ID = "eu.amazon.nova-pro-v1:0"
elif region.startswith("ap"):
    NOVA_PRO_MODEL_ID = "apac.amazon.nova-pro-v1:0"

print(f"Using Nova Pro Model ID: {NOVA_PRO_MODEL_ID}")

st.title("MCP Trading Agent Demo")
st.write("Streamlit app with real MCP server integration using Strands Agent")

# Configuration section
st.sidebar.header("Configuration")
mcp_url = st.sidebar.text_input("MCP Server URL", value="http://localhost:8000/mcp")
model_id = st.sidebar.text_input("Bedrock Model ID", value=NOVA_PRO_MODEL_ID)

# Demo questions
st.header("Demo Questions")
demo_questions = [
    "Execute a trade for 1000 shares of AMZN at $150.25",
    "Send trade details for trade ID T12345",
    "What is the status of my recent trades?",
    "Execute a trade for 500 shares of TSLA at $200.50 and then send the details"
]

selected_question = st.selectbox("Select a demo question:", [""] + demo_questions)
custom_question = st.text_area("Or enter your own question:")

question = custom_question if custom_question else selected_question

# Function to create and run agent
def run_trading_agent(question, mcp_url, model_id):
    """Connect to MCP server and run the strands agent with the question"""
    try:
        st.info("🔄 Connecting to MCP Server...")
        
        # Connect to the MCP server
        trading_server = MCPClient(lambda: streamablehttp_client(mcp_url))
        
        with trading_server:
            # Get available tools
            mcp_tools = trading_server.list_tools_sync()
            tool_names = [tool.tool_name for tool in mcp_tools]
            
            st.success(f"✅ Connected! Available tools: {tool_names}")
            
            # Create agent with MCP tools
            agent = Agent(
                model=BedrockModel(model_id=model_id),
                system_prompt="""You are a trading assistant that can execute trades and send trade details. 
                Always provide clear confirmation of actions taken and include relevant details like trade IDs, 
                timestamps, and status information.""",
                tools=mcp_tools,
            )
            
            st.info("🤖 Processing question with agent...")
            
            # Run the agent with the question
            response = agent(question)
            
            return response, tool_names
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, []

# Connection test
st.subheader("MCP Server Status")
if st.button("Test Connection"):
    try:
        trading_server = MCPClient(lambda: streamablehttp_client(mcp_url))
        with trading_server:
            mcp_tools = trading_server.list_tools_sync()
            tool_names = [tool.tool_name for tool in mcp_tools]
            st.success(f"✅ Connected to MCP Server")
            st.info(f"Available tools: {tool_names}")
    except Exception as e:
        st.error(f"❌ Connection failed: {str(e)}")

# Main interaction area
st.subheader("Agent Response")

if st.button("Submit Question") and question:
    with st.spinner("Running Strands Agent..."):
        st.write("**Question:**", question)
        
        # Run the actual strands agent
        response, tools = run_trading_agent(question, mcp_url, model_id)
        
        if response:
            st.success("🎉 Agent completed successfully!")
            
            # Extract and display the text response
            st.subheader("Agent Response:")
            try:
                # Extract text from AgentResult object
                if hasattr(response, 'message') and 'content' in response.message:
                    content = response.message['content']
                    if isinstance(content, list) and len(content) > 0:
                        text_response = content[0].get('text', str(response))
                        st.markdown(text_response)
                    else:
                        st.write(response)
                else:
                    st.write(response)
            except Exception as e:
                st.warning(f"Could not parse response format: {e}")
                st.write(response)
            
        else:
            st.error("Failed to get response from agent")
