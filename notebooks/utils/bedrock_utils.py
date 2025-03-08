# %%
import awswrangler as wr
import boto3
import textwrap
from loguru import logger

# %%
##############################################################################
# SQL PROMPTS
##############################################################################

SYSTEM_PROMPT_SQL = """
    You are an agent expert in building and running ANSI SQL statements, compatible with Amazon Athena or PrestoDB.
    """

BASE_PROMPT_TEMPLATE_SQL = """
    You are an agent expert in building and running ANSI SQL statements, compatible with Amazon Athena or PrestoDB.
    Reply on the following questions, by only returning the SQL statement and nothing else: 
    <questions>{questions}</questions>.
    <instructions> Instructions for answer: 
    The rules to create the SQL statements are:
        1.  Avoid fetching all rows. If user does not specify any aggregation, fetch all columns but use a LIMIT 10 to bring only 10 rows.
        1.1 Use standard SQL ANSI aggregations (e.g. Count(*), Group by). Do not use non-standard or non-ANSI SQL functions or syntax (e.g. RAND()), different than these: 'SELECT', 'FROM', 'WHERE', 'AND', 'IN', 'LIMIT'.
        2.  Follow these schema specs:
        2.1 The table name is "coffee_shop_sales". Do not use the Database name in the SQL statement.
        2.2 The columns to filter through the "WHERE" clause are the following four, showing in XML tags to specify the column names and their data types:
            <column_name>date</column_name><column_type>DATE</column_type>
            <column_name>cash_type</column_name><column_type>STRING</column_type>
            <column_name>money</column_name><column_type>DOUBLE</column_type>
            <column_name>coffee_name</column_name><column_type>STRING</column_type>
        3.  Do not create your own VALUES for the Filters below, at the WHERE clause. Instead, only use the following values:
        3.1 For column "cash_type", use values only the following values, separated by XML tags: 
            <cash_type>cash</cash_type>
            <cash_type>card</cash_type>
        3.2 For column "coffee_name", use only the following values, separated by XML tags: 
            <coffee_name>Espresso</coffee_name>
            <coffee_name>Cortado</coffee_name>
            <coffee_name>Latte</coffee_name>
            <coffee_name>Hot Chocolate</coffee_name>
            <coffee_name>Americano</coffee_name>
            <coffee_name>Americano with Milk</coffee_name>
            <coffee_name>Cocoa</coffee_name>
            <coffee_name>Cappuccino</coffee_name>
        3.4 For column 'date' you should use the following format: e.g. DATE('2024-12-23'), or use AWS Athena or MySQL date_parse functions. For example:
        <question>What is the total amount of money spent in the coffee shop between March 1st and March 4th, 2024?</question>
        <sql>
        select sum(money) 
        from coffee_shop_sales 
        where "date" between DATE('2024-03-01') and DATE('2024-03-04');
        </sql>
        
        Or a different example:
        
        <question>What is the total amount of money spent in the coffee shop between March 1st and March 31st, 2024, grouped by day and cash type?</question>
        <sql>
        SELECT "date" as "operation_day", cash_type, ROUND(SUM(money),2) as "sales_amount"
        FROM coffee_shop_sales 
        WHERE "date" BETWEEN DATE('2024-03-01') and DATE('2024-03-31')
        GROUP BY "date",cash_type
        ORDER BY 1,2;
        </sql>
        4. You will return nothing but the SQL statement only, ready to execute.
    </instructions> 
    Assistant:
    """

BASE_PROMPT_TEMPLATE_WITH_TOOLS_SQL = """
    You are an agent expert in building and running ANSI SQL statements.
    Your task is to:
    1. Generate the appropriate SQL query
    2. ALWAYS use the run_query tool to execute the query
    3. Never return the SQL query directly in your response

    Reply on the following questions: 
    <questions>{questions}</questions>.
    <instructions> Instructions for answer: 
    1. You should use the following tools to complete the user request:
    </instructions> 
    <tools_use_instructions>
    {tools_instructions}
    </tools_use_instructions>
    """


# Specify here what you need the LLM to call:
tools_instructions = """
    You must use the following tool to execute your SQL query:
    - run_query: This function executes the SQL query and returns the results
    Input: {"sql_query": "your SQL query here"}
    
    Never return the SQL query directly - always use the run_query tool to execute it.
    """

# %%
##############################################################################
# SQL Function Arguments for Function Calling feature
##############################################################################

# Configure names
table_name = 'coffee_shop_sales'

# Schema definition for tool. No schema for now; i.e. call the Lambda with no args.
get_tool_spec_sql = {
        "name": "run_query",
        "description": "Run SQL to get details about our Coffee Shop sales.",
        "inputSchema": {
             "json": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": f"Ansi SQL statement to query table {table_name}."
                    }
                },
                "required": ["sql_query"],  # Make sql_query required
            }
        }
    }


toolConfig = {
    'tools': [
        {
            'toolSpec': get_tool_spec_sql
        }
    ]
}

# %%
##############################################################################
# Adding Functions for TOOL Calling using SQL against Athena
##############################################################################

def run_query(query: str, 
              db_name: str = "db_coffee_shop_sales") -> None:
    """
    Generic function to run athena query and ensures it is successfully completed

    Parameters
    ----------
    query : str
        formatted string containing athena sql query
    results : str
        query output path
    
    Returns pandas dataframe
    """

    # Simple logging
    print(f"\n{'='*50}\nSQL Statement\n{'='*50}")
    print(query)

    # Run query
    response = wr.athena.read_sql_query(query,
                                        database=db_name,
                                        ctas_approach=False)
    
    return response


# %%
def process_tool_call(tool_name, tool_input):
    """_summary_

    Args:
        tool_name (_type_): _description_
        tool_input (_type_): _description_

    Returns:
        _type_: _description_
    """
    if tool_name == "run_query":
        # Pending to filter here for Lambda arguments
        print("calling run_query tool")
        
        return str(run_query(tool_input['sql_query']))
    
    else:
        logger.error(f"Tool {tool_name} not implemented")
        return None

# %%

def generate_conversation(bedrock_client,
                          model_id,
                          system_prompts,
                          messages,
                          tool_config={},
                          show_token_usage=False):
    """
    Sends messages to a model.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        system_prompts (JSON) : The system prompts for the model to use.
        messages (JSON) : The messages to send to the model.
        tool_config (JSON) : The tool_config containing tool specs

    Returns:
        response (JSON): The conversation that the model generated.

    """

    # Inference parameters to use.
    temperature = 0.0
    max_tokens = 4096
    top_k = 200

    # Base inference parameters to use.
    inference_config = {"temperature": temperature, "maxTokens": max_tokens}
    # Additional inference parameters to use.
    additional_model_fields = {"top_k": top_k}

    # Send the message.
    if tool_config == {}:
        print("No tools used in this conversation")
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields
        )
    else:
        print(f"Calling Conversational API with tools")
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
            toolConfig=tool_config,
            additionalModelRequestFields=additional_model_fields
        )

    # Log token usage.
    if show_token_usage:
        token_usage = response['usage']
        print(f"\n{'='*50}\nToken USage:\n{'='*50}")
        print(f"Input tokens: {token_usage['inputTokens']}")
        print(f"Output tokens: {token_usage['outputTokens']}")
        print(f"Total tokens: {token_usage['totalTokens']}")
        print(f"Stop reason: {response['stopReason']}")

    return response

# %%
def append_tool_result(user_query, message, tool_use, tool_result):
    """_summary_

    Args:
        user_query (_type_): _description_
        message (_type_): _description_
        tool_use (_type_): _description_
        tool_result (_type_): _description_

    Returns:
        _type_: _description_
    """
    messages = [
                {"role": "user", "content": [{"text": user_query}]},
                {"role": "assistant", "content": message['content']},
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": tool_use['toolUseId'],
                                "content": [{"text": tool_result}],
                            }
                        }
                    ],
                },
            ]
    return messages

# %%
def chat_with_claude_nl_to_sql(messages,
                               toolConfig=toolConfig,
                               system_prompts=[{"text": BASE_PROMPT_TEMPLATE_SQL}],
                               model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"):
    """_summary_

    Args:
        messages (_type_): _description_
        toolConfig (_type_): _description_
        system_prompts (list, optional): _description_. Defaults to [{"text": BASE_PROMPT_TEMPLATE_SQL}].

    Returns:
        _type_: _description_
    """
    # Bedrock settings
    bedrock_client = boto3.client(service_name='bedrock-runtime')
    
    # Get user question
    user_query = messages[0]['content'][0]['text']
    print(f"\n{'='*50}\nUser Message: {user_query}\n{'='*50}")
    
    # Send the message.
    converse_response = generate_conversation(bedrock_client, model_id, system_prompts, messages, toolConfig)

    # Log token usage.
    token_usage = converse_response['usage']
    print(f"\n{'='*50}\nToken Usage:\n{'='*50}")
    print(f"Input tokens: {token_usage['inputTokens']}")
    print(f"Output tokens: {token_usage['outputTokens']}")
    print(f"Total tokens: {token_usage['totalTokens']}")
    
    message = converse_response['output']['message']
    
    print(f"\n{'='*50}\nStep by step thinking:\n{'='*50}")
    print(f"\nInitial Response:")
    print(f"Stop Reason: {converse_response['stopReason']}")
    print(f"Content: {message['content'][0]['text']}")
    
    if converse_response['stopReason'] == "tool_use":
        using_tool  = True
    else:
        using_tool  = False
        
    while using_tool:
        if converse_response['stopReason'] == "tool_use":
            tool_use = next(block['toolUse'] for block in message['content'] if block.get('toolUse'))
            tool_name = tool_use["name"]
            tool_input = tool_use["input"]
    
            # print(f"\nTool Used: {tool_name}")
            # print(f"Tool Input: {tool_input}")
    
            tool_result = process_tool_call(tool_name, tool_input)

            print(f"\n{'='*50}\nTool Result(SQL output):\n{'='*50}")
            print(tool_result)
            print("="*50)
    
            messages = append_tool_result(user_query, message, tool_use, tool_result)
            converse_response = generate_conversation(bedrock_client, model_id, system_prompts, messages, toolConfig)

            # Log token usage.
            token_usage = converse_response['usage']
            
            print(f"\n{'='*50}\nTokens usage:\n{'='*50}")
            print(f"Input tokens: {token_usage['inputTokens']}")
            print(f"Output tokens: {token_usage['outputTokens']}")
            print(f"Total tokens: {token_usage['totalTokens']}")
            print(f"\n{'='*50}\n")
    
            message = converse_response['output']['message']
        else:
            response = message
            
        for block in message['content']:
            if block.get('toolUse'):
                using_tool = True
                break
            else:
                using_tool = False
    
    final_response = next(
        (block['text'] for block in message['content'] if block.get('text')),
        None,
    )

    return {
        "response": final_response
        }

# %%
def build_llm_query(questions, tools_instructions=None, examples=None):
    """_summary_

    Args:
        questions (_type_): _description_
        tools_instructions (_type_, optional): _description_. Defaults to None.
        examples (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # use base prompt by default
    prompt_template = BASE_PROMPT_TEMPLATE_SQL

    if tools_instructions:
        prompt_template = BASE_PROMPT_TEMPLATE_WITH_TOOLS_SQL

    messages = [{
        "role": "user",
        "content": [{"text": prompt_template.format(questions=questions,
                                                    tools_instructions=tools_instructions)
                     }]
    }]

    return messages


## Simple tools to format output
def format_pretty_output(text, width=70):
    # Split into paragraphs (preserve empty lines)
    paragraphs = text.split('\n\n')
    
    # Wrap each paragraph
    wrapped_paragraphs = []
    for paragraph in paragraphs:
        # Wrap the text while preserving sentence breaks
        wrapped = textwrap.fill(paragraph, width=width, 
                              break_long_words=False, 
                              break_on_hyphens=False)
        wrapped_paragraphs.append(wrapped)
    
    # Join paragraphs with double newlines
    return '\n\n'.join(wrapped_paragraphs)
