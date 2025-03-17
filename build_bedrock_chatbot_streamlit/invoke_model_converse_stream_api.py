"""
@ Credits:
  Some sections of this Streamlit app have been obtained from: https://github.com/aws-samples/genai-quickstart-pocs
"""
import json
import os


def stream_conversation(bedrock_client, question, system_prompt, input_model_id, input_temperature, input_top_k, messages=[], pricing_list='bedrock_pricing.json'):
    """
    Sends messages to a model and streams back the response.
    Args:
        messages: A list of messages to send to the model that helps preserve context along with the latest message.
        input_model_id: The ID of the model to use for the conversation.
        
    Returns:
        Nothing.
    """

    # Set the temperature for the model inference, controlling the randomness of the responses.
    inference_config = {"temperature": input_temperature}
    
    # Set the top_k parameter for the model inference, determining how many of the top predictions to consider.
    additional_model_fields = {"top_k": input_top_k}
    
    # Define the system prompts to guide the model's behavior, and set the general direction of the models role.
    system_prompts = [{"text": system_prompt}]
    
    # Format the user's message as a dictionary with role and content
    message = {
        "role": "user",
        "content": [{"text": question}]
    }
    
    # Append the formatted user message to the list of messages.
    messages.append(message)

    response = bedrock_client.converse_stream(
        modelId=input_model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )

    stream = response.get('stream')
    
    # Looping through the response from the converse_stream api call
    if stream:
        
        # create a variable that will be used to store the streaming content so that we can later append it to the messages
        streaming_text = ""
        
        for event in stream:

            if 'messageStart' in event:
                print(f"\nRole: {event['messageStart']['role']}")
                
            if 'contentBlockDelta' in event:
                # using a generator object to stream the text to the streamlit front end.
                yield event['contentBlockDelta']['delta']['text']
                
                # Add the streaming chunks to our place holder
                streaming_text += event['contentBlockDelta']['delta']['text']

            if 'messageStop' in event:
                print(f"\nStop reason: {event['messageStop']['stopReason']}")
                
                # Construct the message for the next conversation turn
                message = {
                    "role": "assistant",
                    "content": [{"text": streaming_text}]
                }
                
                messages.append(message)

            if 'metadata' in event:
                # Print somme information regarging input and output tokesns as well as latency in ms
                metadata = event['metadata']
                print('#'*100)

                if 'usage' in metadata:
                    print("\nToken usage")
                    print(f"Input tokens: {metadata['usage']['inputTokens']}")
                    print(f"Output tokens: {metadata['usage']['outputTokens']}")

                    # Fetch pricing info
                    pricing_dir = os.path.dirname(os.path.abspath(__file__))
                    pricing_list = os.path.join(pricing_dir, 'bedrock_pricing.json')
                    
                    with open(pricing_list,'r', encoding='utf-8') as f:
                        pricing_file = json.load(f)

                    # Find matching model in pricing file
                    matching_model = None
                    for price_model_id in pricing_file.keys():
                        if input_model_id.startswith(price_model_id):
                            matching_model = price_model_id
                            break

                    if matching_model:
                        # Estimate cost of call
                        print(f"Model: {input_model_id}, at temperature {input_temperature} and Top-K of {input_top_k}")
                        print("""
                              \nImportant: confirm pricing is up-to-date at https://aws.amazon.com/bedrock/pricing/)
                                and update bedrock_pricing.json accordingly.
                              """)
                        print(f"Price per 1,000 input tokens: {pricing_file[matching_model]['input']*1000:.5f}")
                        print(f"Price per 1,000 output tokens: {pricing_file[matching_model]['output']*1000:.5f}")
                        cost_input_tokens = float(metadata['usage']['inputTokens']) * pricing_file[matching_model]['input']
                        cost_output_tokens = float(metadata['usage']['outputTokens']) * pricing_file[matching_model]['output']
                        total_cost = round(cost_input_tokens + cost_output_tokens,16)

                        # Print estimated cost
                        print(f"\nTotal tokens in session: {metadata['usage']['totalTokens']}. Estimated cost: ${total_cost:.10f}")
                    else:
                        print(f"\nWarning: Model '{input_model_id}' is not included in the pricing file. Please update bedrock_pricing.json with current pricing information.")
                        print(f"Total tokens in session: {metadata['usage']['totalTokens']}. Cost estimation not available.")

                if 'metrics' in event['metadata']:
                    print(
                        f"\nLatency: {metadata['metrics']['latencyMs']} milliseconds\n")
                
                print('\n'.join([str(d) for d in messages]))
                print('#'*100)
