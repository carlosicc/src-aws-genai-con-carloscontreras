"""
@ Credits:
  Some sections of this Streamlit app have been obtained from: https://github.com/aws-samples/genai-quickstart-pocs
"""
import json
import os


def stream_conversation(bedrock_client, question, system_prompt, input_model_id, input_temperature, input_top_k, messages=[], pricing_data=None):
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
    
    # Define the system prompts to guide the model's behavior, and set the general direction of the models role.
    system_prompts = [{"text": system_prompt}]

    # Format the user's message as a dictionary with role and content
    message = {
        "role": "user",
        "content": [{"text": question}]
    }

    # Append the formatted user message to the list of messages.
    messages.append(message)

    # Build the API call arguments
    # Note: Amazon Nova models do not accept additionalModelRequestFields via the Converse API.
    #       top_k is only passed for Anthropic models (as "top_k" in additionalModelRequestFields).
    call_kwargs = dict(
        modelId=input_model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
    )
    if 'amazon.' not in input_model_id:
        call_kwargs['additionalModelRequestFields'] = {"top_k": input_top_k}

    response = bedrock_client.converse_stream(**call_kwargs)

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

                    pricing_file = pricing_data or {}

                    # Strip cross-region inference profile prefix (e.g. "us.", "eu.", "ap.", "global.")
                    # so that pricing keys like "amazon.nova-micro-v1" match IDs like "us.amazon.nova-micro-v1:0"
                    import re
                    pricing_model_id = re.sub(r'^(us|eu|ap|global)\.', '', input_model_id)

                    # Find the most specific (longest) matching prefix.
                    # Longest-match avoids e.g. 'claude-opus-4' swallowing 'claude-opus-4-5'.
                    matching_model = None
                    for price_model_id in pricing_file.keys():
                        if pricing_model_id.startswith(price_model_id):
                            if matching_model is None or len(price_model_id) > len(matching_model):
                                matching_model = price_model_id

                    if matching_model:
                        # Estimate cost of call
                        print(f"Model: {input_model_id}, at temperature {input_temperature} and Top-K of {input_top_k}")
                        print(f"Note: prices shown are US East (N. Virginia) on-demand — https://aws.amazon.com/bedrock/pricing/")
                        print(f"Price per 1,000 input tokens: {pricing_file[matching_model]['input']*1000:.5f}")
                        print(f"Price per 1,000 output tokens: {pricing_file[matching_model]['output']*1000:.5f}")
                        cost_input_tokens = float(metadata['usage']['inputTokens']) * pricing_file[matching_model]['input']
                        cost_output_tokens = float(metadata['usage']['outputTokens']) * pricing_file[matching_model]['output']
                        total_cost = round(cost_input_tokens + cost_output_tokens,16)

                        # Print estimated cost
                        print(f"\nTotal tokens in session: {metadata['usage']['totalTokens']}. Estimated cost: ${total_cost:.10f}")
                    else:
                        print(f"\nWarning: Pricing not available for '{input_model_id}'. It may not yet be indexed in the AWS Pricing API.")
                        print(f"Total tokens in session: {metadata['usage']['totalTokens']}. Cost estimation not available.")

                if 'metrics' in event['metadata']:
                    print(
                        f"\nLatency: {metadata['metrics']['latencyMs']} milliseconds\n")
                
                print('\n'.join([str(d) for d in messages]))
                print('#'*100)
