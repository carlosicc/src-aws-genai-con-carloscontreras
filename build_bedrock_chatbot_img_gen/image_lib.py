# This Image Workshop and code credits: https://catalog.workshops.aws/building-with-amazon-bedrock/en-US
import os
import boto3
import json
import base64
from io import BytesIO

# SDK Init
session = boto3.Session() 
bedrock = session.client(
    service_name='bedrock-runtime',
    region_name="us-west-2"
) 

def get_image_response(prompt_content, bedrock_model_id = "stability.stable-image-ultra-v1:1"):
    """
    Generate an image from text using the Bedrock Stable Diffusion model

    :param prompt_content: the prompt with which to generate the image
    :return: a BytesIO object containing the image
    """

    # Prompts & how closely the model tries to match the prompt
    request_body = json.dumps({'prompt': prompt_content})
    
    # Call the Bedrock endpoint
    response = bedrock.invoke_model(body=request_body, modelId=bedrock_model_id)
    
    # load the response body into a json object
    output_body = json.loads(response["body"].read().decode("utf-8"))
    base64_output_image = output_body["images"][0]
    image_data = base64.b64decode(base64_output_image)

    #return a BytesIO object for client app consumption
    return BytesIO(image_data) 
