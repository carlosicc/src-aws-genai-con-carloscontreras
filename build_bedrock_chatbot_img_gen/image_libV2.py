# Image Generation using Amazon Nova Canvas via Bedrock InvokeModel API
# Adapted from: https://catalog.workshops.aws/building-with-amazon-bedrock/en-US
# Docs: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
import boto3
import json
import base64
import random
from io import BytesIO

# SDK Init
session = boto3.Session()
bedrock = session.client(
    service_name='bedrock-runtime',
    region_name="us-east-1"
)

def get_image_response(prompt_content, bedrock_model_id="amazon.nova-canvas-v1:0"):
    """
    Generate an image from text using Amazon Nova Canvas via Bedrock InvokeModel API.

    :param prompt_content: the prompt with which to generate the image
    :param bedrock_model_id: the Bedrock model ID to use
    :return: a BytesIO object containing the generated image
    """

    # Build the request body using Nova Canvas native format
    seed = random.randint(0, 858993460)

    request_body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt_content
        },
        "imageGenerationConfig": {
            "seed": seed,
            "quality": "standard",
            "height": 512,
            "width": 512,
            "numberOfImages": 1
        }
    })

    # Call the Bedrock endpoint
    response = bedrock.invoke_model(body=request_body, modelId=bedrock_model_id)

    # Parse the response
    output_body = json.loads(response["body"].read())

    # Extract the base64-encoded image
    base64_image_data = output_body["images"][0]
    image_data = base64.b64decode(base64_image_data)

    # Return a BytesIO object for client app consumption
    return BytesIO(image_data)
