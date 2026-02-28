# This Image Workshop and code credits: https://catalog.workshops.aws/building-with-amazon-bedrock/en-US
import boto3
import json
import base64
import random
from io import BytesIO

# Models we know how to call — display name → model ID.
# Add new entries here when support for a new model is implemented below.
SUPPORTED_MODELS = {
    'Amazon Nova Canvas':                    'amazon.nova-canvas-v1:0',
    'Stability AI - Stable Image Ultra':     'stability.stable-image-ultra-v1:1',
}


def get_image_models(region):
    """Return a {display_name: model_id} dict of supported image models
    that are ACTIVE in the given region.
    """
    try:
        bedrock = boto3.client('bedrock', region_name=region)
        response = bedrock.list_foundation_models(byOutputModality='IMAGE')
        available_ids = {
            m['modelId'] for m in response['modelSummaries']
            if m['modelLifecycle']['status'] == 'ACTIVE'
        }
        return {
            name: model_id
            for name, model_id in SUPPORTED_MODELS.items()
            if model_id in available_ids
        }
    except Exception:
        return {}


def get_image_response(prompt_content, bedrock_client, model_id):
    """
    Generate an image from text using the Bedrock InvokeModel API.
    Supports Stability AI and Amazon Nova Canvas models.

    :param prompt_content: the prompt with which to generate the image
    :param bedrock_client: a boto3 bedrock-runtime client (region already set by caller)
    :param model_id: the Bedrock model ID to use
    :return: a BytesIO object containing the generated image
    """
    if model_id.startswith('amazon.nova-canvas'):
        # Nova Canvas native request format
        # Docs: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
        request_body = json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt_content
            },
            "imageGenerationConfig": {
                "seed": random.randint(0, 858993460),
                "quality": "standard",
                "height": 768,
                "width": 768,
                "numberOfImages": 1
            }
        })
    elif model_id.startswith('stability.'):
        # Stability AI request format
        request_body = json.dumps({'prompt': prompt_content})
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    response = bedrock_client.invoke_model(body=request_body, modelId=model_id)
    output_body = json.loads(response["body"].read())
    image_data = base64.b64decode(output_body["images"][0])
    return BytesIO(image_data)
