import boto3
import json

# Init the Bedrock client, and passing in the CLI profile
session = boto3.Session(profile_name="default")
bedrock_runtime = session.client(service_name='bedrock-runtime', region_name="us-west-2")

# Prompt
prompt_data = "Un hombre cambia la rueda de su bicicleta, después de un pinchazo"
body = json.dumps({"inputText": prompt_data})

# Using Titan Embeddings, as an ML embedding model  
modelId = "amazon.titan-embed-text-v2:0"
accept = "application/json"
contentType = "application/json"

# Call Amazon Bedrock
response = bedrock_runtime.invoke_model(
    body=body, modelId=modelId, accept=accept, contentType=contentType
)
response_body = json.loads(response.get("body").read())

# Show the Embedding Vector
embedding = response_body.get("embedding")
print(f"El embedding vector tiene {len(embedding)} elementos\n{embedding[0:3]+['...']+embedding[-3:]}")
