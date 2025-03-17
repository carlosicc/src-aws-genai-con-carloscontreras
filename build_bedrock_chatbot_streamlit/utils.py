import streamlit as st
import boto3
import re
from boto3.session import Session


def get_available_regions():
    """Get available AWS regions for Bedrock"""
    s = Session()
    return s.get_available_regions('bedrock')


def get_model_summaries(region, provider=None):
    """Get model summaries filtered by provider"""
    bedrock = boto3.client('bedrock', region_name=region)
    
    params = {
        'byInferenceType': 'ON_DEMAND',
        'byOutputModality': 'TEXT'
    }
    
    if provider:
        params['byProvider'] = provider
        
    try:
        response = bedrock.list_foundation_models(**params)
        return response['modelSummaries']
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []


def get_unique_providers(model_summaries):
    """Extract unique provider names from model summaries"""
    return sorted(list(set(model['providerName'] for model in model_summaries)))


def filter_models(model_summaries):
    """Filter models based on status and modelId format"""
    # Pattern matches ':' followed by any numbers followed by 'k'
    context_window_pattern = re.compile(r':\d+k$')
    
    return [
        model for model in model_summaries
        if (model['modelLifecycle']['status'] == 'ACTIVE' and
            not context_window_pattern.search(model['modelId']))
    ]

