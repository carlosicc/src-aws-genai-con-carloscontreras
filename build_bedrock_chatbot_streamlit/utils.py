import streamlit as st
import boto3
import re
from boto3.session import Session


# ---------------------------------------------------------------------------
# Pricing fallback for models NOT yet indexed in the AWS Pricing API.
# All prices are per token (USD).  Divide the $/1M figure from the AWS page
# by 1,000,000 to get the per-token value.
#
# Region: US East (N. Virginia) on-demand — used for all cost estimates.
# Source: https://aws.amazon.com/bedrock/pricing/
#
# Keys are model-ID prefixes (most-specific first — avoids prefix collisions).
# Replace None with the actual price once confirmed on the pricing page.
# ---------------------------------------------------------------------------
BEDROCK_PRICING_FALLBACK = {
    # -----------------------------------------------------------------------
    # Amazon Nova — Standard Tier, US East (N. Virginia) on-demand
    # Current models are also auto-fetched from the AWS Pricing API (entries
    # here act as a static backup and reference).
    # Add new model families when they appear in the inference profiles list.
    # Source: https://aws.amazon.com/bedrock/pricing/
    # -----------------------------------------------------------------------
    'amazon.nova-2-lite':    {'input': 0.00000033,  'output': 0.00000275},  # $0.33  / $2.75  per 1M
    'amazon.nova-micro':     {'input': 0.000000035, 'output': 0.00000014},  # $0.035 / $0.14  per 1M
    'amazon.nova-lite':      {'input': 0.00000006,  'output': 0.00000024},  # $0.06  / $0.24  per 1M
    'amazon.nova-pro':       {'input': 0.0000008,   'output': 0.0000032},   # $0.80  / $3.20  per 1M
    'amazon.nova-premier':   {'input': 0.0000025,   'output': 0.0000125},   # $2.50  / $12.50 per 1M
    # Nova Pro latency-optimized: $1.00 / $4.00 per 1M — not yet in inference
    # profiles; uncomment if it appears: 'amazon.nova-pro-latency': {'input': 0.000001, 'output': 0.000004}
    # Future models — uncomment and fill in when released:
    # 'amazon.nova-2-pro':   {'input': None, 'output': None},
    # 'amazon.nova-3-lite':  {'input': None, 'output': None},

    # -----------------------------------------------------------------------
    # Anthropic Claude — Global Cross-region Inference, on-demand
    # Claude 3 Haiku / Sonnet / Opus are auto-fetched from the AWS Pricing API.
    # All entries below are NOT yet in the API.
    # Prices per token = $/1M ÷ 1,000,000.
    # Source: https://aws.amazon.com/bedrock/pricing/
    # -----------------------------------------------------------------------
    # Claude 3.5 / 3.7
    'anthropic.claude-3-5-haiku':    {'input': 0.0000008,  'output': 0.000004},   # $0.80  / $4.00  per 1M
    'anthropic.claude-3-5-sonnet':   {'input': 0.000003,   'output': 0.000015},   # $3.00  / $15.00 per 1M
    'anthropic.claude-3-7-sonnet':   {'input': 0.000003,   'output': 0.000015},   # $3.00  / $15.00 per 1M
    # Claude 4.x — most-specific keys first to avoid prefix collisions
    'anthropic.claude-haiku-4-5':    {'input': 0.000001,   'output': 0.000005},   # $1.00  / $5.00  per 1M
    'anthropic.claude-sonnet-4-6':   {'input': 0.000003,   'output': 0.000015},   # $3.00  / $15.00 per 1M
    'anthropic.claude-sonnet-4-5':   {'input': 0.000003,   'output': 0.000015},   # $3.00  / $15.00 per 1M
    'anthropic.claude-sonnet-4':     {'input': 0.000003,   'output': 0.000015},   # $3.00  / $15.00 per 1M
    'anthropic.claude-opus-4-6':     {'input': 0.000005,   'output': 0.000025},   # $5.00  / $25.00 per 1M
    'anthropic.claude-opus-4-5':     {'input': 0.000005,   'output': 0.000025},   # $5.00  / $25.00 per 1M
    'anthropic.claude-opus-4-1':     {'input': None, 'output': None},             # TODO
    'anthropic.claude-opus-4':       {'input': None, 'output': None},             # TODO
}


def _keep_latest_per_family(profiles, family_keywords, max_per_family=2):
    """Return only the `max_per_family` newest profiles for each model family.

    Recency is determined by the `createdAt` datetime returned by
    list_inference_profiles(). Each `family_keywords` entry is matched as a
    case-insensitive substring of the inferenceProfileId.
    """
    result = []
    for keyword in family_keywords:
        family = [p for p in profiles if keyword in p['inferenceProfileId'].lower()]
        family.sort(key=lambda p: p.get('createdAt', ''), reverse=True)
        result.extend(family[:max_per_family])
    return result


def get_anthropic_models(region):
    """Fetch available Anthropic inference profiles for the given region.
    Uses list_inference_profiles() so only profiles active in the region are returned.
    Global-prefix duplicates are excluded and only the 2 latest per model family
    are kept (haiku / sonnet / opus) to avoid legacy models that have been disabled.
    """
    try:
        bedrock = boto3.client('bedrock', region_name=region)
        response = bedrock.list_inference_profiles()
        anthropic_profiles = [
            p for p in response['inferenceProfileSummaries']
            if 'anthropic' in p['inferenceProfileId']
            and not p['inferenceProfileId'].startswith('global.')
            and p['status'] == 'ACTIVE'
        ]
        anthropic_profiles = _keep_latest_per_family(
            anthropic_profiles, ['haiku', 'sonnet', 'opus']
        )
        return [
            {
                "modelId": p['inferenceProfileId'],
                "modelName": re.sub(r'(Claude) (\d+(?:\.\d+)?) (\w+)', r'\1 \3 \2', p['inferenceProfileName']),
                "providerName": "Anthropic",
                "responseStreamingSupported": True,
                "modelLifecycle": {"status": "ACTIVE"},
            }
            for p in anthropic_profiles
        ]
    except Exception as e:
        st.error(f"Error fetching Anthropic inference profiles: {str(e)}")
        return []


def get_nova_models(region):
    """Fetch available Amazon Nova text inference profiles for the given region.
    Uses list_inference_profiles() so only profiles active in the region are returned.
    Global-prefix duplicates are excluded and only the 2 latest per model family
    are kept (micro / lite / pro / premier) to avoid disabled legacy versions.
    """
    try:
        bedrock = boto3.client('bedrock', region_name=region)
        response = bedrock.list_inference_profiles()
        nova_profiles = [
            p for p in response['inferenceProfileSummaries']
            if 'amazon.nova' in p['inferenceProfileId']
            and not p['inferenceProfileId'].startswith('global.')
            and p['status'] == 'ACTIVE'
        ]
        nova_profiles = _keep_latest_per_family(
            nova_profiles, ['micro', 'lite', 'pro', 'premier']
        )
        return [
            {
                "modelId": p['inferenceProfileId'],
                "modelName": p['inferenceProfileName'],
                "providerName": "Amazon",
                "responseStreamingSupported": True,
                "modelLifecycle": {"status": "ACTIVE"},
            }
            for p in nova_profiles
        ]
    except Exception as e:
        st.error(f"Error fetching Nova inference profiles: {str(e)}")
        return []


def get_default_model_index(provider_models, keyword):
    """Return the index of the most recent model whose ID contains keyword.

    Recency is determined by:
      1. An 8-digit date (YYYYMMDD) embedded in the model ID  →  e.g. claude-haiku-4-5-20251001
      2. A generational version prefix before the keyword     →  e.g. nova-2-lite (v2 > v1)
    Falls back to index 0 if no match is found.
    """
    candidates = [
        (idx, m) for idx, m in enumerate(provider_models)
        if keyword in m['modelId'].lower()
    ]
    if not candidates:
        return 0

    def sort_key(item):
        model_id = item[1]['modelId']
        date_match = re.search(r'(\d{8})', model_id)
        if date_match:
            return int(date_match.group(1))
        # Nova-style: nova-2-lite → 2, nova-lite → 1
        ver_match = re.search(r'nova-(\d+)-', model_id)
        return int(ver_match.group(1)) if ver_match else 1

    return max(candidates, key=sort_key)[0]


@st.cache_data(ttl=3600)
def get_bedrock_pricing():
    """Fetch standard on-demand per-token pricing from the AWS Pricing API.

    Cached for 1 hour so the API is not called on every Streamlit rerun.
    Returns: {model_id_prefix: {"input": price_per_token, "output": price_per_token}}

    Note: the AWS Pricing API only indexes a subset of Bedrock models.
    Newer Anthropic models (3.5+, 4.x) are not yet in the API; pricing for
    those will be unavailable and logged as a warning in the terminal.
    """
    # Maps pricing API model names → base model ID prefix (no region prefix, no version suffix)
    # Only model names live here — prices are fetched from the API dynamically.
    MODEL_NAME_MAP = {
        'Nova Micro':     'amazon.nova-micro',
        'Nova Lite':      'amazon.nova-lite',
        'Nova Pro':       'amazon.nova-pro',
        'Nova Premier':   'amazon.nova-premier',
        'Nova 2.0 Lite':  'amazon.nova-2-lite',
        'Claude 3 Haiku': 'anthropic.claude-3-haiku',
        'Claude 3 Sonnet':'anthropic.claude-3-sonnet',
        'Claude 3 Opus':  'anthropic.claude-3-opus',
    }
    try:
        # Pricing API is a global service — always query us-east-1
        pricing_client = boto3.client('pricing', region_name='us-east-1')
        result = {}

        for inference_type in ['Input tokens', 'Output tokens']:
            paginator = pricing_client.get_paginator('get_products')
            for page in paginator.paginate(
                ServiceCode='AmazonBedrock',
                Filters=[{'Type': 'TERM_MATCH', 'Field': 'inferenceType', 'Value': inference_type}]
            ):
                import json as _json
                for price_item_str in page.get('PriceList', []):
                    price_item = _json.loads(price_item_str)
                    model_name = price_item.get('product', {}).get('attributes', {}).get('model')
                    if model_name not in MODEL_NAME_MAP:
                        continue
                    model_prefix = MODEL_NAME_MAP[model_name]
                    for term in price_item.get('terms', {}).get('OnDemand', {}).values():
                        for dim in term.get('priceDimensions', {}).values():
                            price_per_token = float(dim['pricePerUnit']['USD']) / 1000
                            key = 'input' if 'Input' in inference_type else 'output'
                            if model_prefix not in result:
                                result[model_prefix] = {}
                            # Keep the maximum price = standard on-demand (batch pricing is always lower)
                            if key not in result[model_prefix] or price_per_token > result[model_prefix][key]:
                                result[model_prefix][key] = price_per_token

        # Anthropic output tokens are not indexed in the pricing API.
        # All current Claude models use a 5:1 output:input ratio.
        for prefix, prices in result.items():
            if 'anthropic' in prefix and 'input' in prices and 'output' not in prices:
                prices['output'] = prices['input'] * 5

        # Merge fallback: fills in models not covered by the API.
        # Entries with None prices are intentional placeholders — skip them so
        # the "pricing not available" warning is shown for those models.
        for prefix, prices in BEDROCK_PRICING_FALLBACK.items():
            if prefix not in result and prices['input'] is not None and prices['output'] is not None:
                result[prefix] = prices

        return result
    except Exception as e:
        st.error(f"Could not fetch Bedrock pricing data: {e}")
        return {}


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

