"""
src/embeddings/titan_embeddings.py
=====================================
Wraps LangChain's BedrockEmbeddings, configured for Amazon Titan
Embeddings.

WHY TITAN EMBEDDINGS, NOT A THIRD-PARTY EMBEDDING PROVIDER?
------------------------------------------------------------------
The practical brief specifies Titan Embeddings via Bedrock -- and
practically, it means the ENTIRE pipeline (Textract, embeddings, and
later an LLM if one's added) stays inside AWS/Bedrock with one IAM
permission model and one bill, rather than adding a second vendor
(OpenAI, Cohere, etc.) with its own API key and network dependency just
for this one step.

WHY THE V2 MODEL ID, NOT V1?
------------------------------
`amazon.titan-embed-text-v2:0` is Titan's newer embedding model --
improved retrieval quality over v1 and supports configurable output
dimensions (256/512/1024) if you ever need to trade a little accuracy
for a smaller, cheaper-to-store vector. Defaulting to v2 here rather than
v1 is simply "use the better current option" absent any specific reason
to pin to the older one.

WHY WRAP THIS IN A FUNCTION INSTEAD OF INSTANTIATING BedrockEmbeddings
DIRECTLY WHEREVER IT'S NEEDED?
------------------------------------------------------------------------
One place to configure model_id/region (from utils.config), and one
place to mock in tests -- both vectorstore/opensearch_store.py and any
notebook cell that needs embeddings call this same function instead of
each constructing their own BedrockEmbeddings instance.
"""

from langchain_aws import BedrockEmbeddings

from utils.config import settings


def get_titan_embeddings() -> BedrockEmbeddings:
    """Returns a configured BedrockEmbeddings instance. Not cached with
    @lru_cache (unlike this curriculum's boto3 client factories) because
    BedrockEmbeddings itself is a lightweight LangChain wrapper object,
    not something with meaningful construction cost to avoid repeating."""
    return BedrockEmbeddings(model_id=settings.embedding_model_id, region_name=settings.aws_region)
