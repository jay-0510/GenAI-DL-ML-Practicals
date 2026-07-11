"""
src package
===========
Application code for the AWS Bedrock "Hello World" practical.

Module responsibilities (kept deliberately separate — single responsibility
per file, so each is independently testable and independently readable):

    bedrock_client.py      -> builds & caches boto3 clients (connections only)
    list_bedrock_models.py -> discovery: what models exist? (control plane)
    converse.py             -> inference: send a prompt, get a reply (data plane)
"""
