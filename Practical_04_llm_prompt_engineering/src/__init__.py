"""
src package
===========
Application code for the Prompt Engineering practical.

    variants.py -> task 1: zero-shot / few-shot / CoT prompt variants for a
                    classification task, PLUS the single shared low-level
                    Bedrock Converse call (call_bedrock) that roles.py reuses.
    roles.py     -> task 2: system-message persona comparison
                    (senior data analyst vs creative writer), built on top
                    of variants.call_bedrock rather than duplicating it.
"""
