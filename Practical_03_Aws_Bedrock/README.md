# Practical 3 — AWS Bedrock Hello World

Minimal, modular, tested project that:

1. Lists all AWS Bedrock foundation models available in the account/region.
2. Invokes a model (NovaMicro or NovaLite) via Bedrock's **Converse API**.
3. Experiments with `temperature` and `top_p` and documents how output changes.

---

## Project structure

```
practical_03_aws_bedrock/
├── README.md
├── requirements.txt
├── notebook/
│   └── bedrock_hello_world.ipynb   # narrated, end-to-end walkthrough of all 3 tasks
├── src/
│   ├── __init__.py
│   ├── bedrock_client.py           # builds/caches boto3 clients (the ONLY file that calls boto3.client)
│   ├── list_bedrock_models.py      # task 1: list foundation models (control plane)
│   └── converse.py                 # task 2 & 3: invoke a model via Converse (data plane)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # shared pytest fixtures (mocked AWS clients)
│   ├── test_bedrock_client.py
│   └── test_converse.py
└── utils/
    ├── __init__.py
    └── config.py                   # single source of truth for region / model / retry settings
```

### Design principle

Each file has **one job**, so each is independently testable:

| File | Responsibility | NOT responsible for |
|---|---|---|
| `utils/config.py` | Load settings (region, model id, defaults) from env vars | Making any AWS call |
| `src/bedrock_client.py` | Build & cache boto3 clients | Any business logic (listing, inference) |
| `src/list_bedrock_models.py` | Discovery — what models exist | Running inference |
| `src/converse.py` | Inference — send a prompt, get a reply | Client construction, listing models |
| `tests/conftest.py` | Shared fake AWS clients for all tests | Any real network call |

---

## What do `bedrock_client.py` and `conftest.py` actually do?

**`src/bedrock_client.py`** is the single place in the whole project that calls `boto3.client(...)`.
Bedrock is really two separate AWS services glued under one name:

- `"bedrock"` — the **control plane**: list/describe models, check model access.
- `"bedrock-runtime"` — the **data plane**: actually run inference (Converse, InvokeModel).

This file exposes two cached functions, `get_bedrock_client()` and `get_bedrock_runtime_client()`,
that every other module calls instead of building their own client. That gives us:
- one place to configure region/retries consistently,
- one place to catch "no AWS credentials" and turn it into a clear error message,
- and — most importantly for testing — one narrow seam to mock, instead of patching `boto3.client`
  in every file that happens to need AWS.

**`tests/conftest.py`** is a special pytest filename that's auto-loaded before any test runs.
Fixtures defined there (like `mock_bedrock_runtime_client`) become available to *every* test file
in `tests/` with no import needed. Here it holds:
- a fake Bedrock Runtime client with a pre-programmed `.converse()` response, and
- a fake Bedrock control-plane client with a pre-programmed `.list_foundation_models()` response,
- plus an `autouse` fixture that clears `bedrock_client.py`'s `@lru_cache` before every test, so
  tests don't leak cached clients into each other.

This is what lets `pytest` run the whole suite **with zero AWS credentials, zero cost, and
deterministic results** — no test ever touches the real network.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure AWS credentials
```bash
aws configure
# AWS Access Key ID:      <provided key>
# AWS Secret Access Key:  <provided secret>
# Default region name:    us-east-1        # or wherever Bedrock is enabled
# Default output format:  json
```
boto3 automatically discovers credentials configured this way — nothing is hardcoded in code.

### 3. (Optional) override defaults via `.env`
Create a `.env` file in the project root if you want non-default settings:
```
AWS_REGION=us-east-1
BEDROCK_MODEL_ID="amazon.nova-lite-v1:0"
BEDROCK_MAX_TOKENS=512
BEDROCK_TEMPERATURE=0.5
BEDROCK_TOP_P=0.9
```

---

## Usage

### Run the notebook (recommended — walks through all 3 tasks with explanations)
```bash
cd notebook
jupyter notebook bedrock_hello_world.ipynb
```

### Or run the scripts directly
```bash
# Task 1: list models
python -m src.list_bedrock_models

# Task 2/3: invoke a model
python -m src.converse
```

### Run the tests
```bash
pytest tests/ -v
```
All tests run offline against mocked AWS clients — no credentials required.

---

## Task 3 summary: temperature vs top_p

| Parameter | What it controls | Low value | High value |
|---|---|---|---|
| `temperature` | How sharply peaked the next-token probability distribution is | Deterministic, focused, repeatable output — good for facts/code | Varied, creative, sometimes less coherent — good for brainstorming |
| `top_p` | Size of the candidate token pool sampled from (nucleus sampling) | Narrow pool of only the most-likely words — safe, conventional phrasing | Larger pool — more lexical variety, more surprising word choices |

They control randomness through different mechanisms, so they should be varied independently —
see the notebook's Section 3 for actual side-by-side runs and observations.

---

## Troubleshooting

- **`AccessDeniedException` / "you don't have access to the model"** — go to the Bedrock console
  → **Model access** page and request access to the specific model ID. This is a one-time,
  per-account approval step separate from IAM permissions.
- **`NoCredentialsError`** — run `aws configure`, or set `AWS_ACCESS_KEY_ID` /
  `AWS_SECRET_ACCESS_KEY` environment variables.
- **`ThrottlingException`** — the client already retries with adaptive backoff
  (`utils/config.py` → `max_retry_attempts`); if it persists, you're likely on a
  rate-limited trial account — wait and retry, or reduce request frequency.
