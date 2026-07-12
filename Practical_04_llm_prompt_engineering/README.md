# Practical 4 — Prompt Engineering Experiments

Modular, tested project that:

1. Compares **zero-shot, few-shot, and chain-of-thought (CoT)** prompting on
   the same classification task (HR helpdesk ticket → category), via
   boto3 + Bedrock's Converse API.
2. Compares **system-message personas** — "senior data analyst" vs
   "creative writer" — on the same user prompt, documenting tone and
   quality differences.

---

## Project structure

```
practical_04_prompt_engineering/
├── README.md
├── requirements.txt
├── notebook/
│   └── prompt_engineering.ipynb   # narrated, end-to-end walkthrough of both tasks
├── src/
│   ├── __init__.py
│   ├── variants.py                 # task 1: zero-shot/few-shot/CoT prompt builders
│   │                                #         + the single shared Bedrock call function
│   └── roles.py                    # task 2: persona/system-message comparison
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # shared pytest fixtures (mocked Bedrock client)
│   ├── test_variants.py
│   └── test_roles.py
└── utils/
    ├── __init__.py
    └── config.py                   # settings + the one Bedrock client factory
```

### Design principle

| File | Responsibility |
|---|---|
| `utils/config.py` | Settings (region, model id, defaults) + the single `get_bedrock_runtime_client()` factory |
| `src/variants.py` | Task 1's three prompt-building techniques, **and** `call_bedrock()` — the one function that actually calls `client.converse(...)` |
| `src/roles.py` | Task 2's persona comparison, built by **reusing** `variants.call_bedrock()` rather than duplicating the API call |
| `tests/conftest.py` | One shared mocked-client fixture used by both test files |

**Why does `roles.py` import from `variants.py` instead of each file making
its own Bedrock call?** Task 1 varies the *user* prompt structure while
keeping the system message empty; task 2 varies the *system* message while
keeping the user prompt fixed. Routing both through the same low-level
`call_bedrock()` guarantees the request format is identical everywhere, and
means there is exactly one place to mock in tests, one place to fix if the
Converse request shape ever needs to change.

**Why is there no separate `bedrock_client.py` file here** (unlike a
Bedrock-basics practical that also needs to *list* models)? This practical
only ever sends prompts — it never calls the control-plane "list models"
API — so there's only one client to build. That single function lives in
`utils/config.py` next to the settings it depends on, rather than being
split into its own near-empty module.

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
# Default region name:    us-east-1        # wherever Bedrock + your model access is enabled
# Default output format:  json
```

### 3. (Optional) override defaults via `.env`
```
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
BEDROCK_MAX_TOKENS=512
BEDROCK_TEMPERATURE=0.3
BEDROCK_TOP_P=0.9
```

---

## Usage

### Run the notebook (recommended)
```bash
cd notebook
jupyter notebook prompt_engineering.ipynb
```

### Or run the scripts directly
```bash
# Task 1: zero-shot vs few-shot vs CoT, on a sample ticket
python -m src.variants

# Task 2: senior data analyst vs creative writer, on a sample prompt
python -m src.roles
```

### Run the tests
```bash
pytest tests/ -v
```
All 14 tests run offline against a mocked Bedrock client — no AWS
credentials or cost required.

---

## Task 1 summary: prompt technique comparison

| Technique | What it adds | Cost/latency | Best for |
|---|---|---|---|
| Zero-shot | Task instructions + category list only | Lowest | Simple, unambiguous tasks |
| Few-shot | 3 worked examples before the real input | Medium | Easily-confused categories, strict output formatting |
| Chain-of-thought | Explicit "think step by step" reasoning ask | Highest (more output tokens) | Ambiguous inputs, when you need an auditable rationale |

The classification task (HR ticket → Payroll / Leave & Attendance /
IT Support / Policy & General) is deliberately tested with one ambiguous
ticket that mentions two topics at once, since that's where the techniques
should visibly diverge — see the comparison table.

## Task 2 summary: persona / system-message comparison

Same user prompt sent twice, only the `system` field changes:

| Persona | Tone | Structure | Best for |
|---|---|---|---|
| Senior data analyst | Dry, precise, quantitative | Structured, leads with the "what" | Reports, dashboards, decision-support text |
| Creative writer | Expressive, narrative, metaphor-heavy | Flowing prose, leads with a hook | Blog posts, culture decks, employer-branding copy |

See for the full comparison and space to record your
own observed output.

---

## Troubleshooting

- **`AccessDeniedException`** — enable the specific model on the Bedrock
  console's **Model access** page (one-time, per-account, per-model step).
- **`NoCredentialsError`** — run `aws configure`, or set
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars.
- **`ThrottlingException`** — the client retries with adaptive backoff
  already (`utils/config.py` → `max_retry_attempts`); on a trial account,
  wait and retry or reduce request frequency.
