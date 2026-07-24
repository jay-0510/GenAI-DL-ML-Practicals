# Milestone 1: LLM Microservice

A FastAPI microservice backed by AWS Bedrock for text classification and summarization.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Folder Structure](#folder-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the Service](#running-the-service)
8. [API Endpoints](#api-endpoints)
9. [Testing](#testing)
10. [Error Handling](#error-handling)
11. [Deployment](#deployment)

---

## Project Overview

This microservice provides two core NLP capabilities powered by AWS Bedrock:

| Endpoint | Description | Input | Output |
|----------|-------------|-------|--------|
| `POST /classify` | Classifies text into predefined categories | Text string | `{label, confidence}` |
| `POST /summarize` | Generates concise summary of long text | Long text | Summary string |

**Key Features:**
- FastAPI framework with async support
- AWS Bedrock integration via boto3
- Pydantic models for request/response validation
- Swagger/OpenAPI documentation at `/docs`
- Comprehensive pytest test suite
- Structured error handling

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Client Layer                          │
│              (HTTP Requests / cURL / Postman)              │
└─────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                    API Gateway Layer                       │
│                                                            │
│   ┌────────────────────┐   ┌────────────────────┐        │
│   │  POST /classify    │   │  POST /summarize   │        │
│   │  (classify.py)    │   │  (summarize.py)    │        │
│   └─────────┬──────────┘   └─────────┬──────────┘        │
│             └──────────┬─────────────┘                     │
│                        ▼                                   │
│   ┌────────────────────────────────────────────┐          │
│   │       Pydantic Validation Layer            │          │
│   │   - Request Models (requests.py)           │          │
│   │   - Response Models (schemas.py)           │          │
│   └────────────────────┬───────────────────────┘          │
└─────────────────────────┼────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                   Business Logic Layer                     │
│                                                            │
│   ┌────────────────────────────────────────────┐          │
│   │       bedrock_service.py                    │          │
│   │   - Prompt engineering                     │          │
│   │   - Model invocation                       │          │
│   │   - Response parsing                       │          │
│   └────────────────────┬───────────────────────┘          │
└─────────────────────────┼────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                External Services Layer                     │
│                                                            │
│   ┌────────────────────────────────────────────┐          │
│   │           AWS Bedrock                       │          │
│   │   - Foundation Models (Claude, Llama)      │          │
│   │   - boto3 SDK integration                  │          │
│   └────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. Client sends HTTP POST request with JSON payload
2. FastAPI route receives request and validates via Pydantic
3. Validated data passed to Bedrock service layer
4. Service constructs prompt and calls AWS Bedrock via boto3
5. Bedrock returns LLM response
6. Service parses response and returns structured JSON
7. FastAPI sends formatted response to client

---

## Folder Structure

```
milestone-1/
│
├── app/                          # Main application package
│   ├── __init__.py               # Package initializer
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # Environment & AWS configuration
│   │
│   ├── models/                   # Data models
│   │   ├── __init__.py
│   │   ├── schemas.py            # Pydantic response models
│   │   └── requests.py           # Pydantic request models
│   │
│   ├── routes/                   # API route handlers
│   │   ├── __init__.py
│   │   ├── classify.py           # /classify endpoint
│   │   └── summarize.py           # /summarize endpoint
│   │
│   ├── services/                 # Business logic
│   │   ├── __init__.py
│   │   └── bedrock_service.py    # Bedrock integration
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       └── exceptions.py         # Custom exceptions & handlers
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures
│   ├── test_classify.py          # Classify endpoint tests
│   ├── test_summarize.py         # Summarize endpoint tests
│   └── test_error_handling.py    # Error handling tests
│
├── .env.example                   # Environment template
├── requirements.txt               # Python dependencies
├── pytest.ini                     # Pytest configuration
└── README.md                      # Project documentation
```

---

## Prerequisites

Before installation, ensure you have:

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.9+ | Runtime environment |
| AWS Account | - | Bedrock access |
| AWS CLI | 2.x | Credential configuration |
| boto3 | Latest | AWS SDK for Python |
| FastAPI | 0.100+ | Web framework |
| Uvicorn | 0.23+ | ASGI server |

**AWS Bedrock Access:**
1. Request Bedrock access in AWS Console
2. Ensure IAM permissions for `bedrock:ConverseModel`
3. Configure AWS credentials via AWS CLI or environment variables

---

## Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd milestone-1
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected Dependencies:**
- `fastapi` - Web framework
- `uvicorn[standard]` - ASGI server
- `boto3` - AWS SDK
- `pydantic` - Data validation
- `pytest` - Testing framework
- `pytest-asyncio` - Async test support
- `httpx` - Test client
- `python-dotenv` - Environment management

### Step 4: Configure AWS Credentials

```bash
aws configure
```

Or set environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
```

---

## Configuration

### Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

**`.env` Configuration:**

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region for Bedrock | `us-east-1` |
| `BEDROCK_MODEL_ID` | Bedrock model identifier | `amazon.nova-micro-v1:0` |
| `AWS_ACCESS_KEY_ID` | AWS access key | (from AWS CLI) |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | (from AWS CLI) |
| `APP_ENV` | Application environment | `development` |
| `LOG_LEVEL` | Logging level | `INFO` |

---

## Running the Service

### Development Mode

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Verify Service

- **Root:** http://localhost:8000
- **Swagger Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

---

## API Endpoints

### POST /classify

Classifies text into predefined categories.

**Request:**
```json
{
  "text": "The stock market crashed today due to inflation concerns."
}
```

**Response:**
```json
{
  "label": "finance",
  "confidence": 0.92
}
```

**Features:**
- Auto-detects category from content
- Returns confidence score (0.0 to 1.0)
- Supports multiple classification domains

---

### POST /summarize

Generates a concise summary of long text.

**Request:**
```json
{
  "text": "Long article content here...",
  "max_length": 100
}
```

**Response:**
```json
{
  "summary": "Concise summary of the input text.",
  "original_length": 1500,
  "summary_length": 45
}
```

**Features:**
- Preserves key information
- Configurable maximum length
- Length comparison metrics

---

## Testing

### Run All Tests

```bash
pytest
```

### Run with Verbose Output

```bash
pytest -v
```

### Run Specific Test File

```bash
pytest tests/test_classify.py
```

### Run with Coverage

```bash
pytest --cov=app --cov-report=html
```

### Test Categories

| Test File | Coverage |
|-----------|----------|
| `test_classify.py` | Classification endpoint, response structure |
| `test_summarize.py` | Summarization endpoint, output validation |
| `test_error_handling.py` | Invalid inputs, empty payloads, API errors |

**Minimum Test Cases (3 required):**
1. Test successful classification returns valid JSON
2. Test successful summarization with expected format
3. Test error handling for invalid input

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Text field is required",
    "details": null
  }
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request / Validation Error |
| 401 | Unauthorized (AWS credentials) |
| 500 | Internal Server Error |
| 503 | Bedrock Service Unavailable |

### Custom Exceptions

| Exception | HTTP Code | Trigger |
|-----------|-----------|---------|
| `ValidationError` | 400 | Invalid request payload |
| `BedrockError` | 503 | AWS Bedrock failure |
| `AuthenticationError` | 401 | AWS credential issue |

---

## Deployment

### Local Deployment

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment (Optional)

```bash
docker build -t llm-microservice .
docker run -p 8000:8000 llm-microservice
```

### AWS Lambda / ECS (Future)

Configurable for containerized deployment to:
- AWS ECS Fargate
- AWS Lambda (via Mangum adapter)
- AWS Elastic Beanstalk

---

## Swagger Documentation

Interactive API documentation available at:

- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **OpenAPI Spec:** `/openapi.json`

Features:
- Interactive request testing
- Schema visualization
- Response examples
- Authentication testing

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `BedrockInvocationError` | Check AWS credentials and region |
| `ModuleNotFoundError` | Install dependencies: `pip install -r requirements.txt` |
| `ConnectionError` | Verify AWS Bedrock access in console |
| `ValidationError` | Check request JSON format |

---
