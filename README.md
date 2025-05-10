# FitGPT Backend

A production-ready FastAPI backend for the FitGPT conversational fitness assistant.

## Features

- RESTful API with versioned endpoints
- Persistent storage with SQLite and Redis support
- Rate limiting for API endpoints
- Robust error handling and logging
- CORS configuration for multiple environments
- Swagger documentation for API exploration

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fitgpt-backend.git
   cd fitgpt-backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on the example:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file with your configuration.

### Running the Application

For development:
```bash
python -m app.main
```

For production:
```bash
gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000
```

## API Documentation

API documentation is available at:
- `/api/v1/docs` - Swagger UI
- `/api/v1/openapi.json` - OpenAPI specification

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| ENVIRONMENT | Application environment | development |
| PORT | API server port | 8000 |
| ALLOWED_ORIGINS | Comma-separated list of allowed origins for CORS | http://localhost:5173 |
| RATE_LIMIT | Rate limit for API requests | 20/minute |
| STORAGE_TYPE | Storage backend type (sqlite or redis) | sqlite |
| SQLITE_PATH | Path for SQLite database | ./data/conversations.db |
| REDIS_URL | Redis connection URL | redis://localhost:6379/0 |
| LOG_DIR | Directory for log files | ./logs |
| LOG_LEVEL | Logging level | INFO |

## Directory Structure

```
fitgpt-backend/
├── app/                  # Application code
│   ├── api/              # API endpoints
│   ├── middleware/       # Custom middleware
│   ├── models/           # Pydantic data models
│   ├── services/         # Business logic and services
│   └── utils/            # Utility functions
├── data/                 # Data directory for storage
├── logs/                 # Log files
└── tests/                # Unit and integration tests
```

## Testing

Run the tests with pytest:
```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
