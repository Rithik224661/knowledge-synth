# AI Research Assistant - Backend

This is the backend service for the AI Research Assistant, built with FastAPI and LangChain.

## Features

- **Enhanced Research Agent**: Powered by OpenAI's GPT-4 with improved query decomposition and orchestration
- **Multi-Tool Integration**: Web search, academic paper retrieval, document processing, and trend analysis
- **Advanced Visualizations**: Interactive publication trend analysis with topic identification
- **RESTful API**: Built with FastAPI for high performance and easy integration
- **Asynchronous Processing**: For handling multiple research requests efficiently
- **Evaluation Framework**: Built-in tools for evaluating agent performance with faithfulness and relevance metrics

## Prerequisites

- Python 3.9+
- OpenAI API key
- Google API key and Custom Search Engine ID (for web search)

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd knowledge-synth/backend
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   - Copy `.env.example` to `.env`
   - Update the values in `.env` with your API keys

## Running the Application

### Development

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### Production

For production, use a production-ready server like Uvicorn with Gunicorn:

```bash
pip install gunicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, you can access:

- **Interactive API docs**: `http://localhost:8000/docs`
- **Alternative API docs**: `http://localhost:8000/redoc`
- **Health check**: `http://localhost:8000/health`

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   └── research.py
│   │       └── __init__.py
│   ├── core/
│   │   └── config.py
│   ├── services/
│   │   └── research_agent.py
│   ├── __init__.py
│   └── main.py
├── tests/
├── .env
├── .gitignore
└── requirements.txt
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `GOOGLE_API_KEY` | Google API key for web search | No |
| `GOOGLE_CSE_ID` | Google Custom Search Engine ID | No |
| `SECRET_KEY` | Secret key for JWT token signing | Yes |
| `ENVIRONMENT` | Environment (development/production) | No |

## Architecture

### Research Agent Architecture

The enhanced research agent is designed with a modular architecture that enables complex query processing and multi-tool orchestration:

```
ResearchAgent
├── Query Decomposition
│   ├── Pattern Recognition
│   ├── LLM-based Decomposition
│   └── Dependency Management
├── Tool Integration
│   ├── Web Search (Tavily/Google/DuckDuckGo)
│   ├── Academic Search (ArXiv)
│   ├── Document Processing
│   └── Trend Analysis
├── Visualization
│   ├── Publication Trends
│   ├── Growth Rate Analysis
│   └── Topic Identification
└── Evaluation Framework
    ├── Faithfulness Scoring
    ├── Relevance Scoring
    └── Overall Quality Assessment
```

### Key Components

1. **Query Decomposition**
   - Breaks down complex research queries into manageable sub-tasks
   - Identifies dependencies between tasks for proper orchestration
   - Uses pattern recognition for common research query types
   - Falls back to LLM-based decomposition for novel queries

2. **Multi-Tool Integration**
   - Selects appropriate tools based on sub-task requirements
   - Manages token usage to prevent LLM context limitations
   - Handles failures gracefully with fallback mechanisms

3. **Enhanced Visualizations**
   - Creates interactive publication trend visualizations
   - Calculates and displays year-over-year growth rates
   - Uses LLM to identify key research topics from paper titles
   - Presents notable papers organized by year

4. **Evaluation Framework**
   - Assesses faithfulness of responses to source materials
   - Evaluates relevance of responses to original queries
   - Calculates overall quality scores

## Evaluation Report

### Performance Improvements

| Feature | Before | After | Improvement |
|---------|--------|-------|-------------|
| Query Complexity | Simple queries only | Complex multi-step queries | Significant |
| Tool Orchestration | Manual selection | Automatic based on sub-tasks | High |
| Visualization | Basic bar charts | Interactive with trends and topics | Substantial |
| Token Efficiency | Frequent context limits | Managed token usage | Moderate |

### Query Decomposition Evaluation

The enhanced query decomposition system was tested with various complex queries and demonstrated the following capabilities:

- Successfully breaks down complex queries into 3-7 sub-tasks on average
- Correctly identifies dependencies between tasks (e.g., search before summarization)
- Properly selects appropriate tools for each sub-task
- Handles edge cases with pattern recognition when LLM decomposition fails

### Visualization Enhancements

The improved visualization system provides:

- Publication trend analysis with trend lines and data labels
- Year-over-year growth rate visualization
- Automatic identification of key research topics
- Tabular presentation of notable papers by year
- Summary statistics for quick insights

## Demo

A demonstration notebook is available at `backend/demo_notebook.ipynb` that showcases the enhanced research agent capabilities with example queries.

## Testing

To run tests:

```bash
pytest
```

## Deployment

### Docker

1. Build the Docker image:
   ```bash
   docker build -t research-assistant-backend .
   ```

2. Run the container:
   ```bash
   docker run -d -p 8000:8000 --env-file .env research-assistant-backend
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License.
