# PlannerAgent Tests

Integration tests for the `PlanAndExecuteAgent`.

## Running Tests Locally

### 1. Install Dependencies
Create a virtual environment and install dependencies.
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file from the `.env.example` and add your API credentials:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-anthropic-key
TAVILY_API_KEY=your-tavily-key
```

### 3. Run Tests
```bash
# Run all tests
python -m unittest tests

# Run specific test class
python -m unittest tests.test_agent.TestAgentResponse

# Run specific test method
python -m unittest tests.test_agent.TestAgentResponse.test_agent_run
```

## Test Coverage

- **TestKnowledgeBaseLoading** - Knowledge base file loading and error handling
- **TestLLMCreation** - LLM initialization for different providers
- **TestSaveGraphDiagram** - Graph diagram saving and directory creation
- **TestToolValidation** - Tool validation and BaseTool handling
- **TestAgentInitialization** - Full agent initialization with various configurations
- **TestAgentResponse** - End-to-end agent execution (run and stream)

## Notes

Tests will skip gracefully if required credentials are not available.
