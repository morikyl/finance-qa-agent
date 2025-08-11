# FinanceQA OpenAI Agent System

A sophisticated multi-agent financial analysis platform that leverages OpenAI's Agent SDK to provide intelligent, tool-enabled financial question answering with hierarchical routing and comprehensive validation.

## 🚀 Features

- **Multi-Agent Architecture**: Specialized agents for different types of financial questions
- **Intelligent Question Routing**: Automatic classification and handoff to appropriate specialists
- **Tool-Enabled Analysis**: Integration with file search and web search capabilities
- **Tree-like Reasoning**: Structured approach to complex financial calculations
- **Validation & Critique**: Built-in review mechanisms for quality assurance
- **Provenance Tracking**: Complete audit trail for all calculations and assumptions
- **Async Execution**: High-performance asynchronous processing

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Triage Agent   │───▶│ Specialized     │
│   (Questions)   │    │                 │    │ Agents         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  File Search    │    └  Critic Agent   │
                       │     Tool        │         (Review)     │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Web Search     │
                       │     Tool        │
                       └─────────────────┘
```

This README provides a comprehensive overview of your FinanceQA OpenAI Agent System, including:

- **Clear feature descriptions** with emojis for visual appeal
- **Detailed system architecture** with ASCII diagrams
- **Complete agent documentation** explaining each agent's purpose and capabilities
- **Step-by-step installation** and setup instructions
- **Usage examples** with code snippets
- **Configuration options** for tools and agents
- **Workflow explanation** showing the complete process
- **Output format examples** in JSON
- **Testing and validation** procedures
- **Troubleshooting guide** for common issues
- **Future enhancement roadmap**
- **Contributing guidelines**
- **Support information**

The README is structured to be both informative for developers and accessible for users who want to understand and use your system.

## 🤔 Agent Types

### 1. **Triage Agent** 🎯
- **Purpose**: Classifies incoming financial questions and routes them to appropriate specialists
- **Categories**: 
  - **Tactical – Basic**: Directly answerable from provided data
  - **Tactical – Assumption-Based**: Requires logical assumptions or estimates
  - **Conceptual**: Definitional or methodological questions
- **Tools**: File Search Tool, Web Search Tool
- **Output**: JSON classification with routing instructions
- **Delegation**: Routes questions to appropriate specialized agents

### 2. **Assumption Agent** 🧮
- **Purpose**: Handles complex financial questions requiring assumptions, estimates, and multi-step calculations
- **Workflow**:
  1. **Phase 1**: Generate complete plan with formula validation
  2. **Phase 2**: Execute step-by-step resolution with mandatory tool usage
  3. **Phase 3**: Final calculation and validation
- **Tools**: File Search Tool, Web Search Tool
- **Output**: Structured JSON with complete calculation tree and provenance
- **Access**: Full access to both search tools for comprehensive analysis

### 3. **Basic Agent** 📊
- **Purpose**: Handles straightforward financial questions answerable directly from context
- **Capabilities**: Direct data extraction, simple calculations, ratio computations
- **Tools**: File Search Tool, Web Search Tool
- **Output**: JSON with direct answers and calculations
- **Access**: Uses both tools to validate data and verify calculations

### 4. **Conceptual Agent** 📚
- **Purpose**: Answers questions about financial concepts, definitions, and methodologies
- **Capabilities**: Concept explanations, methodology descriptions, relationship analysis
- **Tools**: File Search Tool, Web Search Tool
- **Output**: JSON with conceptual explanations and examples
- **Access**: Uses tools to research authoritative definitions and validate concepts

### 5. **Critic Agent** 🔍
- **Purpose**: Provides critical review and validation of other agents' analyses
- **Responsibilities**: Data accuracy verification, calculation validation, logical soundness assessment
- **Tools**: File Search Tool, Web Search Tool
- **Output**: Detailed critique and validation report
- **Access**: Uses both tools to independently verify data and validate assumptions

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Internet connection for web search functionality

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd financeQA-openai
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set environment variables**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

5. **Configure vector store**
   - Update `vector_store_ids` in `src/orchestration.py` with your OpenAI vector store ID
   - Upload your financial context file to the vector store

## 📖 Usage

### Basic Usage

```python
import asyncio
from src.orchestration import main

# Run the system with default questions
asyncio.run(main())
```

### Custom Questions

```python
# Modify the questions list in src/orchestration.py
questions = [
    "What is the market debt to equity ratio of Costco?",
    "Calculate the EV/Sales ratio for 2024",
    "What is adjusted EBITDA for the year ending in 2024?"
]
```

### Agent Configuration

```python
# Configure agent models and tools
assumption_agent = Agent(
    name="assumption_agent",
    model="gpt-4o",  # or "gpt-4o-mini" for cost optimization
    tools=[file_search_tool, web_search_tool],
    instructions=assumption_agent_prompt,
    handoffs=[critic_agent]
)
```

## ⚙️ Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your-api-key-here
TAVILY_API_KEY=your-tavily-key-here  # Optional: for enhanced web search
```

### Tool Configuration

#### File Search Tool
```python
file_search_tool = FileSearchTool(
    vector_store_ids=["your-vector-store-id"],
    max_num_results=5,
    include_search_results=True,
    ranking_options=None,
    filters=None
)
```

#### Web Search Tool
```python
web_search_tool = WebSearchTool(
    user_location=None,
    search_context_size="medium"  # "low", "medium", or "high"
)
```

### Agent Prompts
- Customize agent behavior by modifying the prompt templates in `src/orchestration.py`
- Each agent has specific instructions for optimal performance
- Prompts include tool usage requirements and output format specifications

## 📊 Data Sources

### Financial Context (`src/data/context.txt`)
- Costco Wholesale Corporation financial statements (2022-2024)
- Consolidated Income Statements
- Consolidated Balance Sheets
- Consolidated Statements of Equity
- Financial Notes and Disclosures

### Vector Store Integration
- OpenAI vector store for semantic search
- Configurable search parameters
- Result ranking and filtering options

## 🔄 Workflow

### 1. Question Input
- User submits financial question
- System initializes with triage agent

### 2. Classification & Routing
- Triage agent analyzes question using both search tools
- Classifies into appropriate category
- **Delegates to specialized agent** based on classification

### 3. Analysis & Calculation
- **Specialized agent** (Basic/Assumption/Conceptual) processes question
- **Each specialized agent has full access to both tools**:
  - File Search Tool for company-specific data
  - Web Search Tool for formula validation and industry standards
- Performs calculations with comprehensive validation

### 4. Review & Validation
- Critic agent reviews analysis using both tools
- Validates calculations and logic independently
- Provides quality assurance with tool-enabled verification

### 5. Output Generation
- Structured JSON response from specialized agent
- Complete calculation provenance
- Confidence levels and assumptions
- Optional critic review if handoff is configured

## 📝 Output Formats

### Triage Agent Output
```json
{
    "classification": "tactical_assumption_based",
    "reason": "Question requires assumptions and estimates",
    "search_results": "Context validation results"
}
```

### Assumption Agent Output
```json
{
    "question": "What is the market debt to equity ratio?",
    "formula_validation": {
        "formula": "Market D/E = Total Debt / Market Value of Equity",
        "internet_validation_result": "Formula confirmed via web search"
    },
    "plan": ["Step 1: Find total debt", "Step 2: Find market equity"],
    "execution": {
        "step_1": {
            "description": "Find total debt components",
            "file_search_results": {...},
            "calculation": "103 + 5,794 + 4,052 = 8,949 million",
            "result": 8949
        }
    },
    "final_result": 0.0214,
    "confidence_level": "High - All components sourced",
    "assumptions_made": [...]
}
```

## 🧪 Testing

### Run Tests
```bash
# Run the main system
python src/orchestration.py

# Test specific agents
python -c "
import asyncio
from src.orchestration import assumption_agent, Runner
result = asyncio.run(Runner.run(assumption_agent, 'What is the market debt to equity ratio?'))
print(result.final_output)
"
```

### Validation
- Each agent includes built-in validation mechanisms
- Critic agent provides independent review
- Tool usage is mandatory and tracked
- All calculations include provenance

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **API Key Issues**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   echo $OPENAI_API_KEY  # Verify it's set
   ```

3. **Vector Store Errors**
   - Verify vector store ID is correct
   - Ensure context file is uploaded
   - Check OpenAI account permissions

4. **Tool Usage Errors**
   - Verify tool configurations
   - Check internet connectivity for web search
   - Ensure proper tool permissions

### Debug Mode
```python
# Enable tracing for debugging
with trace("financial_analysis_workflow") as tracer:
    result = await Runner.run(agent, question)
    print(f"Trace events: {tracer.events}")
```

## 🔮 Future Enhancements

- **Multi-Company Support**: Extend to multiple company contexts
- **Real-time Data**: Integration with live financial data feeds
- **Advanced Analytics**: Machine learning for pattern recognition
- **API Endpoints**: RESTful API for external integrations
- **Dashboard Interface**: Web-based user interface
- **Custom Tools**: User-defined financial analysis tools

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤔 Acknowledgments

- OpenAI for the Agent SDK and API
- The financial analysis community for best practices
- Contributors and users of this system

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

---

**Note**: This system is designed for educational and research purposes. Always verify financial calculations independently before making investment decisions.