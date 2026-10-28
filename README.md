# Salary Prediction MCP Server

This project exposes a FastMCP salary prediction API and a Streamlit chat UI that calls it through an MCP client.

## Features
- Predict market salary based on experience, education, skills, and location
- Explain the prediction using SHAP feature contributions
- Analyze salary gaps versus current salary

## Quick start
1. Create a virtual environment and install the project dependencies.
2. Add your keys in a `.env` file.
3. Run the MCP server or the Streamlit app.

## Environment variables
Create a `.env` file in the project root or in `salary_mcp_chat_agent/` with:

```env
MCP_SERVER_URL=https://salary-predict-mcp.fastmcp.app/mcp
MCP_API_KEY=your_fastmcp_key
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
```

## Run the salary MCP server
```bash
python salary_mcp_server.py
```

## Run the chat UI
```bash
cd salary_mcp_chat_agent
pip install -r requirements.txt
streamlit run app.py
```

The app expects the MCP and Groq keys to be available in the environment or in a `.env` file.