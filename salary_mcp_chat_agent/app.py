import asyncio
import os
import threading

import streamlit as st
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

MCP_URL = os.getenv(
    "MCP_SERVER_URL",
    "https://salary-predict-mcp.fastmcp.app/mcp",
)

SYSTEM_PROMPT = """
You are a Salary Intelligence Assistant.
Use MCP tools to predict salaries, explain predictions,
and analyze salary gaps.
Never invent results or guess education and location codes.
"""


# Keep one event loop running across Streamlit reruns.
def run_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


@st.cache_resource
def get_event_loop():
    loop = asyncio.new_event_loop()

    thread = threading.Thread(
        target=run_event_loop,
        args=(loop,),
        daemon=True,
    )
    thread.start()

    return loop


# MCP + Groq agent
async def chat(messages):
    mcp_key = (os.getenv("MCP_API_KEY") or "").strip()
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()

    if not mcp_key or not groq_key:
        raise ValueError(
            "Check MCP_API_KEY and GROQ_API_KEY in your .env file."
        )

    if mcp_key.lower().startswith("bearer "):
        mcp_key = mcp_key.split(None, 1)[1]

    client = MultiServerMCPClient({
        "salary": {
            "url": MCP_URL,
            "transport": "streamable_http",
            "headers": {
                "Authorization": "Bearer " + mcp_key
            },
        }
    })

    tools = await client.get_tools(server_name="salary")

    model = ChatOpenAI(
        model=os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        ),
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    result = await agent.ainvoke({"messages": messages})

    for message in reversed(result["messages"]):
        if message.type != "ai" or message.tool_calls:
            continue

        content = message.content

        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            text = "\n".join(
                item["text"]
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and item.get("text")
            )
            if text:
                return text

    return "No response received from the agent."


# Streamlit interface
st.set_page_config(
    page_title="Salary Intelligence Agent",
    page_icon="💼",
)

st.title("💼 Salary Intelligence Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about your salary...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                loop = get_event_loop()

                future = asyncio.run_coroutine_threadsafe(
                    chat(st.session_state.messages.copy()),
                    loop,
                )

                answer = future.result(timeout=120)

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception as error:
            st.error(
                f"{type(error).__name__}: {error}"
            )
