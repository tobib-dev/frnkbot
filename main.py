from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from pprint import pformat

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY not set!!!")

model = init_chat_model(
    "openai/gpt-5-nano",
    model_provider="openrouter",
    model_kwargs={"models": ["openai/gpt-5-mini"]}
)

summary_model = init_chat_model(
    "openai/gpt-4o-mini",
    model_provider="openrouter"
)


agent = create_agent(
    model=model,
)

def main():
    resp = agent.invoke({"messages": [HumanMessage("What model are you using?")]})
    msg = resp["messages"][-1]
    print(msg.content)
    print()
    print(f"Model name: {msg.response_metadata["model_name"]}")


if __name__ == "__main__":
    main()
