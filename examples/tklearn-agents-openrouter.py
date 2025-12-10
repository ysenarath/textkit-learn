from __future__ import annotations

import os
from typing import Any

from tklearn.agents.core import Tool, ToolCallingAgent
from tklearn.agents.core.tools import tool
from tklearn.agents.models import model_factory
from tklearn.nn.utils.devices import get_device

device = get_device()
max_steps = 20

model = model_factory(
    client="openai",
    model_id="openai/gpt-4o-mini",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=1.0,
    seed=42,
)


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {
        "answer": {
            "type": "any",
            "description": "The final answer to the problem",
        }
    }
    output_type = "any"

    def __init__(self) -> None:
        super().__init__()

    def forward(self, answer: str) -> Any:
        return answer


final_answer = FinalAnswerTool()


@tool
def wikipedia_tool(query: str) -> str:
    """Search Wikipedia for the given query and return a summary.

    Args:
        query (str): The search query.

    Returns:
        str: A summary of the Wikipedia article.
    """
    return "Apple is not a company in this context; it is a fruit."


agent = ToolCallingAgent(
    tools=[final_answer, wikipedia_tool],
    model=model,
    max_steps=max_steps,
)

agent.run("""Search wikipedia for best matching meaning for the term in brackets in the text.

Text:
(Apple) is the best company that produces smarter phones.""")
