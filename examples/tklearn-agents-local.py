from __future__ import annotations

from typing import Any

from tklearn.agents.core import Tool, ToolCallingAgent
from tklearn.agents.models import model_factory
from tklearn.nn.utils.devices import get_device

device = get_device()
max_steps = 20

# CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python
# module load cuda
# CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
model = model_factory(
    "llama_cpp",
    filename="gemma-3-12b-it-q4_0.gguf",
    repo_id="google/gemma-3-12b-it-qat-q4_0-gguf",
    temperature=1.0,
    seed=42,
    device=device,
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

agent = ToolCallingAgent(
    tools=[final_answer],
    model=model,
    max_steps=max_steps,
)

agent.run("""Search wikipedia for best matching meaning for the term in brackets in the text.

Text:
(Apple) is the best company that produces smarter phones.""")
