"""This module provides a factory function to create model instances based on the specified client.
It supports LlamaCppModel, OpenAIModel, and TransformersModel.
It also includes a utility function to parse tool arguments in chat messages.
"""

import torch

from tklearn.agents.core.models import (
    ChatMessage,
    MessageRole,
    Model,
    OpenAIModel,
    TokenUsage,
    TransformersModel,
    parse_json_if_needed,
    remove_stop_sequences,
)
from tklearn.agents.core.tools import Tool
from tklearn.logging import get_logger
from tklearn.nn.utils.devices import get_device

try:
    from llama_cpp import Llama, LlamaGrammar
except ModuleNotFoundError:
    Llama = None
    LlamaGrammar = None

logger = get_logger(__name__)


def parse_tool_args_if_needed(message: ChatMessage) -> ChatMessage:
    """
    Parses the arguments of tool calls in a chat message if they are in JSON format.

    This function iterates over all tool calls in the provided `ChatMessage` object
    and applies `parse_json_if_needed` to the `arguments` of each tool call's function.

    Parameters
    ----------
    message : ChatMessage
        The chat message containing tool calls with potential JSON arguments.

    Returns
    -------
    ChatMessage
        The updated chat message with parsed tool call arguments.
    """
    for tool_call in message.tool_calls:
        tool_call.function.arguments = parse_json_if_needed(
            tool_call.function.arguments
        )
    return message


class LlamaCppModel(Model):
    """
    A model wrapper for interacting with the `llama-cpp-python` library, enabling
    text generation and chat completion functionalities. This class supports
    loading models either from a local path or from a repository.

    Parameters
    ----------
    model_path : str or None, optional
        Path to the local model file. Required if `repo_id` and `filename` are not provided.
    repo_id : str or None, optional
        Repository ID for downloading the model. Required if `model_path` is not provided.
    filename : str or None, optional
        Filename of the model within the repository. Required if `repo_id` is provided.
    device : str or torch.device or None, optional
        Device to run the model on. Defaults to CPU if not specified.
    n_ctx : int, optional
        Context window size for the model. Default is 8192.
    max_tokens : int, optional
        Maximum number of tokens to generate. Default is 1024.
    temperature : float, optional
        Sampling temperature for generation. Default is 0.2.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.
    **kwargs : dict
        Additional keyword arguments passed to the underlying `llama-cpp-python` library.

    Raises
    ------
    ImportError
        If the `llama-cpp-python` library is not installed.
    ValueError
        If neither `model_path` nor `repo_id`+`filename` are provided.

    Methods
    -------
    generate(messages, stop_sequences=None, grammar=None, tools_to_call_from=None, **kwargs)
        Generates a response from the llama.cpp model based on the provided messages
        and optional parameters. Integrates tool usage if tools are provided.

    Notes
    -----
    - The `generate` method supports additional features such as stop sequences, grammar
      constraints, and tool integration for advanced use cases.
    - The model can be loaded either from a local path or a repository, providing flexibility
      in deployment.
    """

    def __init__(
        self,
        model_path: str | None = None,
        repo_id: str | None = None,
        filename: str | None = None,
        device: str | torch.device | None = None,
        n_ctx: int = 8192,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        seed: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if Llama is None:
            raise ImportError("`llama-cpp-python` is not installed")
        self.temperature = temperature
        self.seed = seed
        device: torch.device = get_device(device)
        n_gpu_layers = (
            device.index
            if device.index
            else -1
            if device.type == "cuda"
            else 0
        )
        if model_path:
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                max_tokens=max_tokens,
                verbose=verbose,
                **kwargs,
            )
        elif repo_id and filename:
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                max_tokens=max_tokens,
                verbose=verbose,
                **kwargs,
            )
        else:
            raise ValueError(
                "must provide either model_path or repo_id+filename"
            )

    def __del__(self):
        """
        Destructor to clean up the Llama model instance.
        This ensures that resources are released when the model instance is no longer needed.
        """
        try:
            # Workaround for llama-cpp-python issue: https://github.com/abetlen/llama-cpp-python/issues/2002
            self.llm.close()
        except Exception:
            pass

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        try:
            generation_kwargs = self._prepare_completion_kwargs(
                messages=messages,
                stop_sequences=stop_sequences,
                grammar=grammar,
                tools_to_call_from=tools_to_call_from,
                flatten_messages_as_text=True,
                **kwargs,
            )

            if not tools_to_call_from:
                generation_kwargs.pop("tools", None)
                generation_kwargs.pop("tool_choice", None)

            generation_kwargs.setdefault("temperature", self.temperature)
            generation_kwargs.setdefault("seed", self.seed)

            filtered_kwargs = {
                k: v
                for k, v in generation_kwargs.items()
                if k
                not in [
                    "messages",
                    "stop",
                    "grammar",
                    "max_tokens",
                    "tools_to_call_from",
                    "device_map",
                ]
            }
            max_new_tokens = (
                kwargs.get("max_new_tokens")
                or kwargs.get("max_tokens")
                or self.kwargs.get("max_new_tokens")
                or self.kwargs.get("max_tokens")
            )

            if max_new_tokens:
                generation_kwargs["max_new_tokens"] = max_new_tokens

            generation_kwargs.setdefault("stop", [])

            response = self.llm.create_chat_completion(
                messages=generation_kwargs["messages"],
                stop=generation_kwargs["stop"],
                grammar=LlamaGrammar.from_string(grammar) if grammar else None,
                max_tokens=generation_kwargs.get("max_new_tokens", None),
                **filtered_kwargs,
            )

            content = response["choices"][0]["message"]["content"]

            if stop_sequences:
                content = remove_stop_sequences(content, stop_sequences)

            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                raw={
                    "out": content,
                    "completion_kwargs": {
                        key: value
                        for key, value in generation_kwargs.items()
                        if key != "inputs"
                    },
                },
                token_usage=TokenUsage(
                    input_tokens=response.get(
                        "usage", {"prompt_tokens": 0}
                    ).get("prompt_tokens", 0),
                    output_tokens=response.get(
                        "usage", {"completion_tokens": 0}
                    ).get("completion_tokens", 0),
                ),
            )
        except Exception as e:
            logger.error(f"Model error: {e}")
            return ChatMessage(role="assistant", content=f"Error: {str(e)}")


# model = OpenAIModel(
#     model_id="meta-llama/llama-3.2-1b-instruct",
#     api_base="https://openrouter.ai/api/v1",
#     api_key=os.environ["OPENROUTER_API_KEY"],
#     temperature=temperature or 1.0,
#     seed=seed,
# )

# model = TransformersModel(
#     model_id="watt-ai/watt-tool-8B",
#     temperature=temperature or 1.0,
#     seed=seed,
#     device_map=get_device(device),
# )


def model_factory(client: str, **kwargs):
    """
    Factory function to create and return a model instance based on the specified client.

    Parameters
    ----------
    client : str
        The name of the client for which the model is to be created.
        Supported values are "llama_cpp", "openai", and others defaulting to TransformersModel.
    kwargs : dict, optional
        Additional keyword arguments to be passed to the model constructor.
        Defaults to an empty dictionary if not provided.

    Returns
    -------
    object
        An instance of the model corresponding to the specified client.

    Notes
    -----
    - If `client` is "llama_cpp", an instance of `LlamaCppModel` is returned.
    - If `client` is "openai", an instance of `OpenAIModel` is returned.
    - For any other value of `client`, an instance of `TransformersModel` is returned.
    """
    if kwargs is None:
        kwargs = {}
    if client == "llama_cpp":
        return LlamaCppModel(**kwargs)
    elif client == "openai":
        return OpenAIModel(**kwargs)
    return TransformersModel(**kwargs)
