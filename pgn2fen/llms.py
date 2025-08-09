import os

import backoff
import openai
from google import genai
from google.genai import types
from llama_cpp import Llama

from pgn2fen.models import Provider

# ruff: noqa: E501
PROMPT_TEMPLATE = """
## Task
Your task is to convert the provided PGN representation of a {variant} chess game into a FEN string.

## Instructions
1. Read the provided PGN text carefully.
2. Convert the PGN text into a FEN string.
3. Do not include any additional text, explanations, or backticks in your response. ONLY return the FEN string.
4. Do not use code to convert the PGN to FEN. Use your own knowledge and understanding of chess to perform the conversion.
{additional_instructions}
For example, if the PGN text represented the starting position of a standard chess game, you would return the following and nothing else:
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

## Input
{pgn_text}
""".strip()

PROMPT_TEMPLATE_CHESSGPT = """
{pgn_moves} Convert the PGN to FEN
""".strip()


def format_prompt(pgn_text: str) -> str:
    variant = "standard"
    additional_instructions = ""
    if '[Variant "Chess960"]' in pgn_text:
        variant = "Chess960 "
        additional_instructions = "5. Use standard castling notation (KQkq) for the FEN string.\n"

    return PROMPT_TEMPLATE.format(
        pgn_text=pgn_text, variant=variant, additional_instructions=additional_instructions
    )


def format_prompt_chessgpt(pgn_text: str) -> str:
    lines = pgn_text.splitlines()
    return PROMPT_TEMPLATE_CHESSGPT.format(pgn_moves=lines[-1])


def get_gemini_fen(
    pgn_text: str,
    model: str = "gemini-2.0-flash-001",
    thinking_budget: int | None = None,
) -> tuple[str, str | None]:
    """
    FEN retrieval for the Google Gemini API client.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = format_prompt(pgn_text)

    thinking_config = None
    if thinking_budget is not None and "2.5" in model:
        thinking_config = types.ThinkingConfig(
            include_thoughts=True, thinking_budget=thinking_budget
        )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=1.0,
                thinking_config=thinking_config,
            ),
        )
    except Exception as e:
        raise RuntimeError(f"Error during API call: {e}") from e

    response_thoughts = ""
    response_text = ""
    for part in response.candidates[0].content.parts:
        if not part.text:
            continue
        if part.thought:
            response_thoughts += part.text + "\n"
        else:
            response_text = part.text

    return response_text, response_thoughts or None


OPENAI_FLEX_MODELS: list[str] = [
    "o3",
    "o3-2025-04-16",
    "o4-mini",
    "o4-mini-2025-04-16",
]


@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=6)
def get_openai_fen(
    pgn_text: str,
    model: str = "gpt-4.1-mini-2025-04-14",
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
) -> tuple[str, str | None]:
    """
    FEN retrieval for the OpenAI API client. Also supports any OpenAI-compatible API,
    such as DeepSeek.
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=900.0)

    prompt = format_prompt(pgn_text)

    def _call_openai_instruct(
        client: openai.OpenAI, model: str, prompt: str
    ) -> tuple[str, str | None]:
        try:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=1.0,
                max_tokens=200,
            )
        except Exception as e:
            raise RuntimeError(f"Error during API call: {e}") from e
        return str(response.choices[0].text).strip(), None

    def _call_openai_chat(client: openai.OpenAI, model: str, prompt: str) -> tuple[str, str | None]:
        try:
            service_tier = None
            if model in OPENAI_FLEX_MODELS:
                service_tier = "flex"
            response = client.responses.create(
                model=model,
                input=prompt,
                temperature=1.0,
                service_tier=service_tier,
                reasoning={"summary": "auto"},
            )
        except Exception as e:
            raise RuntimeError(f"Error during API call: {e}") from e

        reasoning = ""
        for output in response.output:
            if output.type == "reasoning":
                for summary in output.summary:
                    reasoning += summary.text + "\n"

        return response.output_text, reasoning or None

    if model == "gpt-3.5-turbo-instruct":
        return _call_openai_instruct(client, model, prompt)
    return _call_openai_chat(client, model, prompt)


def get_chessgpt_fen(
    pgn_text: str,
    model: str = "chessgpt-chat-v1.Q4_K.gguf",
    model_directory: str | None = None,
) -> tuple[str, str | None]:
    """
    FEN retrieval using ChessGPT chat or base via llama.cpp.
    """
    if model_directory is None:
        model_directory = os.getenv("CHESSGPT_MODEL_DIR", "~/Models/")

    llm = Llama(
        model_path=model_directory + model,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )

    user_prompt = format_prompt_chessgpt(pgn_text)
    if "chat" in model:
        prompt = f"A friendly, helpful chat between some humans.<|endoftext|>Human 0: {user_prompt}<|endoftext|>Human 1:"
    elif "base" in model:
        prompt = f"Q: {user_prompt} A:"
    else:
        raise ValueError(f"Unsupported ChessGPT model: {model}")

    try:
        response = llm(
            prompt,
            max_tokens=100,
            stop=["<|endoftext|>", "Human 0:", "Human 1:"],
            echo=False,
        )
    except Exception as e:
        raise RuntimeError(f"Error during llama.cpp call: {e}") from e

    response_text = response["choices"][0]["text"].strip()
    return response_text, None


def get_fen(
    pgn_text: str,
    provider: Provider = Provider.GOOGLE,
    model: str = "gemini-2.0-flash-001",
    thinking_budget: int | None = None,
) -> tuple[str, str | None]:
    """
    Get the FEN string from the PGN text using the specified provider and model.

    Args:
        pgn_text (str): The PGN text to convert.
        provider (Provider): The LLM provider to use.
        model (str): The model to use for the provider.
        thinking_budget (int | None): The maximum number of tokens for the LLM to think.
            Currently only implemented for the GOOGLE provider.

    Returns:
        str: The FEN string and the LLM reasoning (if applicable).

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider == Provider.GOOGLE:
        fen_string, reasoning = get_gemini_fen(
            pgn_text, model=model, thinking_budget=thinking_budget
        )
    elif provider == Provider.OPENAI:
        fen_string, reasoning = get_openai_fen(pgn_text, model=model)
    elif provider == Provider.DEEPSEEK:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        fen_string, reasoning = get_openai_fen(
            pgn_text, model=model, api_key=api_key, base_url="https://api.deepseek.com/v1"
        )
    elif provider == Provider.CHESSGPT:
        fen_string, reasoning = get_chessgpt_fen(pgn_text, model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return fen_string, reasoning
