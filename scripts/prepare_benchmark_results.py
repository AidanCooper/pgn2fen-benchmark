"""Prepares benchmark results and visualisations for PGN2FEN experiments"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from pgn2fen.evaluate import get_metric
from pgn2fen.models import PGN2FENLog
from pgn2fen.pgn_io import load_logs_from_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_TYPE_TO_FILES_STANDARD = {
    "reasoning": [
        "openai_o3-mini-2025-01-31",
        "openai_o4-mini-2025-04-16",
        "openai_o3-2025-04-16",
        "google_gemini-2.5-pro-preview-03-25",
        "google_gemini-2.5-flash-preview-04-17",
        "deepseek_deepseek-reasoner",
    ],
    "non_reasoning": [
        "openai_gpt-4.1-nano-2025-04-14",
        "openai_gpt-4.1-mini-2025-04-14",
        "openai_gpt-4.1-2025-04-14",
        "openai_gpt-3.5-turbo-instruct",
        "google_gemini-2.0-flash-lite-001",
        "google_gemini-2.0-flash-001",
        "deepseek_deepseek-chat",
        "chessgpt_chessgpt-chat-v1.Q4_K",
    ],
}

MODEL_TYPE_TO_FILES_RANDOMISED = {
    "reasoning": [
        "openai_o3-mini-2025-01-31_randomised",
        "openai_o4-mini-2025-04-16_randomised",
        "openai_o3-2025-04-16_randomised",
        # "google_gemini-2.5-flash-preview-04-17_randomised",
    ],
    "non_reasoning": [
        "google_gemini-2.0-flash-001_randomised",
        "google_gemini-2.0-flash-lite-001_randomised",
        "openai_gpt-4.1-nano-2025-04-14_randomised",
        "openai_gpt-4.1-mini-2025-04-14_randomised",
        "openai_gpt-4o-mini-2024-07-18_randomised",
        "openai_gpt-3.5-turbo-instruct_randomised",
    ],
}

MODEL_TYPE_TO_FILES_FISCHER = {
    "reasoning": [
        "openai_o3-mini-2025-01-31_fischer_random",
        "openai_o4-mini-2025-04-16_fischer_random",
        "openai_o3-2025-04-16_fischer_random",
    ],
    "non_reasoning": [
        "google_gemini-2.0-flash-001_fischer_random",
        "google_gemini-2.0-flash-lite-001_fischer_random",
        "openai_gpt-4.1-nano-2025-04-14_fischer_random",
        "openai_gpt-4.1-mini-2025-04-14_fischer_random",
        "openai_gpt-4o-mini-2024-07-18_fischer_random",
        "openai_gpt-3.5-turbo-instruct_fischer_random",
    ],
}


def prepare_table(
    json_files: list[Path],
    model_type: str,
    evaluation_metric: str = "full_correctness",
    strata: list[tuple[int, int]] | None = None,
    subdir: str = "",
) -> pd.DataFrame:
    """
    Prepares a table summarising evaluation metrics for PGN2FEN experiments.

    Args:
        json_files (list[Path]): List of paths to JSONL files containing experiment logs.
        model_type (str): Type of model (e.g., "reasoning" or "non_reasoning").
        evaluation_col (str): The evaluation metric to calculate.
        strata (list[tuple[int, int]] | None): List of move ranges for stratified analysis.
        subdir (str): Subdirectory for saving results.

    Returns:
        pd.DataFrame: DataFrame containing evaluation results.
    """
    if strata is None:
        strata = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]

    evaluation_cols = [f"{strata[i][0]}-{strata[i][1]} moves" for i in range(len(strata))]
    df = pd.DataFrame(columns=["provider", "model", *evaluation_cols])
    for json_file in json_files:
        logs: list[PGN2FENLog] = load_logs_from_jsonl(json_file)
        if not logs:
            continue

        metrics_dict = {}
        for stratum in strata:
            logs_ = [
                log for log in logs if stratum[0] <= log.game_info.number_of_halfmoves <= stratum[1]
            ]
            metrics_dict[stratum] = get_metric(evaluation_metric, logs_)

        row = {
            "provider": logs[0].llm_info.provider,
            "model": logs[0].llm_info.model,
            **{f"{stratum[0]}-{stratum[1]} moves": metrics_dict[stratum] for stratum in strata},
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df = df.sort_values(
        by=evaluation_cols,
        ascending=False,
    )

    output_dir = PROJECT_ROOT / "results" / subdir / evaluation_metric
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_dir / f"{model_type}.csv",
        index=False,
    )
    df.to_markdown(
        output_dir / f"{model_type}.md",
        index=False,
    )
    return df


def prepare_bar_plot(
    df: pd.DataFrame,
    model_type: str,
    evaluation_metric: str,
    strata: list[tuple[int, int]],
    subdir: str = "",
) -> None:
    """
    Creates a bar plot visualising evaluation metrics for PGN2FEN experiments.

    Args:
        df (pd.DataFrame): DataFrame containing evaluation results.
        model_type (str): Type of model (e.g., "reasoning" or "non_reasoning") for labeling.
        evaluation_metric (str): The evaluation metric to visualise.
        strata (list[tuple[int, int]]): List of move ranges for stratified analysis.
        subdir (str): Subdirectory for saving results.

    Returns:
        None
    """
    colours = []
    providers: {str, (int, list[str])} = {
        "google": [0, ["#3367D6", "#5C9DFF", "#4285F4", "#174EA6", "#0B3D91", "#7BAAF7"]],
        "openai": [0, ["#8E59FF", "#C084FC", "#5E2CA5", "#B266FF"]],
        "deepseek": [0, ["#00A88E", "#00D1C1", "#00BFAE", "#008578"]],
        "chessgpt": [0, ["#E6AC00", "#FF9900", "#FFB800", "#CC8800"]],
        "baseline": [0, ["#D9D9D9", "#A6A6A6", "#737373", "#404040"]],
    }  # int tracks how many colours have been used for each provider
    for provider, model in zip(df["provider"], df["model"], strict=False):
        try:
            colours.append(providers[provider][1][providers[provider][0]])
            providers[provider][0] += 1
        except IndexError as e:
            raise ValueError(f"Too many models for provider {provider} (model: {model})") from e
        except KeyError as e:
            raise ValueError(f"Unknown provider: {provider} (model: {model})") from e

    df_ = df.copy()
    df_ = df_.fillna(0)
    df_ = df_.set_index(["provider", "model"])

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(10, 7))
    df_.T.plot(kind="bar", ax=ax, color=colours)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_xticklabels([f"{stratum[0]}-{stratum[1]}\nmoves" for stratum in strata], rotation=45)
    ax.grid(axis="y", linestyle="--")
    ax.set_title(
        f"{evaluation_metric.replace("_", " ").title()} ({model_type.replace("_", "-").title()} Models)"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    plt.tight_layout()

    output_dir = PROJECT_ROOT / "results" / subdir / evaluation_metric
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{model_type}.png")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare benchmark results and visualizations.")
    parser.add_argument(
        "--evaluation-metrics",
        nargs="+",
        default=[
            "full_correctness",
            "piece_placement",
            "turn",
            "castling",
            "en_passant",
            "halfmove_clock",
            "fullmove_number",
            "levenshtein_ratio",
        ],
        help="List of evaluation metrics to compute. Options: ['full_correctness' 'piece_placement' 'turn' 'castling' 'en_passant' 'halfmove_clock' 'fullmove_number' 'levenshtein_ratio'].",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    evaluation_metrics = args.evaluation_metrics
    strata = [(0, 10), (11, 20), (21, 40), (41, 60), (61, 80), (81, 100)]

    input_dir = PROJECT_ROOT / "model_logs"

    for benchmark, model_type_to_files in zip(
        [
            "standard",
            "randomised",
            "fischer",
        ],
        [MODEL_TYPE_TO_FILES_STANDARD, MODEL_TYPE_TO_FILES_RANDOMISED, MODEL_TYPE_TO_FILES_FISCHER],
    ):
        for model_type, model_files in model_type_to_files.items():
            for evaluation_metric in evaluation_metrics:
                json_files = [input_dir / f"{file}.jsonl" for file in model_files]
                if evaluation_metric == "levenshtein_ratio":
                    json_files.append(input_dir / f"baseline_starting_board_{benchmark}.jsonl")

                # Prepare data for analysis
                df = prepare_table(
                    json_files,
                    model_type,
                    evaluation_metric,
                    strata,
                    subdir=benchmark,
                )

                # Visualise results as a bar plot
                prepare_bar_plot(
                    df,
                    model_type,
                    evaluation_metric,
                    strata,
                    subdir=benchmark,
                )


if __name__ == "__main__":
    main()
