"""Generate PGN2FEN benchmark results for a specified model"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import argparse

from dotenv import load_dotenv
from tqdm import tqdm

from pgn2fen.llms import get_fen
from pgn2fen.models import LLMInfo, PGN2FENLog, PGNGameInfo, Provider
from pgn2fen.pgn_io import load_logs_from_jsonl, parse_board_from_pgn_file
from pgn2fen.utils import is_fen, process_llm_raw_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def process_single_pgn(
    pgn_file: Path,
    provider: Provider,
    model: str,
    thinking_budget: int | None = None,
    extract_fen: bool = False,
) -> PGN2FENLog:
    """
    Processes a single PGN file.

    Args:
        pgn_file (Path): Path to the PGN file.
        provider (Provider): The LLM provider (e.g., GOOGLE, OPENAI).
        model (str): The model name to use for processing.
        thinking_budget (int | None): Optional thinking budget for the model. Currently
            only implemented for the GOOGLE provider.
        extract_fen (bool): Whether to extract FEN from raw LLM output, or use the text
            response directly as is.

    Returns:
        PGN2FENExperiment: The experiment results including evaluation and metadata.
    """
    with open(pgn_file) as file:
        pgn_text = file.read()
    board = parse_board_from_pgn_file(pgn_file)
    board_fen = board.fen()

    try:
        llm_raw_text = get_fen(pgn_text, provider, model, thinking_budget)
    except Exception as e:
        raise RuntimeError(f"Error calling {provider.value}_{model} for {pgn_file}: {e}") from e

    game_info = PGNGameInfo(
        datetime=str(datetime.now()),
        input_pgn_file=str(pgn_file.relative_to(PROJECT_ROOT)),
        input_fen=board_fen,
        number_of_halfmoves=len(board.move_stack),
    )
    llm_info = LLMInfo(
        provider=provider.value,
        model=model,
        llm_raw_text=llm_raw_text,
        llm_fen=process_llm_raw_text(llm_raw_text, extract_fen),
    )

    return PGN2FENLog(
        game_info=game_info,
        llm_info=llm_info,
    )


def run_experiment(
    pgn_input_files: list[Path],
    jsonl_write_file: str | None = None,
    print_to_console: bool = True,
    provider: Provider = Provider.GOOGLE,
    model: str = "gemini-2.0-flash-001",
    thinking_budget: int | None = None,
    extract_fen: bool = False,
    max_workers: int = 10,
) -> None:
    """
    Runs the PGN to FEN benchmark experiment on a list of PGN files.

    Args:
        pgn_input_files (list[Path]): List of PGN file paths to process.
        jsonl_write_file (str | None): Path to the output JSONL file.
        print_to_console (bool): Whether to print results to the console.
        provider (Provider): The LLM provider (e.g., GOOGLE, OPENAI).
        model (str): The model name to use for processing.
        thinking_budget (int | None): Optional thinking budget for the model. Currently
            only implemented for the GOOGLE provider.
        extract_fen (bool): Whether to extract FEN from raw LLM output, or use the text
            response directly as is.
        max_workers (int): Maximum number of workers for parallel processing.

    Returns:
        None
    """
    load_dotenv()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_pgn, pgn_file, provider, model, thinking_budget, extract_fen
            )
            for pgn_file in pgn_input_files
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            pgn2fen_experiment = future.result()

            if print_to_console:
                print(str(pgn2fen_experiment) + "\n")

            if jsonl_write_file:
                with open(jsonl_write_file, "a") as f:
                    f.write(json.dumps(asdict(pgn2fen_experiment)) + "\n")


def get_pgn_paths(
    pgn_dir: Path,
    jsonl_write_file: Path,
    start_index: int = 0,
    end_index: int = 1000,
    reprocess: bool = False,
) -> list[Path]:
    """
    Retrieves a list of PGN file paths to process, optionally skipping already processed files.

    Args:
        pgn_dir (Path): Directory containing PGN files.
        jsonl_write_file (Path): Path to the JSONL file with previous experiments.
        start_index (int): Start index for PGN files.
        end_index (int): End index for PGN files.
        reprocess (bool): Whether to reprocess files already processed.

    Returns:
        list[Path]: List of PGN file paths to process.
    """
    pgn_files = sorted(pgn_dir.glob("*.pgn"))[start_index:end_index]
    if not reprocess and jsonl_write_file.exists():
        previous_experiments = load_logs_from_jsonl(jsonl_write_file)
        pgn_files = [
            pgn_file
            for pgn_file in pgn_files
            if not any(
                experiment.game_info.input_pgn_file.endswith(pgn_file.name)
                for experiment in previous_experiments
            )
        ]
    return pgn_files


def parse_arguments():
    """
    Parses command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Run PGN to FEN benchmark experiments.")
    parser.add_argument(
        "--provider",
        type=str,
        default="GOOGLE",
        help="Provider name (e.g., GOOGLE, OPENAI, DEEPSEEK).",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-pro-preview-03-25", help="Model name."
    )
    parser.add_argument(
        "--thinking_budget", type=int, default=None, help="Thinking budget for the model."
    )
    parser.add_argument(
        "--reprocess_pgns",
        action="store_true",
        help="Reprocess PGN files even if they were processed before.",
    )
    parser.add_argument("--start_index", type=int, default=0, help="Start index for PGN files.")
    parser.add_argument("--end_index", type=int, default=1000, help="End index for PGN files.")
    parser.add_argument("--print_to_console", action="store_true", help="Print results to console.")
    parser.add_argument(
        "--extract_fen", action="store_true", help="Extract FEN from raw LLM output."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of workers for parallel processing.",
    )
    parser.add_argument(
        "--pgn_dir",
        type=str,
        default="data/WorldCup/truncated",
        help="Directory containing PGN files.",
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="Path to the output JSONL file."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    provider = Provider[args.provider.upper()]
    model = args.model
    thinking_budget = args.thinking_budget
    reprocess_pgns = args.reprocess_pgns
    start_index = args.start_index
    end_index = args.end_index
    print_to_console = args.print_to_console
    extract_fen = args.extract_fen
    max_workers = args.max_workers

    pgn_dir = PROJECT_ROOT / args.pgn_dir
    think_str = f"_think{thinking_budget}" if thinking_budget is not None else ""
    jsonl_write_file = (
        PROJECT_ROOT / args.output_file
        if args.output_file
        else PROJECT_ROOT / "model_logs" / f"{provider.value}_{model}{think_str}.jsonl"
    )

    pgn_input_files = get_pgn_paths(
        pgn_dir,
        jsonl_write_file=jsonl_write_file,
        start_index=start_index,
        end_index=end_index,
        reprocess=reprocess_pgns,
    )

    print(len(pgn_input_files), "PGN files to process")
    run_experiment(
        jsonl_write_file=jsonl_write_file,
        pgn_input_files=pgn_input_files,
        print_to_console=print_to_console,
        provider=provider,
        model=model,
        thinking_budget=thinking_budget,
        extract_fen=extract_fen,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    main()
