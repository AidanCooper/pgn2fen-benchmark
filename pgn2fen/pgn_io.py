import json
import os
from pathlib import Path

import chess
import chess.pgn

from pgn2fen.models import LLMInfo, PGN2FENLog, PGNGameInfo


def parse_board_from_pgn_file(pgn_file_path: str | Path) -> chess.Board:
    """
    Parses the final board position from the first game in a PGN file.

    Args:
        pgn_file_path (str): Path to the PGN file.

    Returns:
        chess.Board: Final board state after all mainline moves.

    Raises:
        ValueError: If no valid game is found in the PGN file.
    """
    with open(pgn_file_path, encoding="utf-8") as pgn:
        game = chess.pgn.read_game(pgn)
        if game is None:
            raise ValueError(f"No valid game found in PGN file: {pgn_file_path}")

        board = game.board()
        for move in game.mainline_moves():
            board.push(move)

        return board


def load_logs_from_jsonl(results_jsonl: str) -> list[PGN2FENLog]:
    """
    Load the logs from a JSONL file.
    """

    logs = []
    with open(results_jsonl) as f:
        for line in f:
            if line == "\n":
                continue
            data = json.loads(line)
            game_info = PGNGameInfo(
                datetime=data["game_info"]["datetime"],
                input_pgn_file=data["game_info"]["input_pgn_file"],
                input_fen=data["game_info"]["input_fen"],
                number_of_halfmoves=data["game_info"]["number_of_halfmoves"],
            )
            llm_info = LLMInfo(
                provider=data["llm_info"]["provider"],
                model=data["llm_info"]["model"],
                llm_raw_text=data["llm_info"].get("llm_raw_text", data["llm_info"]["llm_fen"]),
                llm_fen=data["llm_info"]["llm_fen"],
            )
            logs.append(PGN2FENLog(game_info, llm_info))
    return logs


def split_pgn_file(input_pgn_path: str, output_dir: str) -> None:
    """
    Splits a PGN file containing multiple games into separate PGN files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_pgn_path, encoding="utf-8") as f:
        pgn_contents = f.read()

    filename = input_pgn_path.split("/")[-1][:-4]

    games = pgn_contents.strip().split("\n\n[")
    # Reattach the missing [ on every game except the first
    games = [games[0]] + ["[" + game for game in games[1:]]

    for idx, game in enumerate(games, 1):
        output_file = os.path.join(output_dir, f"{filename}_{idx}.pgn")
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(game.strip() + "\n")


def split_pgn_into_individual_games(pgn_dir: str, output_dir: str) -> None:
    pgn_files = [os.path.join(pgn_dir, f) for f in os.listdir(pgn_dir) if f.endswith(".pgn")]
    for pgn_file in pgn_files:
        split_pgn_file(pgn_file, output_dir)


def clean_headers(game: chess.pgn.Game, headers_to_delete: list[str]) -> chess.pgn.Game:
    """Remove unwanted headers and set result to in-progress."""
    for header in headers_to_delete:
        if header in game.headers:
            del game.headers[header]
    game.headers["Result"] = "*"
    return game


def truncate_game(game: chess.pgn.Game, halfmove_count: int) -> chess.pgn.Game | None:
    """Truncate a game to a specified number of halfmoves."""
    node = game
    moves_made = 0
    while moves_made < halfmove_count and node.variations:
        node = node.variation(0)
        moves_made += 1

    if moves_made < halfmove_count:
        return None  # Not enough moves, skip this game

    # Cut off further moves
    node.variations.clear()
    return game


def process_pgn_file(
    pgn_path: str | Path, halfmove_count: int, headers_to_delete: list[str]
) -> chess.pgn.Game | None:
    """
    Truncate a PGN file to a specified number of halfmoves and clean headers.
    """
    with open(pgn_path, encoding="utf-8") as f:
        game = chess.pgn.read_game(f)

    game = truncate_game(game, halfmove_count)
    if game is None:
        return None

    game = clean_headers(game, headers_to_delete)
    return game


def generate_truncated_pgns(
    input_files: list[str],
    truncated_dir: Path,
    headers_to_delete: list[str],
    max_halfmoves: int,
    target_per_halfmove: int,
) -> None:
    """
    Generate truncated PGN files with a specified number of halfmoves.
    """
    for halfmoves in range(1, max_halfmoves + 1):
        generated = 0
        input_idx = 0

        while generated < target_per_halfmove:
            pgn_path = input_files[input_idx % len(input_files)]
            input_idx += 1

            game = process_pgn_file(pgn_path, halfmoves, headers_to_delete)
            if game is None:
                continue

            output_file = f"halfmoves{halfmoves:04}_{generated + 1:03}.pgn"
            with open(truncated_dir / output_file, "w", encoding="utf-8") as f:
                f.write(str(game))

            generated += 1
