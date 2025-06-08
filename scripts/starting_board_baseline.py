"""Produces baseline results by predicting the starting board FEN for all inputs."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import chess

from pgn2fen.models import LLMInfo, PGN2FENLog, PGNGameInfo
from pgn2fen.pgn_io import parse_board_from_pgn_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    for benchmark_name, benchmark_pgns in [
        ("standard", PROJECT_ROOT / "data" / "WorldCup" / "truncated"),
        ("randomised", PROJECT_ROOT / "data" / "randomised"),
        ("fischer", PROJECT_ROOT / "data" / "FischerRandom" / "truncated"),
    ]:
        output_file = (
            PROJECT_ROOT / "model_logs" / f"baseline_starting_board_{benchmark_name}.jsonl"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for pgn_file in benchmark_pgns.glob("*.pgn"):
                board = parse_board_from_pgn_file(pgn_file)
                board_fen = board.fen()

                game_info = PGNGameInfo(
                    datetime=str(datetime.now()),
                    input_pgn_file=str(pgn_file.relative_to(PROJECT_ROOT)),
                    input_fen=board_fen,
                    number_of_halfmoves=len(board.move_stack),
                )
                llm_info = LLMInfo(
                    provider="baseline",
                    model="starting_board",
                    llm_raw_text=chess.Board().fen(),
                    llm_fen=chess.Board().fen(),
                )
                pgn2fen_log = PGN2FENLog(
                    game_info=game_info,
                    llm_info=llm_info,
                )

                f.write(json.dumps(asdict(pgn2fen_log)) + "\n")


if __name__ == "__main__":
    main()
