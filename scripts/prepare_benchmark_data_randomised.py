"""This script prepares benchmark data by creating PGN files from randomised move sequences."""

import random
import shutil
from pathlib import Path

import chess
import chess.pgn

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_random_pgn(
    board: chess.Board,
    n_halfmoves: int,
    output_file: Path,
) -> None:
    if n_halfmoves < 1:
        raise ValueError("Number of halfmoves must be at least 1.")

    while True:
        board = chess.Board()
        game = chess.pgn.Game()
        node = game
        for n in range(1, n_halfmoves + 1):
            move = random.choice(list(board.legal_moves))
            board.push(move)
            node = node.add_variation(move)
            if board.is_game_over():
                break
        if board.fullmove_number * 2 - (0 if board.turn else 1) >= n_halfmoves:
            break

    game.headers["Event"] = "World Cup"
    game.headers["Site"] = "NYC"
    game.headers["Date"] = "2010.10.10"
    game.headers["Round"] = "1"
    game.headers["White"] = "Kasparov,G"
    game.headers["Black"] = "Carlsen,M"
    game.headers["Result"] = "*"
    game.headers["WhiteElo"] = "2850"
    game.headers["BlackElo"] = "2850"

    with open(output_file, "w", encoding="utf-8") as f:
        print(game, file=f)


def main():
    OUTPUT_DIR = PROJECT_ROOT / "data" / "randomised"
    TARGET_FILES_PER_HALFMOVE = 10
    MAX_HALFMOVES = 100

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    for n_halfmoves in range(1, MAX_HALFMOVES + 1):
        for n in range(1, TARGET_FILES_PER_HALFMOVE + 1):
            output_file = OUTPUT_DIR / f"halfmoves{n_halfmoves:04}_{n:03}.pgn"
            generate_random_pgn(chess.Board(), n_halfmoves, output_file)


if __name__ == "__main__":
    main()
