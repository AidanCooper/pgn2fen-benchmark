"""This script prepares benchmark data by splitting and truncating PGN files."""

import shutil
from pathlib import Path

from pgn2fen.pgn_io import generate_truncated_pgns, split_pgn_into_individual_games


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def setup_directories(split_dir, truncated_dir):
    """Set up directories for split and truncated PGN files.

    Args:
        split_dir (Path): Path to the directory for split PGN files.
        truncated_dir (Path): Path to the directory for truncated PGN files.
    """
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)

    if truncated_dir.exists():
        shutil.rmtree(truncated_dir)
    truncated_dir.mkdir(parents=True)


def main():
    for benchmark in [
        # "WorldCup",
        "FischerRandom",
    ]:
        PGN_DIR = PROJECT_ROOT / "data" / benchmark
        SPLIT_DIR = PGN_DIR / "split"
        TRUNCATED_DIR = PGN_DIR / "truncated"
        TARGET_FILES_PER_HALFMOVE = 10
        MAX_HALFMOVES = 100
        HEADERS_TO_DELETE = [
            "WhiteTitle",
            "BlackTitle",
            "ECO",
            "Opening",
            "Variation",
            "WhiteFideId",
            "BlackFideId",
            "EventDate",
            "EventType",
        ]

        setup_directories(SPLIT_DIR, TRUNCATED_DIR)
        split_pgn_into_individual_games(PGN_DIR, SPLIT_DIR)
        input_files = list(SPLIT_DIR.glob("*.pgn"))
        if not input_files:
            raise ValueError("No PGN files found.")

        generate_truncated_pgns(
            input_files,
            TRUNCATED_DIR,
            HEADERS_TO_DELETE,
            max_halfmoves=MAX_HALFMOVES,
            target_per_halfmove=TARGET_FILES_PER_HALFMOVE,
            write_starting_fen=benchmark == "FischerRandom",
        )


if __name__ == "__main__":
    main()
