import re

from Levenshtein import ratio

from pgn2fen.models import FEN, FENEvaluation, PGN2FENLog
from pgn2fen.utils import is_fen


def is_en_passant(en_passant: str) -> bool:
    """
    Check if the given string is a valid en passant notation.
    """
    return bool(re.match(r"^[a-h][36]$", en_passant) or re.match(r"^[-]$", en_passant))


def compare_en_passant(true_en_passant: str | None, llm_en_passant: str | None) -> bool:
    """
    Lenient comparison of en passant notations, to allow for adoption of different
    notation conventions.

    The python-chess library uses the convention that en passant is only notated if an
    en passant move is possible. However, a more common convention is to always notate
    en passant, even if it is not possible.

    Args:
        true_en_passant (str | None): The true en passant notation.
        llm_en_passant (str | None): The generated en passant notation.

    Returns:
        bool: True if the en passant notations match, False otherwise.
    """
    if true_en_passant is None or llm_en_passant is None:
        return False
    if not is_en_passant(true_en_passant):
        raise ValueError(f"True en passant must be valid. Received: {true_en_passant}")
    if re.match(r"^[a-h][36]$", true_en_passant):
        return true_en_passant == llm_en_passant
    return is_en_passant(llm_en_passant)


def infer_missing_parts(parts: list[str]) -> list[str | None]:
    """
    Infer the missing parts of a FEN string based on the parts provided.
    """
    turn = None
    castling = None
    en_passant = None

    for part in parts[1:-2]:
        if part == "w" or part == "b":
            turn = part
        elif part in ["KQkq", "KQ", "kq", "K", "Q", "k", "q", "-"]:
            castling = part
        elif is_en_passant(part):
            en_passant = part

    return [turn, castling, en_passant]


def prepare_llm_fen(fen_str: str) -> FEN:
    """
    Process the LLM-generated FEN string and return a FEN object.

    This function tolerates malformed FEN strings by inferring which FEN components are
    missing. For FENs with fewer than 6 space-separated parts, it will attempt to align
    the components with the expected FEN format.

    Args:
        fen_str (str): The LLM-generated FEN string.

    Returns:
        FEN: A FEN object representing the parsed FEN string, including None values for
            missing components.
    """
    parts = fen_str.split()
    p1, p5, p6 = parts[0], parts[-2], parts[-1]
    if len(parts) < 3:
        raise ValueError(f"LLM FEN string must have at least 3 parts. Received: {fen_str}")
    elif len(parts) > 6:
        raise ValueError(f"LLM FEN string must have at most 6 parts. Received: {fen_str}")
    elif 3 <= len(parts) < 6:
        p2, p3, p4 = infer_missing_parts(parts)
    else:
        p2, p3, p4 = parts[1], parts[2], parts[3]

    return FEN(
        piece_placement=p1,
        turn=p2,
        castling=p3,
        en_passant=p4,
        halfmove_clock=int(p5),
        fullmove_number=int(p6),
    )


def prepare_true_fen(fen_str: str) -> FEN:
    """
    Convert a FEN string into a FEN object.
    """
    parts = fen_str.split()
    if len(parts) != 6:
        raise ValueError(f"Invalid FEN string must have 6 parts. Received: {fen_str}")
    return FEN(
        piece_placement=parts[0],
        turn=parts[1],
        castling=parts[2],
        en_passant=parts[3],
        halfmove_clock=int(parts[4]),
        fullmove_number=int(parts[5]),
    )


def compare_fens(true_fen: FEN, llm_fen: FEN) -> FENEvaluation:
    """
    Compare two FENs and return a FENEvaluation object indicating which parts match.

    This is configured to be lenient in the comparison of the en passant notation and
    the fullmove number, as both have a degree of amibiguity that is not addressed by
    a direct string comparison.

    Args:
        true_fen (str): The true FEN string. This must have 6 parts.
        llm_fen (str): The generated FEN string. This can have 3 to 6 parts.

    Returns:
        FENEvaluation: An object indicating which parts of the FEN strings match.

    Raises:
        ValueError: If the FEN strings do not have the expected number of parts.
    """
    comparison = true_fen.compare_to(
        llm_fen,
        en_passant_comparison_func=compare_en_passant,
        fullmove_number_comparison_func=lambda x, y: abs(x - y) <= 1,
    )
    return FENEvaluation(
        full_correctness=all(comparison.values()),
        **comparison,
    )


def get_counts_and_mean_n_halfmoves(
    logs: list[PGN2FENLog],
) -> tuple[dict[str, int], float]:
    """
    Get the counts of each evaluation type from the experiments, and the mean number of
    halfmoves.

    Args:
        logs (list[PGN2FENLog]): A list of PGN2FENLog objects.

    Returns:
        tuple[dict[str, int], float]: A tuple containing a dictionary with the counts of
            each evaluation type, and the mean number of halfmoves.
    """
    counts = {
        "n": 0,
        "full_correctness": 0,
        "piece_placement": 0,
        "turn": 0,
        "castling": 0,
        "en_passant": 0,
        "halfmove_clock": 0,
        "fullmove_number": 0,
    }
    n_halfmoves = []

    for log in logs:
        llm_fen = log.llm_info.llm_fen
        board_fen = log.game_info.input_fen

        if llm_fen is not None and is_fen(llm_fen):
            parsed_board_fen, parsed_llm_fen = prepare_true_fen(board_fen), prepare_llm_fen(llm_fen)
            evaluation = compare_fens(parsed_board_fen, parsed_llm_fen)
        else:
            evaluation = FENEvaluation(False, False, False, False, False, False, False)

        counts["n"] += 1
        n_halfmoves.append(log.game_info.number_of_halfmoves)
        if evaluation.full_correctness:
            counts["full_correctness"] += 1
        if evaluation.piece_placement:
            counts["piece_placement"] += 1
        if evaluation.turn:
            counts["turn"] += 1
        if evaluation.castling:
            counts["castling"] += 1
        if evaluation.en_passant:
            counts["en_passant"] += 1
        if evaluation.halfmove_clock:
            counts["halfmove_clock"] += 1
        if evaluation.fullmove_number:
            counts["fullmove_number"] += 1

    mean_n_halfmoves = sum(n_halfmoves) / len(n_halfmoves) if n_halfmoves else 0

    return counts, mean_n_halfmoves


def get_levenshtein_ratio(logs: list[PGN2FENLog]) -> float | None:
    """
    Compute the Levenshtein distance between the true FEN and the LLM-generated FEN.

    Args:
        logs (list[PGN2FENLog]): A list of PGN2FENLog objects.

    Returns:
        float: The average Levenshtein distance.
    """
    ratios = []
    for log in logs:
        llm_fen = log.llm_info.llm_fen
        board_fen = log.game_info.input_fen

        if llm_fen is not None and is_fen(llm_fen):
            ratios.append(ratio(llm_fen, board_fen))
        else:
            ratios.append(0)

    return sum(ratios) / len(ratios) * 100 if ratios else None


def get_metric(metric: str, logs: list[PGN2FENLog], n_dp: int = 1) -> float | None:
    """
    Compute the score for a given metric based on the logs.

    Args:
        metric (str): The metric to compute.
        logs (list[PGN2FENLog]): A list of PGN2FENLog objects.
        n_dp (int): The number of decimal places to round the result to.

    Returns:
        float | None: The metric value. Returns None if the logs are empty.

    Raises:
        ValueError: If the metric is unknown.
    """
    if metric in [
        "full_correctness",
        "piece_placement",
        "turn",
        "castling",
        "en_passant",
        "halfmove_clock",
        "fullmove_number",
    ]:
        counts, _ = get_counts_and_mean_n_halfmoves(logs)
        return None if counts["n"] == 0 else round(counts[metric] / counts["n"] * 100, n_dp)
    if metric == "levenshtein_ratio":
        lr = get_levenshtein_ratio(logs)
        return None if lr is None else round(lr, n_dp)
    else:
        raise ValueError(f"Unknown metric: {metric}")
