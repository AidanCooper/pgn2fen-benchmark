from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class Provider(Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


@dataclass
class PGNGameInfo:
    datetime: str
    input_pgn_file: str
    input_fen: str
    number_of_halfmoves: int


@dataclass
class LLMInfo:
    provider: str
    model: str
    llm_raw_text: str
    llm_fen: str | None
    llm_reasoning: str | None = None


@dataclass
class FEN:
    piece_placement: str
    turn: str | None
    castling: str | None
    en_passant: str | None
    halfmove_clock: int
    fullmove_number: int

    def compare_to(
        self,
        other: "FEN",
        en_passant_comparison_func: Callable[[str | None, str | None], bool] | None = None,
        fullmove_number_comparison_func: Callable[[int, int], bool] | None = None,
    ) -> dict[str, bool]:
        return {
            "piece_placement": self.piece_placement == other.piece_placement,
            "turn": self.turn == other.turn,
            "castling": self.castling == other.castling,
            "en_passant": (
                self.en_passant == other.en_passant
                if en_passant_comparison_func is None
                else en_passant_comparison_func(self.en_passant, other.en_passant)
            ),
            "halfmove_clock": self.halfmove_clock == other.halfmove_clock,
            "fullmove_number": (
                self.fullmove_number == other.fullmove_number
                if fullmove_number_comparison_func is None
                else fullmove_number_comparison_func(self.fullmove_number, other.fullmove_number)
            ),
        }


@dataclass
class FENEvaluation:
    full_correctness: bool
    piece_placement: bool
    turn: bool
    castling: bool
    en_passant: bool
    halfmove_clock: bool
    fullmove_number: bool

    def __str__(self) -> str:
        return (
            f"Full Correctness : {self.full_correctness}\n"
            f"    Piece Placement : {self.piece_placement}\n"
            f"    Turn            : {self.turn}\n"
            f"    Castling        : {self.castling}\n"
            f"    En Passant      : {self.en_passant}\n"
            f"    Halfmove Clock  : {self.halfmove_clock}\n"
            f"    Fullmove Number : {self.fullmove_number}"
        )


@dataclass
class PGN2FENLog:
    game_info: PGNGameInfo
    llm_info: LLMInfo

    def __str__(self) -> str:
        out = f"Results for {self.game_info.input_pgn_file}:\n"
        out += f"    Original FEN : {self.game_info.input_fen}\n"
        out += f"    LLM FEN      : {self.llm_info.llm_fen}\n"
        return out
