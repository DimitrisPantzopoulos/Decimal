from typing import ClassVar, TypeAlias
from dataclasses import dataclass, field

import numpy as np

Bitboard : TypeAlias = int
Color    : TypeAlias = int
Cell     : TypeAlias = int
Offset   : TypeAlias = tuple[int, int]
Coord    : TypeAlias = tuple[int, int]

BLACK : Color = 0
WHITE : Color = 1

DELTAS : tuple[Offset, ...] = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))

@dataclass(slots=True, frozen=True)
class HexBoard:
    size  : int      = 11
    white : Bitboard = 0
    black : Bitboard = 0
    stm   : Color    = BLACK

    _neighbor_masks_cache : tuple[Bitboard, ...]      = field(init=False, repr=False, compare=False)
    _black_edges_cache    : tuple[Bitboard, Bitboard] = field(init=False, repr=False, compare=False)
    _white_edges_cache    : tuple[Bitboard, Bitboard] = field(init=False, repr=False, compare=False)
    _winner_cache         : Color | None      = field(init=False, default=None, repr=False, compare=False)
    _winner_cached        : bool              = field(init=False, default=False, repr=False, compare=False)
    _is_terminal_cache    : bool              = field(init=False, default=False, repr=False, compare=False)
    _is_terminal_cached   : bool              = field(init=False, default=False, repr=False, compare=False)
    _encoded_board_cache  : np.ndarray | None = field(init=False, default=None, repr=False, compare=False)

    _SHARED_NEIGHBOR_MASKS: ClassVar[dict[int, tuple[Bitboard, ...]]] = {}
    _SHARED_EDGE_MASKS    : ClassVar[
        dict[int, tuple[tuple[Bitboard, Bitboard], tuple[Bitboard, Bitboard]]]
    ] = {}

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("size must be >= 1")
        if self.stm not in (BLACK, WHITE):
            raise ValueError("stm must be BLACK or WHITE")
        if self.white & self.black:
            raise ValueError("white and black bitboards overlap")
        if (self.white | self.black) & ~self.full_mask:
            raise ValueError("stones outside board mask")

        object.__setattr__(self, "_neighbor_masks_cache", self._get_shared_neighbor_masks(self.size))
        black_edges, white_edges = self._get_shared_edge_masks(self.size)
        object.__setattr__(self, "_black_edges_cache", black_edges)
        object.__setattr__(self, "_white_edges_cache", white_edges)

    @staticmethod
    def _neighbor_mask_for_size(size: int, index: Cell) -> Bitboard:
        row, col = divmod(index, size)
        neighbors: Bitboard = 0

        for dr, dc in DELTAS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < size and 0 <= nc < size:
                neighbors |= 1 << (nr * size + nc)

        return neighbors

    @classmethod
    def _get_shared_neighbor_masks(cls, size: int) -> tuple[Bitboard, ...]:
        cached = cls._SHARED_NEIGHBOR_MASKS.get(size)
        if cached is None:
            cached = tuple(cls._neighbor_mask_for_size(size, i) for i in range(size * size))
            cls._SHARED_NEIGHBOR_MASKS[size] = cached
        return cached

    @classmethod
    def _get_shared_edge_masks(
        cls,
        size: int,
    ) -> tuple[tuple[Bitboard, Bitboard], tuple[Bitboard, Bitboard]]:
        cached = cls._SHARED_EDGE_MASKS.get(size)
        if cached is not None:
            return cached

        black_start: Bitboard = 0
        black_goal: Bitboard = 0
        for c in range(size):
            black_start |= 1 << c
            black_goal |= 1 << ((size - 1) * size + c)

        white_start: Bitboard = 0
        white_goal: Bitboard = 0
        for r in range(size):
            white_start |= 1 << (r * size)
            white_goal |= 1 << (r * size + (size - 1))

        cached = ((black_start, black_goal), (white_start, white_goal))
        cls._SHARED_EDGE_MASKS[size] = cached
        return cached

    @property
    def num_cells(self) -> int:
        return self.size * self.size

    @property
    def full_mask(self) -> Bitboard:
        return (1 << self.num_cells) - 1

    @property
    def occupied(self) -> Bitboard:
        return self.white | self.black

    @property
    def empty(self) -> Bitboard:
        return self.full_mask & ~self.occupied

    def to_move(self) -> Color:
        return self.stm

    def other(self, color : Color) -> Color:
        return WHITE if color == BLACK else BLACK

    def idx(self, row : int, col : int) -> Cell:
        if not self.in_bounds(row, col):
            raise IndexError("row/col out of bounds")
        return row * self.size + col

    def rc(self, index : Cell) -> Coord:
        if not 0 <= index < self.num_cells:
            raise IndexError("cell index out of bounds")
        return divmod(index, self.size)

    def in_bounds(self, row : int, col : int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def bit(self, index : Cell) -> Bitboard:
        if not 0 <= index < self.num_cells:
            raise IndexError("cell index out of bounds")
        return 1 << index

    def bit_rc(self, row : int, col : int) -> Bitboard:
        return self.bit(self.idx(row, col))

    def color_at(self, index : Cell) -> Color | None:
        cell : Bitboard = self.bit(index)

        if self.black & cell:
            return BLACK
        if self.white & cell:
            return WHITE
        return None

    def is_legal(self, index : Cell) -> bool:
        move : Bitboard = 1 << index
        return 0 <= index < self.num_cells and not (self.occupied & move)

    def legal_moves(self) -> list[Cell]:
        moves : list[Cell] = []
        work  : Bitboard = self.empty

        while work:
            lsb: Bitboard = work & -work
            work ^= lsb
            moves.append(lsb.bit_length() - 1)

        return moves

    def play(self, index : Cell) -> "HexBoard":
        if not self.is_legal(index):
            raise ValueError(f"illegal move at index {index}")

        move : Bitboard = 1 << index

        if self.stm == BLACK:
            return HexBoard(
                size  = self.size,
                white = self.white,
                black = self.black | move,
                stm   = WHITE,
            )

        return HexBoard(
            size  = self.size,
            white = self.white | move,
            black = self.black,
            stm   = BLACK,
        )

    def play_rc(self, row : int, col : int) -> "HexBoard":
        return self.play(self.idx(row, col))

    def _neighbor_mask(self, index: Cell) -> Bitboard:
        return self._neighbor_masks_cache[index]

    def _edge_masks(self, color : Color) -> tuple[Bitboard, Bitboard]:
        return self._black_edges_cache if color == BLACK else self._white_edges_cache

    def _has_connection(self, color : Color) -> bool:
        stones : Bitboard = self.black if color == BLACK else self.white
        start_edge, goal_edge = self._edge_masks(color)

        frontier : Bitboard = stones & start_edge
        visited  : Bitboard = 0

        while frontier:
            if frontier & goal_edge:
                return True

            visited |= frontier
            expand : Bitboard = 0
            work   : Bitboard = frontier

            while work:
                lsb : Bitboard = work & -work
                work ^= lsb
                index : Cell = lsb.bit_length() - 1
                expand |= self._neighbor_mask(index)

            frontier = expand & stones & ~visited

        return False

    def winner(self) -> Color | None:
        if self._winner_cached:
            return self._winner_cache

        winner : Color | None = None
        if self._has_connection(BLACK):
            winner = BLACK
        elif self._has_connection(WHITE):
            winner = WHITE

        object.__setattr__(self, "_winner_cache", winner)
        object.__setattr__(self, "_winner_cached", True)
        return winner

    def is_terminal(self) -> bool:
        if self._is_terminal_cached:
            return self._is_terminal_cache

        terminal : bool = self.winner() is not None
        object.__setattr__(self, "_is_terminal_cache", terminal)
        object.__setattr__(self, "_is_terminal_cached", True)
        return terminal
    
    def encode_board(self) -> np.ndarray:
        if self._encoded_board_cache is not None:
            return self._encoded_board_cache

        encoded_board : np.ndarray = np.zeros(shape=(3, self.size, self.size), dtype=np.float32)

        current_stones  : Bitboard = self.black if self.stm == BLACK else self.white
        opponent_stones : Bitboard = self.white if self.stm == BLACK else self.black

        current_copy : Bitboard = current_stones
        while current_copy:
            lsb : Bitboard = current_copy & -current_copy
            current_copy ^= lsb
            index : Cell = lsb.bit_length() - 1
            row, col = self.rc(index)
            encoded_board[0, row, col] = 1.0

        opponent_copy : Bitboard = opponent_stones
        while opponent_copy:
            lsb : Bitboard = opponent_copy & -opponent_copy
            opponent_copy ^= lsb
            index : Cell = lsb.bit_length() - 1
            row, col = self.rc(index)
            encoded_board[1, row, col] = 1.0

        legal_work: Bitboard = self.empty
        while legal_work:
            lsb : Bitboard = legal_work & -legal_work
            legal_work ^= lsb
            legal_move : Cell = lsb.bit_length() - 1
            row, col = self.rc(legal_move)
            encoded_board[2, row, col] = 1.0

        encoded_board.setflags(write=False)
        object.__setattr__(self, "_encoded_board_cache", encoded_board)
        return encoded_board
            






