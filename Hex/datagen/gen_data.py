from dataclasses import dataclass
from typing import List

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import argparse
import random
import sys

try:
    from tqdm.auto import tqdm
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, *args, **kwargs):
            return iterable

try:
    from ..Board import HexBoard, Cell, Color
    from ..MCTS.MCTS import BaseMCTS
except ImportError:

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from Hex.Board import HexBoard, Cell, Color
    from Hex.MCTS.MCTS import BaseMCTS

@dataclass(slots=True, frozen=True)
class Memory:
    state  : np.ndarray
    policy : np.ndarray
    value  : np.float32

@dataclass(slots=True, frozen=True)
class GameMemory(Memory):
    stm : Color

def self_play_game(mcts_iters : int = 100, size : int = 11) -> List[Memory]:
    memories : List[GameMemory] = []

    mcts  : BaseMCTS = BaseMCTS()
    board : HexBoard = HexBoard(size=size)

    while not board.is_terminal():
        state  : np.ndarray = board.encode_board().copy()
        move   : Cell = mcts.search(root_state=board, iters=mcts_iters)
        policy : np.ndarray = mcts.get_root_visit_distribution().copy().astype(dtype=np.float32)

        memories.append(GameMemory(
            state  = state,
            policy = policy,
            value  = np.float32(0.0),
            stm    = board.stm
        ))

        if not board.is_legal(index=move):
            raise ValueError(f"MCTS produced an illegal move; {move}")
        
        board = board.play(index=move)

    winner : Color | None = board.winner()

    if winner is None:
        raise RuntimeError("terminal Hex position must have a winner")
    
    return [
        Memory(
            state  = memory.state,
            policy = memory.policy,
            value  = np.float32(1.0 if memory.stm == winner else -1.0),
        )
        for memory in memories
    ]


def _play_one_game_worker(args : tuple[int, int, int | None]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mcts_iters, size, seed = args

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    memories : List[Memory] = self_play_game(mcts_iters=mcts_iters, size=size)

    states   : np.ndarray = np.stack([memory.state for memory in memories], axis=0).astype(np.float32, copy=False)
    policies : np.ndarray = np.stack([memory.policy for memory in memories], axis=0).astype(np.float32, copy=False)
    values   : np.ndarray = np.asarray([memory.value for memory in memories], dtype=np.float32)

    return states, policies, values


def generate_dataset(
    num_games : int,
    mcts_iters : int = 100,
    size : int = 11,
    workers : int = 1,
    backend : str = "process",
    chunksize : int = 1,
    seed : int | None = None,
    output : str | None = None,
) -> Path:
    if num_games < 1:
        raise ValueError("num_games must be >= 1")
    if mcts_iters < 1:
        raise ValueError("mcts_iters must be >= 1")
    if size < 1:
        raise ValueError("size must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if chunksize < 1:
        raise ValueError("chunksize must be >= 1")
    if backend not in ("process", "thread"):
        raise ValueError("backend must be 'process' or 'thread'")

    data_dir : Path = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if output is None:
        stamp : str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"hex_data_{stamp}_g{num_games}_i{mcts_iters}_s{size}.npz"

    out_path : Path = data_dir / output

    game_args : list[tuple[int, int, int | None]] = []
    for i in range(num_games):
        game_seed : int | None = None if seed is None else seed + i
        game_args.append((mcts_iters, size, game_seed))

    states_batches : list[np.ndarray] = []
    policy_batches : list[np.ndarray] = []
    value_batches  : list[np.ndarray] = []

    executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    with executor_cls(max_workers=workers) as executor:
        futures = [executor.submit(_play_one_game_worker, args) for args in game_args]
        for future in tqdm(as_completed(futures), total=num_games, desc="self-play games", unit="game"):
            states, policies, values = future.result()
            states_batches.append(states)
            policy_batches.append(policies)
            value_batches.append(values)

    states_all : np.ndarray = np.concatenate(states_batches, axis=0)
    policies_all : np.ndarray = np.concatenate(policy_batches, axis=0)
    values_all : np.ndarray = np.concatenate(value_batches, axis=0)

    np.savez_compressed(
        out_path,
        states=states_all,
        policies=policies_all,
        values=values_all,
        board_size=np.int32(size),
        mcts_iters=np.int32(mcts_iters),
        num_games=np.int32(num_games),
    )

    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Hex self-play training data")
    parser.add_argument("--games", type=int, default=100, help="number of self-play games")
    parser.add_argument("--iters", type=int, default=100, help="MCTS iterations per move")
    parser.add_argument("--size", type=int, default=11, help="Hex board size")
    parser.add_argument("--workers", type=int, default=1, help="number of parallel workers")
    parser.add_argument("--backend", type=str, default="process", choices=("process", "thread"), help="parallel backend")
    parser.add_argument("--chunksize", type=int, default=1, help="task chunksize for worker map")
    parser.add_argument("--seed", type=int, default=None, help="base random seed")
    parser.add_argument("--output", type=str, default=None, help="output filename under datagen/data")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    out_path : Path = generate_dataset(
        num_games=args.games,
        mcts_iters=args.iters,
        size=args.size,
        workers=args.workers,
        backend=args.backend,
        chunksize=args.chunksize,
        seed=args.seed,
        output=args.output,
    )
    print(f"saved dataset: {out_path}")


if __name__ == "__main__":
    main()
    
