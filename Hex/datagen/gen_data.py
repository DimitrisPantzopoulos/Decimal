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
    from Decimal.Hex.Board import HexBoard, Cell, Color
    from Decimal.Hex.MCTS.MCTS import BaseMCTS

def _rotate_board_90(board: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotate board k times by 90 degrees counterclockwise."""
    for _ in range(k % 4):
        board = np.rot90(board, axes=(1, 2))
    return board

def _flip_board(board: np.ndarray, axis: int) -> np.ndarray:
    """Flip board along axis (1 for horizontal, 2 for vertical)."""
    return np.flip(board, axis=axis)

def _cell_index_transform(idx: int, size: int, rotation: int, flip_h: bool, flip_v: bool) -> int:
    """Transform a cell index according to board symmetry operation."""
    row, col = divmod(idx, size)
    
    # Apply rotation
    for _ in range(rotation % 4):
        row, col = col, size - 1 - row
    
    # Apply flips
    if flip_h:
        col = size - 1 - col
    if flip_v:
        row = size - 1 - row
    
    return row * size + col

def _permute_policy(policy: np.ndarray, size: int, rotation: int, flip_h: bool, flip_v: bool) -> np.ndarray:
    """Permute policy distribution to match board symmetry transformation."""
    num_cells = size * size
    permuted = np.zeros_like(policy)
    
    perm_indices = np.arange(num_cells).reshape(size, size)
    
    # Apply rotation
    for _ in range(rotation % 4):
        perm_indices = np.rot90(perm_indices)
    
    # Apply flips
    if flip_h:
        perm_indices = np.fliplr(perm_indices)
    if flip_v:
        perm_indices = np.flipud(perm_indices)
    
    permuted[perm_indices.flatten()] = policy
    return permuted

def _augment_dataset(
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate all 8 symmetry augmentations for each sample."""
    num_augmentations = 8  # 4 rotations + 4 reflections
    num_samples = len(states)
    total_samples = num_samples * num_augmentations
    
    augmented_states = np.zeros((total_samples, *states.shape[1:]), dtype=np.float32)
    augmented_policies = np.zeros((total_samples, *policies.shape[1:]), dtype=np.float32)
    augmented_values = np.zeros(total_samples, dtype=np.float32)
    
    idx = 0
    for i in range(num_samples):
        for rotation in range(4):
            for flip in [False, True]:
                # Rotate board
                aug_state = _rotate_board_90(states[i].copy(), rotation)
                
                # Apply one flip per symmetry (4 rotations + 4 flips = 8)
                if flip:
                    aug_state = _flip_board(aug_state, axis=2)
                
                # Permute policy to match
                aug_policy = _permute_policy(policies[i], size, rotation, flip, False)
                
                augmented_states[idx] = aug_state
                augmented_policies[idx] = aug_policy
                augmented_values[idx] = values[i]
                idx += 1

    return augmented_states, augmented_policies, augmented_values

def _board_state_hash(state: np.ndarray) -> bytes:
    """Create a hashable representation of a board state."""
    return state.tobytes()

def _deduplicate_data(
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove duplicate board states, averaging policy/value for duplicates."""
    unique_map: dict[bytes, tuple[list, list]] = {}

    for i in range(len(states)):
        state_hash = _board_state_hash(states[i])
        if state_hash not in unique_map:
            unique_map[state_hash] = ([], [])
        unique_map[state_hash][0].append(policies[i])
        unique_map[state_hash][1].append(values[i])

    unique_states = []
    unique_policies = []
    unique_values = []

    for state_hash, (policy_list, value_list) in unique_map.items():
        # Find the original state that maps to this hash
        for i in range(len(states)):
            if _board_state_hash(states[i]) == state_hash:
                unique_states.append(states[i])
                break
        unique_policies.append(np.mean(policy_list, axis=0).astype(np.float32))
        unique_values.append(np.mean(value_list, dtype=np.float32))

    return (
        np.array(unique_states, dtype=np.float32),
        np.array(unique_policies, dtype=np.float32),
        np.array(unique_values, dtype=np.float32),
    )

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

    initial_count = len(states_all)
    states_all, policies_all, values_all = _deduplicate_data(states_all, policies_all, values_all)
    final_count = len(states_all)
    print(f"Deduplicated: {initial_count} → {final_count} samples ({100*final_count/initial_count:.1f}%)")

    # Apply 8-fold symmetry augmentation
    states_all, policies_all, values_all = _augment_dataset(states_all, policies_all, values_all, size)
    print(f"Augmented: {final_count} → {len(states_all)} samples")

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
    
