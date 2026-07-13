from ..Board import HexBoard, Cell, Color
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import random
import math

@dataclass(slots=True)
class MCTSNode:
    state       : HexBoard
    
    parent      : Optional['MCTSNode']
    action      : Optional[Cell]
    player      : Optional[Color]

    children    : List['MCTSNode'] = field(default_factory=list)
    legal_moves : List[Cell]       = field(init=False)
    visits      : int = 0
    wins        : int = 0

    def __post_init__(self) -> None:
        self.legal_moves = self.state.legal_moves()
    
    # Check if node is terminal
    def is_terminal(self) -> bool:
        return self.state.is_terminal()
    
    def is_fully_expanded(self) -> bool:
        return len(self.legal_moves) == 0
    
    def expand(self) -> 'MCTSNode':
        action : Cell = self.legal_moves.pop()
        
        new_state : HexBoard = self.state.play(index=action)

        child : 'MCTSNode' = MCTSNode(
            state  = new_state,
            parent = self,
            action = action,
            player = self.state.stm 
        )

        self.children.append(child)
        return child

    def best_child(self, c : float=1.41) -> 'MCTSNode':
        for child in self.children:
            if child.visits == 0:
                return child
        
        def ucb(child : MCTSNode) -> float:
            exploit : float = child.wins / child.visits
            explore : float = c * math.sqrt(math.log(self.visits) / child.visits)
            return exploit + explore
        
        return max(self.children, key=ucb)

    @staticmethod
    def _sample_random_empty_cell(state: HexBoard) -> Cell:
        empty : int = state.empty

        if empty == 0:
            raise ValueError("cannot sample move from full board")

        target_rank : int = random.randrange(empty.bit_count())
        work        : int = empty

        for _ in range(target_rank):
            work &= work - 1

        lsb : int = work & -work
        return lsb.bit_length() - 1
    
    def rollout(self) -> Color:
        curr_state : HexBoard = self.state

        while True:
            winner : Color | None = curr_state.winner()

            if winner is not None:
                return winner

            action: Cell = self._sample_random_empty_cell(curr_state)
            curr_state = curr_state.play(action)
       
    def backpropagate(self, winner : Color) -> None:
        self.visits += 1

        # There are no draws in Hex so we can only check for wins
        if winner == self.player:
            self.wins += 1

        if self.parent:
            self.parent.backpropagate(winner=winner)

class BaseMCTS:
    def __init__(self) -> None:
        self.root : MCTSNode | None = None

    def get_root_visit_distribution(self) -> np.ndarray:
        if self.root is None:
            raise ValueError("root is not initialized; call search(...) first")

        dist : np.ndarray = np.zeros(self.root.state.num_cells, dtype=np.float64)

        for child in self.root.children:
            if child.action is not None:
                dist[child.action] = float(child.visits)

        total : float = float(dist.sum())

        if total > 0.0:
            return dist / total

        # Fallback if no visits were accumulated yet.
        legal_moves : List[Cell] = self.root.state.legal_moves()

        if not legal_moves:
            return dist

        p : float = 1.0 / len(legal_moves)

        for move in legal_moves:
            dist[move] = p

        return dist
    
    def search(self, root_state : HexBoard, iters : int = 500) -> Cell:
        self.root = MCTSNode(
            state  = root_state,
            parent = None,
            action = None,
            player = None
        )

        for _ in range(iters):
            node : MCTSNode = self.root

            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()
            
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            winner : Color = node.rollout()
            node.backpropagate(winner=winner)

        best_child : MCTSNode = max(self.root.children, key=lambda c: c.visits)
        assert(best_child.action is not None)
        return best_child.action

