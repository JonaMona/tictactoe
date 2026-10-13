
from functools import lru_cache

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

BOARD_SIZE = 9
FULL_BOARD_MASK = (1 << BOARD_SIZE) - 1
MOVE_MASKS = tuple(1 << i for i in range(BOARD_SIZE))

PLAYER_X = "X"
PLAYER_O = "O"
MAX_MOVES = 18
MAX_VISIBLE_MOVES = 6

# Higher values represent better outcomes.
LOSS, DRAW, WIN = 0, 1, 2

# Winning-line bitmasks: rows, columns, and diagonals.
WINNING_LINES = (
    0b000000111, 0b000111000, 0b111000000,
    0b001001001, 0b010010010, 0b100100100,
    0b100010001, 0b001010100,
)

# Convert an opponent's outcome to the current player's perspective.
REVERSE_OUTCOME = (WIN, DRAW, LOSS)


# ---------------------------------------------------------------------
# Board utilities
# ---------------------------------------------------------------------

def get_bit(bitboard: int, index: int) -> int:
    """Return 1 if the specified cell is occupied, otherwise 0."""
    return (bitboard >> index) & 1


def has_won(bitboard: int) -> bool:
    """Preserve the original behavior: match a winning line exactly."""
    return any(bitboard == line for line in WINNING_LINES)


def winner(p1: int, p2: int) -> bool | None:
    """Return True for X, False for O, or None if neither has won."""
    if has_won(p1):
        return True
    if has_won(p2):
        return False
    return None


# ---------------------------------------------------------------------
# Game solver
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def solve(
    p1: int,
    p2: int,
    order: tuple[int, ...],
    moves: int,
    is_x_turn: bool,
) -> tuple[int | None, int, int, tuple[float, float, float]]:
    """
    Evaluate a position using optimal play and random-play probabilities.

    Returns:
        (best_move, outcome, moves_until_outcome, probabilities)

        best_move:
            Index of the selected move, or None if the game has ended.

        outcome:
            Best achievable outcome from the current player's perspective.

        moves_until_outcome:
            Distance to the outcome selected by the solver.

        probabilities:
            (loss, draw, win) probabilities under uniformly random play.

    Move priorities:
        1. Prefer a win over a draw, and a draw over a loss.
        2. Among wins, prefer the shortest.
        3. Among losses, prefer the longest.
        4. Use random-play expected value to break eligible ties.

    The six most recent moves determine the visible board. The game
    ends in a draw after MAX_MOVES plies unless a winning line is found.
    """

    # A winning line means the previous player won.
    if winner(p1, p2) is not None:
        return None, LOSS, 0, (1.0, 0.0, 0.0)

    if moves >= MAX_MOVES:
        return None, DRAW, 0, (0.0, 1.0, 0.0)

    # Preserve the original ordering: determine occupied cells before
    # removing the oldest mark from the visible board.
    occupied = p1 | p2

    # Remove the oldest visible mark before placing the next mark.
    if len(order) == MAX_VISIBLE_MOVES:
        oldest_mask = MOVE_MASKS[order[0]]

        if is_x_turn:
            p1 &= ~oldest_mask
        else:
            p2 &= ~oldest_mask

        order = order[1:]

    best_move = None
    best_outcome = LOSS
    best_distance = -1
    best_random_value = -1.0

    loss_sum = 0.0
    draw_sum = 0.0
    win_sum = 0.0
    legal_moves = 0

    # Iterate over the cells considered available by the original solver.
    legal_mask = FULL_BOARD_MASK ^ occupied

    while legal_mask:
        # Extract and remove the lowest available bit.
        move_mask = legal_mask & -legal_mask
        legal_mask ^= move_mask
        index = move_mask.bit_length() - 1

        # Apply the move and evaluate the resulting position.
        if is_x_turn:
            child = solve(
                p1 | move_mask,
                p2,
                order + (index,),
                moves + 1,
                False,
            )
        else:
            child = solve(
                p1,
                p2 | move_mask,
                order + (index,),
                moves + 1,
                True,
            )

        _, child_outcome, child_distance, child_probs = child

        # Convert the result and probabilities to our perspective.
        outcome = REVERSE_OUTCOME[child_outcome]

        loss_probability = child_probs[2]
        draw_probability = child_probs[1]
        win_probability = child_probs[0]

        loss_sum += loss_probability
        draw_sum += draw_probability
        win_sum += win_probability
        legal_moves += 1

        # Expected value: loss = 0, draw = 0.5, win = 1.
        random_value = draw_probability * 0.5 + win_probability

        # First compare guaranteed outcomes.
        better = outcome > best_outcome

        if outcome == best_outcome:
            if outcome == WIN:
                better = (
                    best_distance == -1
                    or child_distance < best_distance
                )
            elif outcome == LOSS:
                better = child_distance > best_distance
            else:
                better = random_value > best_random_value

            # Preserve the original random-value tie-breaking rule.
            if (
                outcome != DRAW
                and child_distance == best_distance
                and random_value > best_random_value
            ):
                better = True

        if better:
            best_move = index
            best_outcome = outcome
            best_distance = child_distance
            best_random_value = random_value

    if best_move is None or legal_moves == 0:
        display_board(p1, p2)
        raise RuntimeError("No legal move found in a non-terminal position.")

    # Average over all legal moves, not just the optimal move.
    probabilities = (
        loss_sum / legal_moves,
        draw_sum / legal_moves,
        win_sum / legal_moves,
    )

    return best_move, best_outcome, best_distance + 1, probabilities


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def display_board(p1: int, p2: int) -> None:
    """Print the board, using blank cells for unoccupied positions."""
    for row in range(3):
        cells = []

        for col in range(3):
            mask = MOVE_MASKS[row * 3 + col]

            if p1 & mask:
                cells.append(PLAYER_X)
            elif p2 & mask:
                cells.append(PLAYER_O)
            else:
                cells.append(" ")

        print(" | ".join(cells))

        if row < 2:
            print("--+---+--")


def get_input_from_order(
    complete_order: tuple[int, ...],
) -> tuple[int, int, tuple[int, ...], bool]:
    """Reconstruct the visible board and next turn from move history."""
    move_count = len(complete_order)
    is_x_turn = move_count % 2 == 0
    recent_moves = complete_order[-MAX_VISIBLE_MOVES:]

    p1 = 0
    p2 = 0

    # Only the most recent six marks remain visible.
    first_index = max(0, move_count - MAX_VISIBLE_MOVES)

    for index in range(first_index, move_count):
        mask = MOVE_MASKS[complete_order[index]]

        if index % 2 == 0:
            p1 |= mask
        else:
            p2 |= mask

    return p1, p2, recent_moves, is_x_turn


# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    order = (1,)

    p1, p2, order, turn = get_input_from_order(order)

    print("Current board:")
    display_board(p1, p2)

    print(solve(p1, p2, order, 0, turn))

    # Inspect how effectively memoization is working.
    print("Cache statistics:", solve.cache_info())
