
from functools import lru_cache

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

BOARD_SIZE = 9
FULL_BOARD_MASK = (1 << BOARD_SIZE) - 1
MOVE_MASKS = tuple(1 << i for i in range(BOARD_SIZE))

PLAYER_X = "X"
PLAYER_O = "O"

LOSS, DRAW, WIN = 0, 1, 2

# Exact bitmasks for all winning lines.
WINNING_LINES = (
    0b000000111, 0b000111000, 0b111000000,
    0b001001001, 0b010010010, 0b100100100,
    0b100010001, 0b001010100,
)

# Convert the opponent's outcome to the current player's perspective.
REVERSE_OUTCOME = (WIN, DRAW, LOSS)


# ---------------------------------------------------------------------
# Board utilities
# ---------------------------------------------------------------------

def get_bit(bitboard: int, index: int) -> int:
    return (bitboard >> index) & 1


def has_won(bitboard: int) -> bool:
    # Preserve the original exact-match behavior.
    for line in WINNING_LINES:
        if bitboard == line:
            return True
    return False


def winner(p1: int, p2: int) -> bool | None:
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
    p1_blocked: int,
    p2: int,
    p2_blocked: int,
    order: tuple[int, ...],
    is_x_turn: bool,
) -> tuple[int | None, int, int, tuple[float, float, float]]:

    # A winning line means the previous player won.
    if winner(p1, p2) is not None:
        return None, LOSS, 0, (1.0, 0.0, 0.0)

    # Cells that are already occupied or have been used by this player
    # cannot be played again.
    blocked = p1_blocked if is_x_turn else p2_blocked
    unavailable = (p1 | p2) | blocked

    # No legal moves remain.
    if unavailable == FULL_BOARD_MASK:
        return None, DRAW, 0, (0.0, 1.0, 0.0)

    # Remove the oldest visible mark after six moves.
    # The cell remains blocked and therefore cannot be reused.
    if len(order) == 6:
        oldest_mask = MOVE_MASKS[order[0]]

        if is_x_turn:
            p1 ^= p1 & oldest_mask
        else:
            p2 ^= p2 & oldest_mask

        order = order[1:]

    best_move = None
    best_outcome = LOSS
    best_moves = -1
    best_random_value = -1.0

    loss_sum = 0.0
    draw_sum = 0.0
    win_sum = 0.0
    legal_moves = 0

    # Iterate only over legal cells instead of checking all nine cells.
    legal_mask = FULL_BOARD_MASK ^ unavailable

    while legal_mask:
        # Extract the lowest set bit.
        move_mask = legal_mask & -legal_mask
        legal_mask ^= move_mask
        index = move_mask.bit_length() - 1

        # Apply the move.
        if is_x_turn:
            child = solve(
                p1 | move_mask,
                p1_blocked | move_mask,
                p2,
                p2_blocked,
                order + (index,),
                False,
            )
        else:
            child = solve(
                p1,
                p1_blocked,
                p2 | move_mask,
                p2_blocked | move_mask,
                order + (index,),
                True,
            )

        _, child_outcome, child_moves, child_probs = child

        # Convert the child's outcome and probabilities to our perspective.
        outcome = REVERSE_OUTCOME[child_outcome]

        child_loss = child_probs[2]
        child_draw = child_probs[1]
        child_win = child_probs[0]

        loss_sum += child_loss
        draw_sum += child_draw
        win_sum += child_win
        legal_moves += 1

        random_value = child_draw * 0.5 + child_win

        # Preserve the original priorities:
        # 1. Prefer WIN over DRAW over LOSS.
        # 2. Among wins, prefer the quickest.
        # 3. Among losses, delay defeat as long as possible.
        # 4. Use random-play value to break the permitted ties.
        better = outcome > best_outcome

        if outcome == best_outcome:
            if outcome == WIN:
                better = best_moves == -1 or child_moves < best_moves
            elif outcome == LOSS:
                better = child_moves > best_moves
            else:
                better = random_value > best_random_value

            if (
                outcome != DRAW
                and child_moves == best_moves
                and random_value > best_random_value
            ):
                better = True

        if better:
            best_move = index
            best_outcome = outcome
            best_moves = child_moves
            best_random_value = random_value

    if best_move is None or legal_moves == 0:
        display_board(p1, p2)
        raise RuntimeError("No legal move found in a non-terminal position.")

    # Exact random-play probabilities: average across all legal moves.
    probabilities = (
        loss_sum / legal_moves,
        draw_sum / legal_moves,
        win_sum / legal_moves,
    )

    return best_move, best_outcome, best_moves + 1, probabilities


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def display_board(p1: int, p2: int) -> None:
    for row in range(3):
        cells = []

        for col in range(3):
            index = row * 3 + col

            if p1 & MOVE_MASKS[index]:
                cells.append(PLAYER_X)
            elif p2 & MOVE_MASKS[index]:
                cells.append(PLAYER_O)
            else:
                cells.append(str(index))

        print(" | ".join(cells))

        if row < 2:
            print("--+---+--")


def get_input_from_order(
    complete_order: tuple[int, ...],
) -> tuple[int, int, int, int, tuple[int, ...], bool]:
    """Reconstruct the bitboards from the complete move history."""
    is_x_turn = len(complete_order) % 2 == 0
    recent_moves = complete_order[-6:]

    p1_blocked = 0
    p2_blocked = 0

    # Reconstruct all previously used cells.
    for index, move in enumerate(complete_order):
        if index % 2 == 0:
            p1_blocked |= MOVE_MASKS[move]
        else:
            p2_blocked |= MOVE_MASKS[move]

    if len(complete_order) <= 6:
        p1 = p1_blocked
        p2 = p2_blocked
    else:
        p1 = 0
        p2 = 0

        # Reconstruct the visible marks using their original turn parity.
        first_recent_index = len(complete_order) - len(recent_moves)

        for offset, move in enumerate(recent_moves):
            if (first_recent_index + offset) % 2 == 0:
                p1 |= MOVE_MASKS[move]
            else:
                p2 |= MOVE_MASKS[move]

    return (
        p1, p1_blocked,
        p2, p2_blocked,
        recent_moves, is_x_turn,
    )


# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    order = (0,)

    p1, p1_blocked, p2, p2_blocked, order, turn = (
        get_input_from_order(order)
    )

    print("Cells X cannot use:")
    display_board(p1_blocked, 0)

    print("Cells O cannot use:")
    display_board(0, p2_blocked)

    print("Current board:")
    display_board(p1, p2)

    print(solve(p1, p1_blocked, p2, p2_blocked, order, turn))

#18.24s