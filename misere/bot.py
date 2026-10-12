
from functools import lru_cache

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

BOARD_SIZE = 9
FULL_BOARD_MASK = (1 << BOARD_SIZE) - 1

EMPTY = "."
PLAYER_X = "X"
PLAYER_O = "O"

# Outcomes are ordered so that larger values represent better results.
LOSS, DRAW, WIN = 0, 1, 2

# Each tuple contains the cell indices of one possible three-in-a-row.
WINNING_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
    (0, 4, 8), (2, 4, 6),             # Diagonals
)


# ---------------------------------------------------------------------
# Board utilities
# ---------------------------------------------------------------------

def get_turn(p1: int, p2: int) -> bool:
    """Return True if X is to move, otherwise False."""
    return p1.bit_count() == p2.bit_count()


def get_bit(bitboard: int, index: int) -> int:
    """Return 1 if the specified cell is occupied by this player."""
    return (bitboard >> index) & 1


def has_won(bitboard: int) -> bool:
    """Check whether a player's bitboard contains a winning line."""
    return any(
        all(get_bit(bitboard, index) for index in line)
        for line in WINNING_LINES
    )


def winner(p1: int, p2: int) -> bool | None:
    """
    Return False if X has 3 in a row/..., True if O has 3 in a row/..., or None if neither has.

    This function assumes that the supplied position has at most
    one winner.
    """
    if has_won(p1):
        return False
    if has_won(p2):
        return True
    return None


def is_game_over(occupied: int) -> bool:
    """Return True if all nine cells are occupied."""
    return occupied == FULL_BOARD_MASK


# ---------------------------------------------------------------------
# Game solver
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def solve(
    p1: int,
    p2: int,
    is_x_turn: bool,
) -> tuple[int | None, int, tuple[float, float, float]]:
    """
    Evaluate a position assuming optimal play.

    Args:
        p1: Bitboard containing X's marks.
        p2: Bitboard containing O's marks.
        is_x_turn: True if X is to move, otherwise False.

    Returns:
        best_move:
            Index (0-8) of the optimal move, or None if the game ended.

        outcome:
            Best achievable outcome from the current player's perspective:
            LOSS, DRAW, or WIN.

        probabilities:
            (loss, draw, win) probabilities under uniformly random play.
            These probabilities are independent of the optimal move.

    The optimal outcome takes priority over the random-play probabilities.
    If several moves have the same optimal outcome, the solver prefers
    the move with the highest expected value under random play.
    """

    # A winning line means the previous player lost.
    # Therefore, the player whose turn it is has won.
    if winner(p1, p2) is not None:
        return None, WIN, (0.0, 0.0, 1.0)

    occupied = p1 | p2

    # A full board without a winning line is a draw.
    if is_game_over(occupied):
        return None, DRAW, (0.0, 1.0, 0.0)

    best_move = None
    best_outcome = LOSS
    best_random_value = -1.0

    # Accumulate the probabilities resulting from uniformly random moves.
    probability_sums = [0.0, 0.0, 0.0]
    legal_moves = 0

    for index in range(BOARD_SIZE):
        move_mask = 1 << index

        # Skip occupied cells.
        if occupied & move_mask:
            continue

        # Apply the move and evaluate the resulting position.
        if is_x_turn:
            next_p1 = p1 | move_mask
            next_p2 = p2
        else:
            next_p1 = p1
            next_p2 = p2 | move_mask

        _, child_outcome, child_probabilities = solve(
            next_p1,
            next_p2,
            not is_x_turn,
        )

        # Convert the opponent's outcome into our perspective.
        outcome = {
            WIN: LOSS,
            DRAW: DRAW,
            LOSS: WIN,
        }[child_outcome]

        # Reverse (loss, draw, win) to obtain our probabilities.
        probabilities = child_probabilities[::-1]

        for i, probability in enumerate(probabilities):
            probability_sums[i] += probability

        legal_moves += 1

        # Expected value: loss = 0, draw = 0.5, win = 1.
        loss_probability, draw_probability, win_probability = probabilities
        random_value = draw_probability * 0.5 + win_probability

        # Prefer the best guaranteed outcome. Use random-play value
        # only to break ties between moves with equal outcomes.
        if (
            outcome > best_outcome
            or (
                outcome == best_outcome
                and random_value > best_random_value
            )
        ):
            best_move = index
            best_outcome = outcome
            best_random_value = random_value

    # Every non-terminal position must have at least one legal move.
    if best_move is None or legal_moves == 0:
        raise RuntimeError("No legal move found in a non-terminal position.")

    # Average over all legal moves to calculate random-play probabilities.
    probabilities = tuple(
        probability / legal_moves
        for probability in probability_sums
    )

    return best_move, best_outcome, probabilities


# ---------------------------------------------------------------------
# Input and output
# ---------------------------------------------------------------------

def read_board() -> tuple[int, int]:
    """
    Read and validate a board from three rows.

    Accepted characters:
        X - Player X
        O - Player O
        . or space - Empty cell

    Returns:
        A tuple containing X's and O's bitboards.
    """
    print("Enter the board row by row.")
    print("Use X, O, and . for empty cells (spaces also work).")

    while True:
        rows = [
            input(f"Row {row + 1}: ")
            for row in range(3)
        ]

        if any(len(row) != 3 for row in rows):
            print("Each row must contain exactly three characters.\n")
            continue

        board = "".join(rows).upper()

        if any(cell not in "XO. " for cell in board):
            print("Only X, O, . and spaces are allowed.\n")
            continue

        p1 = 0
        p2 = 0

        # Cell indices increase from left to right, then top to bottom.
        for index, cell in enumerate(board):
            if cell == PLAYER_X:
                p1 |= 1 << index
            elif cell == PLAYER_O:
                p2 |= 1 << index

        # X starts, so X can have either the same number of marks as O
        # or exactly one more.
        x_count = p1.bit_count()
        o_count = p2.bit_count()

        if x_count not in (o_count, o_count + 1):
            print("Invalid board: the numbers of X and O are inconsistent.\n")
            continue

        if has_won(p1) and has_won(p2):
            print("Invalid board: both players cannot have winning lines.\n")
            continue

        if has_won(p1) and x_count != o_count + 1:
            print("Invalid board: X cannot have won on O's turn.\n")
            continue

        if has_won(p2) and x_count != o_count:
            print("Invalid board: O cannot have won on X's turn.\n")
            continue

        return p1, p2


def display_board(p1: int, p2: int) -> None:
    """Print the board, showing cell indices in empty positions."""
    for row in range(3):
        cells = []

        for col in range(3):
            index = row * 3 + col

            if get_bit(p1, index):
                cells.append(PLAYER_X)
            elif get_bit(p2, index):
                cells.append(PLAYER_O)
            else:
                cells.append(str(index))

        print(" | ".join(cells))

        if row < 2:
            print("--+---+--")


def describe_outcome(outcome: int) -> str:
    """Convert an outcome constant into readable text."""
    return {
        LOSS: "lose",
        DRAW: "draw",
        WIN: "win",
    }[outcome]


def main() -> None:
    """Run the interactive Tic-Tac-Toe position evaluator."""
    print("Welcome to the Classic Tic-Tac-Toe Solver\n")

    p1, p2 = read_board()
    is_x_turn = get_turn(p1, p2)
    current_player = PLAYER_X if is_x_turn else PLAYER_O

    best_move, outcome, probabilities = solve(p1, p2, is_x_turn)

    print("\nCurrent board:")
    display_board(p1, p2)

    if best_move is None:
        print("\nThe game is already over.")
        print(f"Result for {current_player}: {describe_outcome(outcome)}.")
        return

    print(f"\nBest move for {current_player}: cell {best_move}")

    # Display the board after applying the recommended move.
    move_mask = 1 << best_move

    if is_x_turn:
        display_board(p1 | move_mask, p2)
    else:
        display_board(p1, p2 | move_mask)

    loss_probability, draw_probability, win_probability = probabilities

    print(
        f"\nWith optimal play, {current_player} will "
        f"{describe_outcome(outcome)}."
    )

    print("\nUnder uniformly random play from this position:")
    print(f"  Loss: {loss_probability:.2%}")
    print(f"  Draw: {draw_probability:.2%}")
    print(f"  Win:  {win_probability:.2%}")


if __name__ == "__main__":
    main()