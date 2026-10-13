PLAYER_X = "X"
PLAYER_O = "O"

def get_bit(bitboard: int, index: int) -> int:
    """Return 1 if the specified cell is occupied by this player."""
    return (bitboard >> index) & 1

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
                cells.append(" ")

        print(" | ".join(cells))

        if row < 2:
            print("--+---+--")

def getInputFromOrder(order_complete: tuple[int, ...]) -> tuple[int, int, tuple[int, ...], bool]:
    turn = True if len(order_complete) % 2 == 0 else False
    order = order_complete[-6:] # only view the last 6 elements
    p1 = 0b000000000
    p2 = 0b000000000
    if len(order_complete) > 6:
        is_x_turn = turn
    else:
        is_x_turn = True
    for value in order:
        move_mask = 1 << value
        if is_x_turn:
            p1 = p1 | move_mask
        else:
            p2 = p2 | move_mask
        is_x_turn = not is_x_turn

    return p1, p2, order, turn

if __name__ == "__main__":
    order = (1,6,4,7,8,0,2,5,6,4,3,8,0,)
    for i in range(1, len(order)+1):
        p1, p2, order_cut, turn = getInputFromOrder(order[:i])
        print(f"Next board")
        display_board(p1, p2)