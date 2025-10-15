# Connect Four (https://www.wikiwand.com/en/articles/Connect_Four)
# 
# Instrukcja przygotowania środowiska:
# pip:
#   pip install easyAI
#   python main.py
#
# uv:
#   uv run main.py
#
# Autorzy:
# - Hubert Jóźwiak
# - Kacper Olejnik

from easyAI import TwoPlayerGame, AI_Player, Negamax, Human_Player
from typing import List, Optional, Any


class ConnectFour(TwoPlayerGame):
    """
    Connect Four game

    Attributes:
        players (list): List of player numbers [1, 2]
        board (list): 2D list representing the game board (6 rows x 7 columns)
        current_player (int): The player whose turn it is (1 or 2)
    """

    def __init__(self, players: Optional[List[Any]] = None) -> None:
        """
        Initialize a new Connect Four game.

        Args:
            players (list, optional): List of players (Human_Player or AI_Player objects).
                                     Defaults to None.
        """
        self.players = players or [1, 2]
        self.board: List[List[int]] = [[0 for _ in range(7)] for _ in range(6)]
        self.current_player: int = 1

    def possible_moves(self) -> List[int]:
        """
        Return a list of possible moves (column numbers where a piece can be dropped).

        A move is possible if the column is not full (top row has a 0).

        Returns:
            list: List of valid column indices (0-6) where a piece can be placed.
        """
        return [col for col in range(7) if self.board[0][col] == 0]

    def make_move(self, column: int) -> None:
        """
        Drop a piece in the specified column.

        The piece falls to the lowest available row in that column.

        Args:
            column (int): The column number (0-6) where the piece should be dropped.
        """
        # Find the lowest empty row in the column
        for row in range(5, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                break

    def show(self) -> None:
        """
        Display the current state of the game board.

        Shows column numbers at the top and the board state with:
        - '.' for empty spaces
        - '1' for player 1's pieces
        - '2' for player 2's pieces
        """
        print("\n  " + " ".join(map(str, range(7))))
        print("  " + "-" * 13)
        for row in self.board:
            print(
                "| "
                + " ".join(["." if cell == 0 else str(cell) for cell in row])
                + " |"
            )
        print("  " + "-" * 13)

    def lose(self) -> bool:
        """
        Check if the current player has lost (opponent has won).

        A player wins by getting four of their pieces in a row horizontally,
        vertically, or diagonally.

        Returns:
            bool: True if the opponent has four in a row, False otherwise.
        """
        return self.find_four(self.opponent_index)

    def is_over(self) -> bool:
        """
        Check if the game is over.

        The game ends when either player has four in a row or the board is full.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.find_four(1) or self.find_four(2) or (self.possible_moves() == [])

    def find_four(self, player: int) -> bool:
        """
        Check if the specified player has four pieces in a row.

        Checks all possible horizontal, vertical, and diagonal combinations.

        Args:
            player (int): The player number (1 or 2) to check for.

        Returns:
            bool: True if the player has four in a row, False otherwise.
        """
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if all(self.board[row][col + i] == player for i in range(4)):
                    return True

        # Check vertical
        for row in range(3):
            for col in range(7):
                if all(self.board[row + i][col] == player for i in range(4)):
                    return True

        # Check diagonal (bottom-left to top-right)
        for row in range(3, 6):
            for col in range(4):
                if all(self.board[row - i][col + i] == player for i in range(4)):
                    return True

        # Check diagonal (top-left to bottom-right)
        for row in range(3):
            for col in range(4):
                if all(self.board[row + i][col + i] == player for i in range(4)):
                    return True

        return False

    def scoring(self) -> int:
        """
        Scoring function for the AI to evaluate board positions.

        Returns:
            int: 100 if current player wins, -100 if current player loses, 0 otherwise.
        """
        if self.find_four(self.current_player):
            return 100
        elif self.find_four(self.opponent_index):
            return -100
        else:
            return 0


def main() -> None:
    """
    Main function to run the Connect Four game.

    Sets up a game between a human player and an AI player using the Negamax
    algorithm with a search depth of 5.
    """
    print("Welcome to Connect Four!")
    print("You are Player 1, AI is Player 2")
    print("Enter column numbers (0-6) to make your move\n")

    ai_algo = Negamax(5)

    game = ConnectFour([Human_Player(), AI_Player(ai_algo)])

    game.play()

    # Display final result
    print("\n" + "=" * 40)
    if game.find_four(1):
        print("Player 1 (Human) wins!")
    elif game.find_four(2):
        print("Player 2 (AI) wins!")
    else:
        print("It's a draw!")
    print("=" * 40)


if __name__ == "__main__":
    main()
