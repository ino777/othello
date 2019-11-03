""" Main """
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from models import OthelloBoard, Stone, User, Computer,Color
from controllers import OthelloDirector


if __name__ == "__main__":
    board = OthelloBoard()
    player1 = User('Player1', Color.BLACK)
    player2 = Computer('Player2', Color.WHITE, depth=5)
    game = OthelloDirector(board, black=player1, white=player2)
    game.game()
