""" Controller """
import logging

from models import OthelloBoard, Stone, Color, User, Computer
from views import console


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def turn_color(turn):
    if turn % 2 == 1:
        return Color.BLACK
    elif turn % 2 == 0:
        return Color.WHITE


def turn_player(turn, black, white):
    if turn % 2 == 1:
        return black
    elif turn % 2 == 0:
        return white


class GameDirector(object):
    """ Generic game director class"""
    pass


class OthelloDirector(GameDirector):
    """
    Othello director

    Attributes
    -------
    board : OthelloBoard
    black : User or Computer
    white : User or Computer
    turn : int
    history : dict
    """
    def __init__(self, board:OthelloBoard, black, white, *args, **kwargs):
        if not (type(black) == User or type(black) == Computer
        or type(white) == User or type(white) == Computer):
            raise TypeError('black and white required to be User or Computer object')

        self.board = board
        self.black = black
        self.white = white
        self.turn = 0
        self.history = {}

        self.black.color = Color.BLACK
        self.white.color = Color.WHITE


    def init_board(self):
        size = self.board.size
        if size % 2 == 1:
            raise ValueError('Board.size must be multiple of 2')
        center1 = int(size / 2 - 1)
        center2 = int(size / 2)
        self.board.put(Stone(Color.BLACK), (center1, center1))
        self.board.put(Stone(Color.WHITE), (center2, center1))
        self.board.put(Stone(Color.WHITE), (center1, center2))
        self.board.put(Stone(Color.BLACK), (center2, center2))

    def save(self):
        self.history[self.turn] = self.board.board.copy()

    def game(self):
        self.init_board()

        while True:
            self.turn += 1
            self.save()
            console(self.board)

            color = turn_color(self.turn)
            player = turn_player(self.turn, black=self.black, white=self.white)
            player.turn_flag = True

            candidates = self.board.availables(color)

            if not candidates:
                if self.determine_game_end():
                    break
                else:
                    player.turn_flag = False
                    continue
            
            while True:
                print(candidates)
                pos = player.input(self.board.board.copy(), candidates, color)
                if pos in candidates:
                    logger.debug({
                        'action': 'game',
                        'pos': pos,
                        'status': 'success',
                    })
                    break
                logger.debug({
                    'action': 'game',
                    'pos': pos,
                    'status': 'fail',
                })
            self.board.update_board(pos, color)
            player.turn_flag = False
        self.end()


    def determine_game_end(self):
        return not (
            bool(self.board.availables(Color.BLACK)) or bool(self.board.availables(Color.WHITE)))

    def get_winner(self):
        black = self.board.count_stones(Color.BLACK)
        white = self.board.count_stones(Color.WHITE)
        if black > white:
            return Color.BLACK
        elif white > black:
            return Color.WHITE
        else:
            return
    
    def end(self):
        winner = self.get_winner()
        black = self.board.count_stones(Color.BLACK)
        white = self.board.count_stones(Color.WHITE)
        print('finish')
        if winner:
            print('{} win!'.format(winner.value))
            print(black, white)
        else:
            print('got even')
            print(black, white)


    # def undo(self):
    #     self.board = self.history[self.turn-2]

    # def redo(self):
    #     self.board = self.history[self.turn+2]
