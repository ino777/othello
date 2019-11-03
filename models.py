""" Models of othello game """
import abc
from enum import Enum
import logging
import random
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Color(Enum):
    """ Color enumeration """
    BLACK = '●'
    WHITE = '○'


def opp_color(color):
    if color == Color.BLACK:
        return Color.WHITE
    elif color == Color.WHITE:
        return Color.BLACK
    else:
        raise ValueError


class Stone(object):
    """Stone model of othello game
    
    Attributes
    -------
    color : Color
        stone color
    """
    def __init__(self, color, *args, **kwargs):
        if not color in list(Color):
            raise AttributeError('Stone attribute \'color\' must be a member of \'Color\'(Enum)')
        self.color = color

    def reverse(self):
        self.color = opp_color(self.color)


class Board(object):
    """
    Generic board model

    Attributes
    -------
    board : np.array of Color(Enum) or None
    size : int
        board size (the number of square is size**2)
    """
    def __init__(self, size, *args, **kwargs):
        """
        Parameters
        -------
        size : int
            board size (the number of square is size**2)
        """
        self.size = size
        self.board = np.zeros((self.size, self.size), dtype='object')

    def put(self, stone:Stone, pos):
        x, y = pos
        self.board[y][x] = stone
    
    def is_blank(self, pos):
        x, y = pos
        return self.board[y][x] == 0


class OthelloBoard(Board):
    """
    Board model of othello game

    """
    def __init__(self, board=None, size=8, *args, **kwargs):
        super().__init__(size, *args, **kwargs)
        if not board is None and type(board) == np.ndarray:
            self.board = board

    def count_stones(self, color):
        """ Return the number of color stones """
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                e = self.board[i][j]
                if e.color == color:
                    count += 1
        return count

    def aquaire_reversibles(self, pos, color):
        """ Return a list of points which can be reversed
        
        Parameters
        ------
        pos : tuple
            position put with Stone
        color : Color
            the turn color

        Returns
        ------
        reversible_stones : list of list of Color(Enum)
            [[upper left], [upper mid], ... , [lower right]]
            if there are no elements in the direction, return empty list
        """
        direction_XY = [
            (-1, -1), (+0, -1), (+1, -1),
            (-1, +0),           (+1, +0),
            (-1, +1), (+0, +1), (+1, +1)
        ]
        
        reversible_stones = [[] for _ in range(len(direction_XY))]
        for i, direction in enumerate(direction_XY):
            dx, dy = direction
            x,y = pos
            for _ in range(self.size):
                x += dx
                y += dy
                if not (0 <= x < self.size and 0 <= y < self.size):
                    reversible_stones[i] = []
                    break
                if not type(self.board[y][x]) == Stone:
                    reversible_stones[i]= []
                    break
                if self.board[y][x].color != color:
                    reversible_stones[i].append((x, y))
                elif self.board[y][x].color == color:
                    break
        return reversible_stones

    def is_available(self, pos, color):
        """
        Whether you can put Stone on this position
        
        Parameters
        -------
        pos : tuple
            position put with Stone
        color : Color
            the turn color
        """
        if not self.is_blank(pos):
            return False
        reversible_stones = self.aquaire_reversibles(pos, color)
        return any(reversible_stones[i] for i in range(8))
    
    def availables(self, color):
        """ Return a list of points where you can put Stone """
        available_pos = []
        for y in range(self.size):
            for x in range(self.size):
                pos = (x, y)
                if self.is_available(pos, color):
                    available_pos.append((x, y))
        return available_pos

    def update_board(self, pos, color):
        """ Update board
        """
        self.put(Stone(color), pos)
        reversible_stones = self.aquaire_reversibles(pos, color)
        for i in range(len(reversible_stones)):
            for pos in reversible_stones[i]:
                x, y = pos
                self.board[y][x].reverse()



class Character(metaclass=abc.ABCMeta):
    """
    Generic Character model

    Attributes
    -------
    name : str
        character name
    color : Color
    turn_flag : bool (default False)
    """
    def __init__(self, name, color, *args, **kwargs):
        self.name = name
        self.color = color
        self.turn_flag = False

    @abc.abstractmethod
    def input(self):
        pass


class User(Character):
    """ User model """
    def __init__(self, name, color, *args, **kwargs):
        """
        name : str
        color : Color
        """
        super().__init__(name, color, *args, **kwargs)

    def input(self, *args, **kwargs):
        """ Return input position
        """
        while True:
            pos = map(int, input('(x y)=>').split())
            pos = tuple(pos)
            if len(pos) == 2:
                break
        return pos


class Computer(Character):
    """ COM model 
    
    Attributes
    ------
    depth : (optional) alpha beta depth of AI (default 1)
    """
    def __init__(self, name, color, depth=1, *args, **kwargs):
        """
        name : str
        color : Color
        depth : int
        """
        super().__init__(name, color, *args, **kwargs)
        self.depth = depth
        self.ai = AI(self.color, self.depth)

    def input(self, board, candidates, *args, **kwargs):
        """ Return input position

        Parameters
        -------
        board : ndarray
        candidates : list of tuple
            positions available
        """
        if not type(candidates) == list:
            raise TypeError
        if self.ai:
            pos = self.think(board.copy())
        else:
            pos = random.choice(candidates)
        return pos

    def think(self, board):
        """
        Return the output of thinking

        Returns
        -------
        pos : tuple of int
        """
        self.ai.alpha_beta(board, self.depth, self.color, self.turn_flag)
        pos = self.ai.opt_action
        return pos


class EvalOthelloBoard(OthelloBoard):
    """
    Othello Board model to calculate evaluation value

    Parameters
    -------
    board : ndarray (only size (8, 8))
    """
    evaluate_value = np.array([
        [45.0, -11.0, 4.0, -1.0, -1.0, 4.0, -11.0, 45.0],
        [-11.0, -16.0, -1.0, -3.0, -3.0, -1.0, -16.0, -11.0],
        [4.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, 4.0],
        [-1.0, -3.0, -1.0, 0.0, 0.0, -1.0, -3.0, -1.0],
        [-1.0, -3.0, -1.0, 0.0, 0.0, -1.0, -3.0, -1.0],
        [4.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, 4.0],
        [-11.0, -16.0, -1.0, -3.0, -3.0, -1.0, -16.0, -11.0],
        [45.0, -11.0, 4.0, -1.0, -1.0, 4.0, -11.0, 45.0]
    ])
    w1 = 2.0 + np.random.randn()
    w2 = 5.0 + np.random.randn()
    w3 = 1.0 + np.random.randn()

    def __init__(self, board):
        """
        board : ndarray
        """
        super().__init__(board)
        self.board = board
        if not self.board.shape == (8, 8):
            raise ValueError('EvalOthelloBoard can be used with size 8 othello')

        self.dummy()

    def dummy(self):
        """
        Make dummy board. Stones are replaced by new Stone.
        """
        for i in range(self.size):
            for j in range(self.size):
                e = self.board[i][j]
                if type(e) == Stone:
                    self.board[i][j] = Stone(e.color)

    def is_confirmed(self, pos, color):
        """ Whether the stone at the position is imreversible """
        if pos in {(0, 0), (self.size-1, 0),
                        (0, self.size-1), (self.size-1, self.size-1)}:
            return True
        
        x, y = pos
        if x == 0 or x == self.size-1:
            if all(self.board[j][x].color == color for j in range(y)):
                return True
            if all(self.board[j][x].color == color for j in range(y, self.size)):
                return True
        elif y == 0 or y == self.size-1:
            if all(self.board[y][i].color == color for i in range(x)):
                return True
            if all(self.board[y][i].color == color for i in range(x, self.size)):
                return True
        else:
            raise ValueError
        return False

    def confirms(self, color):
        """ Return the number of imreversible stone """
        if not any(self.board[j][i] == color for i in [0, -1] for j in [0, -1]):
            return 0
        
        sum_confirm_stones = 0
        for row_col in [(0, 0), (self.size-1, self.size-1)]:
            row, col = row_col
            for x, stone in enumerate(self.board[row]):
                if not stone.color == color:
                    continue
                if self.is_confirmed((x, row), color):
                    sum_confirm_stones += 1
            for y, stone in enumerate(self.board[:,col]):
                if not stone.color == color:
                    continue
                if self.is_confirmed((col, y), color):
                    sum_confirm_stones += 1
        return sum_confirm_stones

    def is_semi_confirmed(self, pos, color):
        pass
    def semi_confirms(self, color):
        pass

    def to_float(self, color):
        """
        Convert a board into a ndarray of float.
        Self stone => +1.0, opponent stone => -1.0 and blank square => 0.0.
        """
        array = self.board.copy()
        for i in range(len(array)):
            for j in range(len(array[i])):
                e = array[i][j]
                if not type(e) == Stone:
                    array[i][j] = 0.0
                else:
                    if e.color == color:
                        array[i][j] =  1.0
                    elif e.color == opp_color(color):
                        array[i][j] = -1.0
                    else:
                        array[i][j] = 0.0
        array = array.astype(np.float32)
        return array


    def eval(self, color):
        """
        Return evaluation value

        color : Color
           the turn color
        """

        fs = (self.confirms(color)-self.confirms(opp_color(color)) +np.random.rand()*3) * 11
        cn = (len(self.availables(color)) + np.random.rand()*2) *10
        f_board = self.to_float(color)
        bp = np.sum(self.evaluate_value * f_board * np.random.rand(self.size, self.size)) * 3
        logger.debug({
            'fs': fs,
            'cn': cn,
            'bp': bp,
            'eval' : self.w1*bp + self.w2*fs + self.w3*cn,
        })
        return self.w1*bp + self.w2*fs + self.w3*cn


class AI(object):
    """
    Othello AI using alpha beta search

    Attributes
    -------
    opt_action : tuple
        optimal position
    depth : int
        search depth
    eval_table : dict
        key : hash of ndarray
        value : score (float)
    """
    def __init__(self, color, depth=1):
        self.depth = depth
        self.opt_action = (0,)
        self.color = color
        self.transposition_table = set()
        self.eval_table = {}

    def alpha_beta(self, board, depth, color, turn_flag, alpha=-np.inf, beta=np.inf):
        """ Alpha beta search
        
        Parameters
        -------
        board : ndarray
        depth : int
        color : Color
        turn_flag : bool
        """
        new_board = EvalOthelloBoard(board.copy())

        """ Return eval value if reaching a leaf """
        if depth <= 0:
            return new_board.eval(self.color)
        
        the_other_color = opp_color(color)
        new_candidates = new_board.availables(color)
        logger.debug({
            'action': 'candidates',
            'color': color,
            'candidates': new_candidates,
        })

        if not new_candidates:
            the_other_candidates = new_board.availables(the_other_color)
            if not the_other_candidates:
                """ When the game end """
                return new_board.eval(self.color)
            else:
                """ When the opponent can put stone """
                turn_flag = not turn_flag
                color = the_other_color
                depth -= 1
                new_candidates = the_other_candidates
        
        """
        Expand board and store the all child boards in children.
        Associate the child and the action of that time.
        children : list of ndarray
        """
        children = []
        action_child_dict= {}
        for action in new_candidates:
            child = EvalOthelloBoard(board.copy())
            child.update_board(action, color)
            children.append(child.board)
            action_child_dict[action] = child.board.copy()
            del child

        del new_candidates
        del new_board
        
        """
        Sort children in eval value order to speed up alpha beta search
        In descending order when self turn and in ascending order when opponent turn
        Note : Skip this sort when depth is less than 2
        """
        if depth >= self.depth - 2:
            children = self.sort_by_eval(children, color)
            if turn_flag:
                children.reverse()
            # for _ in range(length_actions // 3):
            #     children.pop()

        """ Alpha beta searching """
        if turn_flag:
            maximum = -np.inf
            max_action = (0,)
            for child in children:
                """
                Check if the child is the same with any boards that came up before.
                If they match, return the value to 'score'.
                If not, start to alpha beta search in deeper layer
                """
                h_child = hash(str(child))
                if h_child in self.transposition_table:
                    score = self.eval_table[h_child]
                else:
                    score = self.alpha_beta(
                        child.copy(), depth-1, the_other_color, not turn_flag, alpha, beta)
                    self.transposition_table.add(h_child)
                    self.eval_table[h_child] = score
                
                if score >= beta:
                    break
                if score > maximum:
                    maximum = score
                    max_action = [
                        action for action, value in action_child_dict.items() if np.all(value == child)][0]
                    alpha = max(alpha, maximum)

            self.opt_action = max_action
            return maximum
        else:
            minimum = np.inf
            for child in children:

                h_child = hash(str(child))
                if h_child in self.transposition_table:
                    score = self.eval_table[h_child]
                else:
                    score = self.alpha_beta(
                        child.copy(), depth-1, the_other_color, not turn_flag, alpha, beta)
                    self.transposition_table.add(h_child)
                    self.eval_table[h_child] = score

                if score <= alpha:
                    break
                if score < minimum:
                    minimum = score
                    beta = min(beta, minimum)

            return minimum

    def quick_sort(self, boards, table):
        """
        Quick sort in ascending order
        """
        left = []
        right = []
        if len(boards) <= 1:
            return boards
        
        ref = boards[0]
        ref_eval = table[hash(str(ref))]
        ref_count = 0

        for board in boards:
            board_eval = table[hash(str(board))]
            if board_eval < ref_eval:
                left.append(board)
            elif board_eval > ref_eval:
                right.append(board)
            else:
                ref_count += 1

        left = self.quick_sort(left, table)
        right = self.quick_sort(right, table)
        return left + [ref] * ref_count + right

    def sort_by_eval(self, boards, color):
        """
        Return list of board in ascending order by eval value
        """
        evals = {}
        corner = []
        x_position = [(0, 0), (-1, 0), (0, -1), (-1, -1)]
        
        for board in boards:
            """ Give priority to boards that has self stone at corner """
            for position in x_position:
                x, y = position
                if board[y][x] == color:
                    corner.append(board)

            new_board = EvalOthelloBoard(board.copy())
            evals[hash(str(board))] = new_board.eval(self.color)

        sorted_by_eval = self.quick_sort(boards, table=evals)
        return corner + sorted_by_eval