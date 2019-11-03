""" Views """
from models import OthelloBoard, Stone

def console(board:OthelloBoard):
    """
    board : OthelloBoard(object)
    """
    column = 'y\\x'
    for i in range(board.size):
        column += ' ' + str(i) + ' '
    print(column)

    for i in range(board.size):
        row = ' ' + str(i) + ' '
        stones = ''
        for j in range(board.size):
            e = board.board[i][j]
            if type(e) == Stone:
                stones += ' {} '.format(board.board[i][j].color.value)
            else:
                stones += '   '
        print(row + stones)