import torch

turn = 1 # Quy định: 1 là lượt trắng, -1 là lượt đen
winner = 0
king_safe_point = 0 # Điểm an toàn vua
kingmoved = 0
blakingmoved = 0
castle_mov_w1 = ((7, 4),(7, 2))
castle_mov_w2 = ((7, 4),(7, 6))
castle_mov_b1 = ((0, 4),(0, 2))
castle_mov_b2 = ((0, 4),(0, 6))

# 1: tốt trắng, 2: xe trắng, 3: mã trắng, 4: tượng trắng, 5: hậu trắng, 6: vua trắngt, màu đen thì thêm dấu - phía trước 
position = torch.tensor([
        [-2, -3, -4, -5, -6, -4, -3, -2],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [2, 3, 4, 5, 6, 4, 3, 2]
        ])

# Hàm di chuyển cho tốt, tìm tất cả tốt, kiểm tra ô phía trên có trống hay không, nếu trống thì thêm vào danh sách di chuyển (moves). 
def pawnMoves(position, turn):
    moves = []
    if turn == 1:
        pawns = torch.nonzero((position.abs() == 1) & (position > 0), as_tuple=False)
        for pawn in pawns:
            x, y = pawn[0].item(), pawn[1].item()
            above = position[x - 1, y]
            if above == 0:
                moves.append(((x, y),(x - 1, y)))
                if x == 6 and position[x - 2, y] == 0:
                    moves.append(((x, y), (x - 2, y)))
        return moves
    
    else:
        pawns = torch.nonzero((position.abs() == 1) & (position < 0), as_tuple=False)
        for pawn in pawns:
            x, y = pawn[0].item(), pawn[1].item()
            below = position[x + 1, y]
            if below == 0:
                moves.append(((x, y), (x + 1, y)))
                if x == 1 and position[x + 2, y] == 0:
                    moves.append(((x, y), (x + 2, y)))
        return moves

# Hàm tấn công cho tốt, tìm tất cả tốt, tìm 2 ô chéo phía trước mặt nó, nếu ô đó có quân địch thì thêm vào danh sách di chuyển (moves).
def pawnAttack(position,turn):
    moves = []
    if turn == 1:
        pawns = torch.nonzero((position.abs() == 1) & (position > 0), as_tuple=False)
        for pawn in pawns:
            x, y = pawn[0].item(), pawn[1].item()
            if y + 1 < 8:
                right_up = position[x - 1, y + 1]
                if right_up < 0:
                    moves.append(((x, y), (x - 1, y + 1)))
            if y - 1 > -1 :
                left_up = position[x - 1, y - 1]
                if left_up < 0:
                    moves.append(((x, y),(x - 1, y - 1)))
        return moves
    else:
        pawns = torch.nonzero((position.abs() == 1) & (position < 0), as_tuple=False)
        for pawn in pawns:
            x, y = pawn[0].item(), pawn[1].item()
            if y + 1 < 8:
                right_up = position[x + 1, y + 1]
                if right_up > 0:
                    moves.append(((x, y),(x + 1, y + 1)))
            if y - 1 > -1 :
                left_up = position[x + 1, y - 1]
                if left_up > 0:
                    moves.append((((x, y),(x + 1, y - 1))))
        return moves  

# Luật bắt tốt qua đường (en passant). Tìm tất cả tốt, kiểm tra xem tốt đã di chuyển chưa,
# nếu chưa thì kiểm tra điều kiện bắt tốt qua đường (en passant), nếu đúng thì thêm vào danh sách di chuyển (moves).
# Chú ý bắt tốt qua đường là nước đi không ưu tiên, nếu thực hiện được thì phải thực hiện ngay, không để qua lượt sau.
def pawnEnPassant(position, turn, last_move):
    moves = []
    if turn == 1:
        pawns = torch.nonzero((position.abs() == 1) & (position > 0), as_tuple=False)
        for pawn in pawns:
            x, y = pawn[0].item(), pawn[1].item()
            if y + 1 < 8:
                if position[last_move[1]] == -1 and last_move[0][0] == 6 and last_move[1][0] == 4 and last_move[1][1] == y + 1:
                    moves.append(((x, y), (x - 1, y + 1)))
            if y - 1 > -1 :
                if position[last_move[1]] == -1 and last_move[0][0] == 6 and last_move[1][0] == 4 and last_move[1][1] == y - 1:
                    moves.append(((x, y),(x - 1, y - 1)))
        return moves
    else:
        pawns = torch.nonzero((position.abs() == 1) & (position < 0), as_tuple=False)
        for pawn in pawns:
            x, y = pawn[0].item(), pawn[1].item()
            if y + 1 < 8:
                if position[last_move[1]] == 1 and last_move[0][0] == 1 and last_move[1][0] == 3 and last_move[1][1] == y + 1:
                    moves.append(((x, y),(x + 1, y + 1)))
            if y - 1 > -1 :
                if position[last_move[1]] == 1 and last_move[0][0] == 1 and last_move[1][0] == 3 and last_move[1][1] == y - 1:
                    moves.append((((x, y),(x + 1, y - 1))))
        return moves

# Hàm di chuyển và tấn công cho tượng: tìm quân tượng, check tất cả các đường chéo, nếu ô đó trống hoặc có quân địch thì thêm vào danh sách di chuyển (moves).
def bishopMoves(position, turn):
    moves = []
    if turn == 1:
        bishops = torch.nonzero((position.abs() == 4) & (position > 0), as_tuple=False)
        for bishop in bishops:
            x, y = bishop[0].item(), bishop[1].item()
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for direct_x, direct_y in directions:
                new_x, new_y = x + direct_x, y + direct_y
                while 0 <= new_x < 8 and 0 <= new_y < 8:
                    if position[new_x, new_y] == 0:
                        moves.append(((x, y), (new_x, new_y)))
                    elif position[new_x, new_y] < 0:
                        moves.append(((x, y), (new_x, new_y)))
                        break
                    else: break
                    new_x += direct_x
                    new_y += direct_y
    else:
        bishops = torch.nonzero((position.abs() == 4) & (position < 0), as_tuple=False)
        for bishop in bishops:
            x, y = bishop[0].item(), bishop[1].item()
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for direct_x, direct_y in directions:
                new_x, new_y = x + direct_x, y + direct_y
                while 0 <= new_x < 8 and 0 <= new_y < 8:
                    if position[new_x, new_y] == 0:
                        moves.append(((x, y), (new_x, new_y)))
                    elif position[new_x, new_y] > 0:
                        moves.append(((x, y), (new_x, new_y)))
                        break
                    else: break
                    new_x += direct_x
                    new_y += direct_y
    return moves

# Hàm di chuyển và tấn công cho mã: tìm quân mã, check tất cả các hướng đi của mã, nếu ô đó trống hoặc có quân địch thì thêm vào danh sách di chuyển (moves).
def knightMoves(position, turn):
    moves = []
    if turn == 1:
        knights = torch.nonzero((position.abs() == 3) & (position > 0), as_tuple=False)
        for knight in knights:
            x, y = knight[0].item(), knight[1].item()
            directions = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
            for direct_x, direct_y in directions:
                new_x, new_y = x + direct_x, y + direct_y
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    if position[new_x, new_y] <= 0:
                        moves.append(((x, y), (new_x, new_y)))
    else:
        knights = torch.nonzero((position.abs() == 3) & (position < 0), as_tuple=False)
        for knight in knights:
            x, y = knight[0].item(), knight[1].item()
            directions = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
            for direct_x, direct_y in directions:
                new_x, new_y = x + direct_x, y + direct_y
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    if position[new_x, new_y] >= 0:
                        moves.append(((x, y), (new_x, new_y)))
    return moves

# Hàm tấn công và di chuyển cho xe, tìm quân xe, check tất cả các đường thẳng 
def rookMoves(position, turn):
    moves = []
    if turn == 1:
        rooks = torch.nonzero((position.abs() == 2) & (position > 0), as_tuple=False)
        for rook in rooks:
            x, y = rook[0].item(), rook[1].item()
            for direct_x in [-1, 1]:
                new_x = x + direct_x
                while 0 <= new_x < 8:
                    if position[new_x, y] == 0:
                        moves.append(((x, y), (new_x, y)))
                    elif position[new_x, y] < 0:
                        moves.append(((x, y), (new_x, y)))
                        break
                    else:
                        break
                    new_x += direct_x
            for direct_y in [-1, 1]:
                new_y = y + direct_y
                while 0 <= new_y < 8:
                    if position[x, new_y] == 0:
                        moves.append(((x, y), (x, new_y)))
                    elif position[x, new_y] < 0:
                        moves.append(((x, y), (x, new_y)))
                        break
                    else:
                        break
                    new_y += direct_y
    else:
        rooks = torch.nonzero((position.abs() == 2) & (position < 0), as_tuple=False)
        for rook in rooks:
            x, y = rook[0].item(), rook[1].item()
            for direct_x in [-1, 1]:
                new_x = x + direct_x
                while 0 <= new_x < 8:
                    if position[new_x, y] == 0:
                        moves.append(((x, y), (new_x, y)))
                    elif position[new_x, y] > 0:
                        moves.append(((x, y), (new_x, y)))
                        break
                    else:
                        break
                    new_x += direct_x
            for direct_y in [-1, 1]:
                new_y = y + direct_y
                while 0 <= new_y < 8:
                    if position[x, new_y] == 0:
                        moves.append(((x, y), (x, new_y)))
                    elif position[x, new_y] > 0:
                        moves.append(((x, y), (x, new_y)))
                        break
                    else:
                        break
                    new_y += direct_y
    return moves

# Hàm tấn công và di chuyển của hậu, check như tượng + xe
def queenMoves(position, turn):
    moves = []
    if turn == 1:
        queens = torch.nonzero((position.abs() == 5) & (position > 0), as_tuple=False)
        for queen in queens:
            x, y = queen[0].item(), queen[1].item()
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for direct_x, direct_y in directions:
                new_x, new_y = x + direct_x, y + direct_y
                while 0 <= new_x < 8 and 0 <= new_y < 8:
                    if position[new_x, new_y] == 0:
                        moves.append(((x, y), (new_x, new_y)))
                    elif position[new_x, new_y] < 0:
                        moves.append(((x, y), (new_x, new_y)))
                        break
                    else:
                        break
                    new_x += direct_x
                    new_y += direct_y
            for direct_x in [-1, 1]:
                new_x = x + direct_x
                while 0 <= new_x < 8:
                    if position[new_x, y] == 0:
                        moves.append(((x, y), (new_x, y)))
                    elif position[new_x, y] < 0:
                        moves.append(((x, y), (new_x, y)))
                        break
                    else: break
                    new_x += direct_x
            for direct_y in [-1, 1]:
                new_y = y + direct_y
                while 0 <= new_y < 8:
                    if position[x, new_y] == 0:
                        moves.append(((x, y), (x, new_y)))
                    elif position[x, new_y] < 0:
                        moves.append(((x, y), (x, new_y)))
                        break
                    else: break
                    new_y += direct_y
    else:
        queens = torch.nonzero((position.abs() == 5) & (position < 0), as_tuple=False)
        for queen in queens:
            x, y = queen[0].item(), queen[1].item()
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for direct_x, direct_y in directions:
                new_x, new_y = x + direct_x, y + direct_y
                while 0 <= new_x < 8 and 0 <= new_y < 8:
                    if position[new_x, new_y] == 0:
                        moves.append(((x, y), (new_x, new_y)))
                    elif position[new_x, new_y] > 0:
                        moves.append(((x, y), (new_x, new_y)))
                        break
                    else: break
                    new_x += direct_x
                    new_y += direct_y
            for direct_x in [-1, 1]:
                new_x = x + direct_x
                while 0 <= new_x < 8:
                    if position[new_x, y] == 0:
                        moves.append(((x, y), (new_x, y)))
                    elif position[new_x, y] > 0:
                        moves.append(((x, y), (new_x, y)))
                        break
                    else: break
                    new_x += direct_x
            for direct_y in [-1, 1]:
                new_y = y + direct_y
                while 0 <= new_y < 8:
                    if position[x, new_y] == 0:
                        moves.append(((x, y), (x, new_y)))
                    elif position[x, new_y] > 0:
                        moves.append(((x, y), (x, new_y)))
                        break
                    else: break
                    new_y += direct_y
    return moves

# Hàm di chuyển cho vua, tìm quân vua, check tất cả 8 ô xung quanh
def kingMoves(position, turn, gamehis, last_move):
    moves = []
    if turn == 1:
        kings = torch.nonzero((position.abs() == 6) & (position > 0), as_tuple=False)
        king = kings[0]
        x, y = king[0].item(), king[1].item()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for direct_x, direct_y in directions:
            new_x, new_y = x + direct_x, y + direct_y
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                if position[new_x, new_y] <= 0:
                    moves.append(((x, y), (new_x, new_y)))
    else:
        kings = torch.nonzero((position.abs() == 6) & (position < 0), as_tuple=False)
        king = kings[0]
        x, y = king[0].item(), king[1].item()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for direct_x, direct_y in directions:
            new_x, new_y = x + direct_x, y + direct_y
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                if position[new_x, new_y] >= 0:
                    moves.append(((x, y), (new_x, new_y)))
    castl = is_castlable(gamehis, turn, position, last_move)
    moves.extend(castl)
    return moves

# Kiểm tra nếu vua có thể nhập thành (castle)
def is_castlable(gamehis, turn, pos, last_move):
    castle_mov = []
    if turn == 1:
        can_castle = True
        for move in gamehis:
            if can_castle:
                if move[0] != (7, 4):
                    if move[0] != (7, 7): pass
                    else: can_castle = False
                else:
                    can_castle = False
        if can_castle and position[7][5].item() == 0 and position[7][6] == 0 and position[7][4] == 6 and position[7][7] == 2:
            castle_mov.append(castle_mov_w2)
        can_castle = True
        for move in gamehis:
            if can_castle:
                if move[0] != (7, 4):
                    if move[0] != (7, 0): pass
                    else: can_castle = False
                else: can_castle = False
        if can_castle and position[7][1].item() == 0 and position[7][2] == 0 and position[7][3] == 0 and position[7][4] == 6 and position[7][0] == 2:
            castle_mov.append(castle_mov_w1)
    else:
        can_castle = True
        for move in gamehis:
            if can_castle:
                if move[0] != (7, 4):
                    if move[0] != (7, 7): pass
                    else: can_castle = False
                else: can_castle = False
        if can_castle and position[0][5].item() == 0 and position[0][6] == 0 and position[0][4] == -6 and position[0][7] == -2:
            castle_mov.append(castle_mov_b2)
        can_castle = True
        for move in gamehis:
            if can_castle:
                if move[0] != (0, 4):
                    if move[0] != (0, 0): pass
                    else: can_castle = False
                else: can_castle = False
        if can_castle and position[0][1].item() == 0 and position[0][2] == 0 and position[0][3] == 0 and position[0][4] == -6 and position[0][0] == -2:
            castle_mov.append(castle_mov_b1)
    return castle_mov

# Điểm của mỗi quân cờ, lấy số liệu từ bản stockfish ~15
def materialEqual(position):
    material_score = 0
    pawns = torch.nonzero(position == 1, as_tuple=False)
    for pawn in pawns:
        material_score += 124
    bishops = torch.nonzero(position == 4, as_tuple=False)
    for bishop in bishops:
        material_score += 825
    knights = torch.nonzero(position == 3, as_tuple=False)
    for knight in knights:
        material_score += 781
    rooks = torch.nonzero(position == 2, as_tuple=False)
    for rook in rooks:
        material_score += 1276
    queens = torch.nonzero(position == 5, as_tuple=False)
    for queen in queens:
        material_score += 2538
    pawns = torch.nonzero(position == -1, as_tuple=False)
    for pawn in pawns:
        material_score -= 124
    bishops = torch.nonzero(position == -4, as_tuple=False)
    for bishop in bishops:
        material_score -= 825
    knights = torch.nonzero(position == -3, as_tuple=False)
    for knight in knights:
        material_score -= 781
    rooks = torch.nonzero(position == -2, as_tuple=False)
    for rook in rooks:
        material_score -= 1276
    queens = torch.nonzero(position == -5, as_tuple=False)
    for queen in queens:
        material_score -= 2538
    return material_score

# Giá trị chênh lệch về vật chất giữa 2 nước đi
def materialDifference(poshis):
    oldpos = poshis[len(poshis) - 2]
    newpos = poshis[len(poshis) - 1]
    oldmat = materialEqual(oldpos)
    newmat = materialEqual(newpos)
    return (newmat - oldmat) * -1

# Lấy thông tin tất cả các nước đi hợp lệ
def getAllLegalMoves(position, turn, last_move, gamehis):
    legalmoves = []
    legalmoves.append(pawnMoves(position, turn))
    legalmoves.append(pawnAttack(position,turn))
    legalmoves.append(pawnEnPassant(position, turn, last_move))
    legalmoves.append(bishopMoves(position, turn))
    legalmoves.append(knightMoves(position, turn))
    legalmoves.append(rookMoves(position, turn))
    legalmoves.append(queenMoves(position, turn))
    legalmoves.append(kingMoves(position, turn, gamehis, last_move))
    legalmoves = [move for move in legalmoves if move]
    return legalmoves

def makeMoves(position, legalmoves):
    positions = []
    for type in legalmoves:
        for move in type:
            positions.append(makeMove(position, move).tolist())
        return torch.tensor(positions)

def makeMove(position, move):
    start, finish = move[0], move[1]
    oldposition = position.clone()
    piece = position[start[0], start[1]] 
    oldposition[start[0], start[1]] = 0
    oldposition[finish[0], finish[1]] = piece
    position = oldposition
    return position

# Nhập thành
def castleMove(curposition, move):
    if move == ((7, 4), (7, 6)):
        position2 = curposition.clone()
        position2[7][4] = 0
        position2[7][6] = 6
        position2[7][7] = 0
        position2[7][5] = 2
        return position2
    elif move == ((7, 4), (7, 2)):
        position2 = curposition.clone()
        position2[7][4] = 0
        position2[7][2] = 6
        position2[7][0] = 0
        position2[7][3] = 2
        return position2
    elif move == ((0, 4), (0, 6)):
        position2 = curposition.clone()
        position2[0][4] = 0
        position2[0][6] = -6
        position2[0][7] = 0
        position2[0][5] = -2
        return position2
    elif move == ((0, 4), (0, 2)):
        position2 = curposition.clone()
        position2[0][4] = 0
        position2[0][2] = -6
        position2[0][0] = 0
        position2[0][3] = -2
        return position2

# Loại bỏ nước đi bị ghim (pinned)
def removePinned(position, turn, legalmoves, last_move, gamehis): 
    global king_safe_point
    king_safe_point = 0
    pinned = []
    presentlegal = legalmoves
    turn = turn * -1
    for thing in presentlegal:
        for one in thing:
            pos = makeMove(position, one)
            legalmoves = getAllLegalMoves(pos, turn, last_move, gamehis)
            for type in legalmoves:
                if type == [[]]:
                    continue
                for move in type:
                    finish = [move[1]]
                    king_cor = [(torch.nonzero(pos * turn == -6,as_tuple=False).tolist()[0][0], torch.nonzero(pos * turn == -6,as_tuple=False).tolist()[0][1])]
                    if finish == king_cor:
                        king_safe_point -= 1
                        pinned.append(one)
    
    if pinned == []:
        presentlegal = [move for thing in presentlegal for move in thing]
        return presentlegal
    else:
        presentlegal = [move for thing in presentlegal for move in thing]
        presentlegal = [move for move in presentlegal if move not in pinned]
        return presentlegal

# Kiểm tra điều kiện thắng thua, có một số điều kiện cần quan tâm:
# 1. Nếu không còn nước đi nào hợp lệ thì kiểm tra xem có bị chiếu không, nếu có thì thua, nếu không thì hòa
# 2. Kiểm tra có 1 thế cờ nào lặp lại 3 lần không, nếu có thì hòa
# 3. Kiểm tra tương quan lực lượng hiện tại, có một số trường hợp hoà như: vua vs vua, vua vs vua + mã, vua vs vua + tượng
def checkCondition(position, turn, legalmove, last_move, history, gamehis):
    global winner
    positioncount = 0
    pieces = []
    piece = torch.nonzero(position, as_tuple=False)
    for cor in piece:
        pieces.append(position[cor[0], cor[1]].item())
    pieces.sort()

    for hisposition in history:
        if torch.equal(hisposition, position):
            positioncount = positioncount + 1

    if len(legalmove) == 0:
        turn = turn * -1
        legalmoves = getAllLegalMoves(position, turn, last_move, gamehis)
        for move in legalmoves:
            for mov in move:
                finish = [mov[1]]
                king_cor = [(torch.nonzero(position * turn == -6,as_tuple=False).tolist()[0][0], torch.nonzero(position * turn == -6,as_tuple=False).tolist()[0][1])]
                if finish == king_cor:
                    winner = turn
                    print("Checkmate by white!" if winner == 1 else "Checkmate by black!")
                    return False
                else:
                    continue
        print("Stalemate!")
        winner = 0.5
        return False
    elif pieces == [-6,6] or pieces == [-6,-3,6] or pieces == [-6,-4,6] or pieces  == [-6,3,6] or pieces == [-6,4,6]:
        winner = 0.5
        print("Draw because cannot checkmate by other material!")
        return False
    elif positioncount >= 3:
        winner = 0.5
        print("Draw by repetition 3 times!")
        return False
    else: return True

# danh sách các nước đi hợp lệ (bỏ các nước đi bị ghim)
def getFinalLegalMoves(position, turn, last_move, gamehis):
    legalmoves = getAllLegalMoves(position, turn, last_move, gamehis)
    legalmoves = removePinned(position, turn, legalmoves, last_move, gamehis)
    return legalmoves

def getMaterialEqual(): return materialEqual(position)

def getNumberOfVaildMoves(legalmove): return len(legalmove)

# Đổi luợt
def changeturn():
    global turn
    turn = -turn

# Điểm an toàn vua
def getKingSafe(): return king_safe_point