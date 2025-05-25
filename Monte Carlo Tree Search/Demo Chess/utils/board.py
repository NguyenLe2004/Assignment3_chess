import chess
import numpy as np
import torch

def board_to_tensor(board, history=[]):
    tensor = np.zeros((8, 8, 112), dtype=np.float16)

    piece_channels = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    def encode_board(b, tensor, start_ch):
        for square in chess.SQUARES:
            piece = b.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                row, col = 7 - rank, file 
                offset = 0 if piece.color == chess.WHITE else 6
                tensor[row, col, start_ch + piece_channels[piece.piece_type] + offset] = 1
                
    encode_board(board, tensor, 0)

    tensor[:, :, 12] = 1 if board.turn == chess.WHITE else 0

    castle_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]

    for i, right in enumerate(castle_rights):
        tensor[:, :, 13+i] = 1 if right else 0

    if board.ep_square:
        ep_file = chess.square_file(board.ep_square)
        tensor[:, :, 17 + ep_file] = 1

    tensor[:, :, 25] = board.halfmove_clock / 50
    tensor[:, :, 26] = board.fullmove_number / 100
    tensor[:, :, 27] = 1 if board.is_check() else 0

    # --- 7 nước đi trước (mỗi bước 12 channel) ---
    for i, past_board in enumerate(history[-7:]):
        encode_board(past_board, tensor, 28 + i * 12)
    return tensor

def rotate_board_tensor(board_tensor):
    return np.fliplr(np.flipud(board_tensor)).copy()

def tensor_to_board(tensor):
    tensor = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    board = chess.Board(fen=None)  
    board.clear()

    piece_channels = {
        0: chess.PAWN,
        1: chess.KNIGHT,
        2: chess.BISHOP,
        3: chess.ROOK,
        4: chess.QUEEN,
        5: chess.KING
    }

    # Decode 8x8x12 (channel 0–11)
    for row in range(8):
        for col in range(8):
            for ch in range(12):
                if tensor[row, col, ch] > 0.5:
                    color = chess.WHITE if ch < 6 else chess.BLACK
                    piece_type = piece_channels[ch % 6]
                    rank = 7 - row
                    file = col
                    square = chess.square(file, rank)
                    board.set_piece_at(square, chess.Piece(piece_type, color))

    # Channel 12: lượt đi
    board.turn = bool(round(tensor[0, 0, 12]))

    if tensor[0, 0, 13] > 0.5:
        board.castling_rights |= chess.BB_H1
    if tensor[0, 0, 14] > 0.5:
        board.castling_rights |= chess.BB_A1
    if tensor[0, 0, 15] > 0.5:
        board.castling_rights |= chess.BB_H8
    if tensor[0, 0, 16] > 0.5:
        board.castling_rights |= chess.BB_A8

    # Channels 17–24: en passant file
    ep_file = None
    for f in range(8):
        if tensor[0, 0, 17 + f] > 0.5:
            ep_file = f
            break
    if ep_file is not None:
        rank = 5 if board.turn == chess.WHITE else 2
        board.ep_square = chess.square(ep_file, rank)

    board.halfmove_clock = int(round(tensor[0, 0, 25] * 50))
    board.fullmove_number = int(round(tensor[0, 0, 26] * 100))

    return board


def decode_move_index(move_index: int, board: chess.Board) -> str:
    NUM_SQUARES = 64
    NUM_MOVE_TYPES = 73
    from_square = move_index // NUM_MOVE_TYPES
    move_type = move_index % NUM_MOVE_TYPES
    from_file = from_square % 8
    from_rank = from_square // 8

    if move_type < 56:
        direction = move_type // 7
        steps = (move_type % 7) + 1
        dxdy_map = {
            0: (0, -1),
            1: (1, -1),
            2: (1, 0), 
            3: (1, 1),
            4: (0, 1), 
            5: (-1, 1),
            6: (-1, 0), 
            7: (-1, -1),
        }
        dx, dy = dxdy_map[direction]
        to_file = from_file + dx * steps
        to_rank = from_rank + dy * steps
    elif move_type < 64:
        knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        dx, dy = knight_moves[move_type - 56]
        to_file = from_file + dx
        to_rank = from_rank + dy
    else:
        promo_index = move_type - 64
        direction = (promo_index // 3) - 1  
        promo_piece = promo_index % 3
        promotion_map = {0: chess.KNIGHT, 1: chess.BISHOP, 2: chess.ROOK,}
        promotion = promotion_map[promo_piece]
        to_file = from_file + direction
        to_rank = from_rank + (1 if board.turn == chess.WHITE else -1)

    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        print(to_file, to_rank)
        raise ValueError("Giải mã ra nước đi không hợp lệ")

    to_square = to_rank * 8 + to_file
    move = chess.Move(from_square, to_square)

    if move_type >= 64:
        move.promotion = promotion

    return move.uci()


