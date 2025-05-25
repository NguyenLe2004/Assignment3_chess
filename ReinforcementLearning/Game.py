import torch
import Board
import random
import torch.nn
import time

all_moves = [((i, j), (k, l)) for i in range(8) for j in range(8) for k in range(8) for l in range(8)]
white_won = 0 # Số game trắng thắng
black_won = 0 # Số game đen thắng
draw = 0 # Số game hòa
castled = 0 # Số lần nhập thành

class BlackAICNN(torch.nn.Module):
    def __init__(self, n_outputs):
        super(BlackAICNN, self).__init__()
        self.memory = []
        self.gamma = 0.99 # Gamma gần 1 để giảm thiểu độ thiên lệch, các quyết định thiên về tương lai hơn là hiện tại
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, n_outputs),
            torch.nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
    def forward(self, x): return self.network(x)
    def remember(self, state, action, reward): self.memory.append((state, action, reward))
    def trainPolicy(self):
        states, actions, rewards = zip(*self.memory)
        states = torch.stack(states).float().unsqueeze(1)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        discounted_rewards = self.compute(rewards)
        self.optimizer.zero_grad()
        action_probs = []
        for state in states:
            prob = self.network(state.unsqueeze(0))
            action_probs.append(prob)
        loss = self.policyLoss(action_probs, actions, discounted_rewards)
        loss.backward()
        self.optimizer.step()
    def policyLoss(self, probs, actions, rewards):
        selected_log_probs = []
        for i in range(len(actions)):
            selected_log_probs.append(torch.log(probs[i][0][all_moves.index(action)]))
        return -torch.mean(torch.stack(selected_log_probs) * rewards)
    def compute(self, rewards):
        R = 0
        returns = []
        for r in rewards:
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)
    def clear_memory(self): self.memory = []

class WhiteAICNN(torch.nn.Module):
    def __init__(self, n_outputs):
        super(WhiteAICNN, self).__init__()
        self.memory = []
        self.gamma = 0.99
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, n_outputs),
            torch.nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
    def forward(self, x): return self.network(x)
    def remember(self, state, action, reward): self.memory.append((state, action, reward))
    def trainPolicy(self):
        states, actions, rewards = zip(*self.memory)
        states = torch.stack(states).float().unsqueeze(1)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        discounted_rewards = self.compute(rewards)
        self.optimizer.zero_grad()
        action_probs = []
        for state in states:
            prob = self.network(state.unsqueeze(0))
            action_probs.append(prob)
        loss = self.policyLoss(action_probs, actions, discounted_rewards)
        loss.backward()
        self.optimizer.step()

    def policyLoss(self, probs, actions, rewards):
        selected_log_probs = []
        for i in range(len(actions)):
            selected_log_probs.append(torch.log(probs[i][0][all_moves.index(action)]))
        return -torch.mean(torch.stack(selected_log_probs) * rewards)

    def compute(self, rewards):
        R = 0
        returns = []
        for r in rewards:
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)
    def clear_memory(self): self.memory = []

n_inputs = 64
gamma = 0.99
n_outputs = 4096

# Phong cấp: nếu tốt đi đến hàng ngang cuối cùng, phong cấp thành quân hậu (do là quân nhiều điểm, thực tế trong các ván cờ có thể phong quân khác)
def promotion(position, turn, promotionto):
    if turn == 1:
        pawns = torch.nonzero((position.abs() == 1) & (position > 0), as_tuple=False)
        for pawn in pawns:
            if pawn[0].item() == 0:
                position[pawn[0]][pawn[1]] = promotionto
                return position
            else: continue
        return position
    else:
        pawns = torch.nonzero((position.abs() == 1) & (position < 0), as_tuple=False)
        for pawn in pawns:
            if pawn[0].item() == 7:
                position[pawn[0]][pawn[1]] = promotionto * turn
                return position
            else: continue
        return position

#Getting the index of the move
def getMoveIndex(move): return all_moves.index(move)

# Lựa chọn nước đi dựa trên xác suất
def selectMoves(network_output, position, legal_moves, game_history, turn, last_move, epsilon=0.15):
    if turn == 1:
        for move in legal_moves:
            if move == ((7, 4), (7, 6)) and position[7][4] == 6:
                oppolegal = Board.getAllLegalMoves(new_position, Board.turn * -1, last_move, game_history)
                for oppomovetype in oppolegal:
                    for oppomove in oppomovetype:
                        if oppomove[1] == (7, 5) or oppomove[1] == (7, 4) or oppomove[1] == (7, 6):
                            if move in legal_moves: legal_moves.remove(move)
                            else: break
            elif move == ((7, 4), (7, 2)) and position[7][4] == 6:
                oppolegal = Board.getAllLegalMoves(new_position, Board.turn * -1, last_move, game_history)
                for oppomovetype in oppolegal:
                    for oppomove in oppomovetype:
                        if oppomove[1] == (7, 3) or oppomove[1] == (7, 2) or oppomove[1] == (7, 4):
                            if move in legal_moves: legal_moves.remove(move)
                            else: break
    else:
        for move in legal_moves:
            if move == ((0, 4), (0, 6)) and position[0][4] == -6:
                oppolegal = Board.getAllLegalMoves(new_position, Board.turn * -1, last_move, game_history)
                for oppomovetype in oppolegal:
                    for oppomove in oppomovetype:
                        if oppomove[1] == (0, 5) or oppomove[1] == (0, 4) or oppomove[1] == (0, 6):
                            if move in legal_moves: legal_moves.remove(move)
                            else: break
            elif move == ((0, 4), (0, 2)) and position[0][4] == -6:
                oppolegal = Board.getAllLegalMoves(new_position, Board.turn * -1, last_move, game_history)
                for oppomovetype in oppolegal:
                    for oppomove in oppomovetype:
                        if oppomove[1] == (0, 3) or oppomove[1] == (0, 2) or oppomove[1] == (0, 4):
                            if move in legal_moves: legal_moves.remove(move)
                            else: break

    if random.random() <= epsilon:
        return random.choice(legal_moves)
    legal_move_indices = [getMoveIndex(move) for move in legal_moves]
    mask = torch.zeros_like(network_output)
    mask[0, legal_move_indices] = 1
    masked_output = network_output * mask
    sum_probs = masked_output.sum()
    if torch.all(torch.isnan(masked_output)):
        return random.choice(legal_moves)
    elif sum_probs <= 0:
        return random.choice(legal_moves)
    else:
        action = torch.multinomial(masked_output, 1)
        return all_moves[action.item()]

times = 1 # Số lần chạy
epochs = 1 # Số epoch trong mỗi lần chạy
cnt = 0
# Giả lập
start_time = time.time()
for times in range(times):
    white_neural_network = WhiteAICNN(n_outputs)
    white_neural_network.load_state_dict(torch.load('White.pth'))
    black_neural_network = BlackAICNN(n_outputs)
    black_neural_network.load_state_dict(torch.load('Black.pth'))
    for epoch in range(epochs):
        proto = 5
        cnt = cnt + 1
        pos_his = []
        game_history = []
        last_move = ((0,0),(0,0))
        new_position = Board.position
        pos_his.append(new_position)
        legalmove = Board.getFinalLegalMoves(new_position, Board.turn, last_move, game_history)
        while Board.checkCondition(new_position, Board.turn, legalmove, last_move, pos_his, game_history):
            kings = torch.nonzero(new_position.abs() == 6, as_tuple=False)
            legalmove = Board.getFinalLegalMoves(new_position, Board.turn, last_move, game_history)
            if len(legalmove) == 0:
                Board.checkCondition(new_position, Board.turn, legalmove, last_move, pos_his, game_history)
                break
            action_probs = white_neural_network(torch.tensor(new_position.unsqueeze(0)).float().unsqueeze(0))
            action = selectMoves(action_probs, new_position, legalmove, game_history, Board.turn, last_move)
            if action == ((7, 4), (7, 6)) and new_position[7][4] == 6 or action == ((7, 4), (7, 2)) and new_position[7][4] == 6:
                new_position = Board.castleMove(new_position, action)
                castled += 1
            else: new_position = Board.makeMove(new_position, action)
            new_position = promotion(new_position, Board.turn, proto)
            material_equ = Board.materialEqual(new_position)
            print(new_position)
            whitelaspos = new_position
            pos_his.append(new_position)
            last_move = action
            game_history.append(last_move)
            matdif = Board.materialDifference(pos_his)
            if matdif == 0: black_neural_network.remember(new_position, action, -3)
            else: black_neural_network.remember(new_position, action, matdif)
            print("\n\n")
            print(f"Game {cnt}:")
            Board.changeturn()
            legalmove = Board.getFinalLegalMoves(new_position, Board.turn, last_move, game_history)
            if len(legalmove) == 0:
                Board.checkCondition(new_position, Board.turn, legalmove, last_move, pos_his, game_history)
                break
            blackaction_probs = black_neural_network(torch.tensor(new_position.unsqueeze(0)).float().unsqueeze(0))
            blackaction = selectMoves(blackaction_probs, new_position, legalmove, game_history, Board.turn, last_move)
            if blackaction == ((0, 4), (0, 6)) and new_position[0][4] == -6 or blackaction == ((0, 4), (0, 2)) and new_position[0][4] == -6:
                new_position = Board.castleMove(new_position, blackaction)
                castled += 1
            else: new_position = Board.makeMove(new_position, blackaction)
            new_position = promotion(new_position, Board.turn, proto)
            blalaspos = new_position
            print(new_position)
            count = 0
            for row in new_position:
                for col in row:
                    if col == 6 or col == -6:
                        count = count + 1
            if count >= 3: input()

            material_equ = Board.materialEqual(new_position)
            blacklast_move = blackaction
            matdif = Board.materialDifference(pos_his)
            if matdif == 0: black_neural_network.remember(new_position, blackaction, -3)
            else: black_neural_network.remember(new_position, blackaction, matdif * -1)
            print("\n\n")
            print(f"Game {cnt}:")
            Board.changeturn()
        if Board.winner == 1:
            white_neural_network.remember(whitelaspos, last_move, 200000)
            black_neural_network.remember(blalaspos, blacklast_move, -200000)
            white_won = white_won + 1
        elif Board.winner == -1:
            white_neural_network.remember(whitelaspos, last_move, -200000)
            black_neural_network.remember(blalaspos, blacklast_move, 200000)
            black_won = black_won + 1
        else:
            white_neural_network.remember(whitelaspos, last_move, -500)
            black_neural_network.remember(blalaspos, blacklast_move, -500)
            draw = draw + 1
        white_neural_network.trainPolicy()
        black_neural_network.trainPolicy()
        print("Done!")
        white_neural_network.clear_memory()
        black_neural_network.clear_memory()
    torch.save(white_neural_network.state_dict(), 'White.pth')
    torch.save(black_neural_network.state_dict(), 'Black.pth')

end_time = time.time()
print(f"After battle, white won {white_won} games, black won {black_won} games, drew {draw} games.")
print(f'Elapsed time = {round(end_time - start_time, 3)} seconds.')
