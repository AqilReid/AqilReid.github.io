import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name = 'reinforce_mod.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Qtrainer:
    def __init__(self, model, gamma, alpha):
        self.gamma = gamma
        self.alpha = alpha
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.alpha)
        self.lossfunc = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, over):
        state = torch.tensor(state, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.long)
        reward = torch.tensor(reward, dtype = torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            over = (over, )

            # predict Q value with current state
            predict = self.model(state)

            target = predict.clone()
            for idx in range(len(over)):
                Qnew = reward[idx]
                if not over[idx]:
                    Qnew = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

                target[idx][torch.argmax(action).item()] = Qnew

            self.optimizer.zero_grad()
            loss = self.lossfunc(target, predict)
            loss.backward()

            self.optimizer.step()





