import random
import torch
import numpy as np
from collections import deque
from present_game import SnakeGamePlaySelf, Direction, Point
from model import Linear_QNet
from model import Qtrainer
from help import plot

Max_Mem = 100000
Batch_Size = 1000
alpha = 0.005

class Agent:

    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.95 #discount rate
        self.alpha = alpha
        self.memory = deque(maxlen = Max_Mem)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = Qtrainer(self.model, gamma = self.gamma, alpha = self.alpha,)


    def get_state(self, present_game):
        head = present_game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = present_game.direction == Direction.LEFT
        dir_r = present_game.direction == Direction.RIGHT
        dir_u = present_game.direction == Direction.UP
        dir_d = present_game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and present_game.is_collision(point_r)) or
            (dir_l and present_game.is_collision(point_l)) or
            (dir_u and present_game.is_collision(point_u)) or
            (dir_d and present_game.is_collision(point_d)),

            # Danger right
            (dir_u and present_game.is_collision(point_r)) or
            (dir_d and present_game.is_collision(point_l)) or
            (dir_l and present_game.is_collision(point_u)) or
            (dir_r and present_game.is_collision(point_d)),

            # Danger left
            (dir_d and present_game.is_collision(point_r)) or
            (dir_u and present_game.is_collision(point_l)) or
            (dir_r and present_game.is_collision(point_u)) or
            (dir_l and present_game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            present_game.food.x < present_game.head.x,  # food left
            present_game.food.x > present_game.head.x,  # food right
            present_game.food.y < present_game.head.y,  # food up
            present_game.food.y > present_game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, over):
        self.memory.append((state, action, reward, next_state, over))

    def train_long_memory(self):
        if len(self.memory) > Batch_Size:
            mini_sample = random.sample(self.memory, Batch_Size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, overs)


    def train_short_memory(self, state, action, reward, next_state, over):
        self.trainer.train_step(state, action, reward, next_state, over)

    def get_action(self, state):
        self.epsilon = 80 -self.num_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
     score_list = []
     plot_mean_scores = []
     total_score = 0
     record = 0
     agent = Agent()
     game = SnakeGamePlaySelf()
     while True:
         #get current state
         last_state = agent.get_state(game)

         #get move
         final_move = agent.get_action(last_state)

         #perform move and get new state
         reward, over, score = game.play_step(final_move)
         new_state = agent.get_state(game)

         #train short memory
         agent.train_short_memory(last_state, final_move, reward, new_state, over)

         #remember
         agent.remember(last_state, final_move, reward, new_state, over)

         if over:
            #train long memory, plot result
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Snake', agent.num_games, 'Score', score, 'Record', record)

            #plot with pytorch.

            score_list.append(score)
            total_score +=score
            mean_score = total_score / agent.num_games


            plot_mean_scores.append(mean_score)
            plot(score_list, plot_mean_scores)

if __name__ == '__main__':
    train()