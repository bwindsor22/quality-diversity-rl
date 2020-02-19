import math
import random
from itertools import count
from datetime import  datetime

import gym
import gym_gvgai
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from IPython import display

from models.dqn import DQN
from models.replay_memory import ReplayMemory, Transition

steps_done = 0
is_ipython = 'inline' in matplotlib.get_backend()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize = T.Compose([T.ToPILImage(),
                    T.Resize(30),
                    T.ToTensor()])

plt.ion()


def run_training_for_params(
        game_level = 'gvgai-aliens-lvl0-v0',
        BATCH_SIZE = 128,
        GAMMA = 0.999,
        EPS_START = 0.9,
        EPS_END = 0.05,
        EPS_DECAY = 200,
        TARGET_UPDATE = 10,
        LINEAR_INPUT_SCALAR=8,
        KERNEL=5
):
    env = gym.make(game_level)
    global steps_done
    steps_done = 0

    def get_screen():
        screen = env.render(mode='rgb_array').transpose((2, 0, 1))

        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

        # (this doesn't require a copy)
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).to(device)

    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    # Training Loop
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape

    print(screen_height, " ", screen_width)

    n_actions = env.action_space.n
    episode_durations = []

    policy_net = DQN(screen_height, screen_width, LINEAR_INPUT_SCALAR, KERNEL, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, LINEAR_INPUT_SCALAR, KERNEL, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)


    num_episodes = 30
    results = []
    for i_episode in range(num_episodes):
        print("Episode #: ", i_episode)
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()

        state = current_screen - last_screen
        sum_score = 0
        aliens_killed = 0
        for t in count():
            # Plot Code
            # plt.figure()
            # plt.imshow(current_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
            # interpolation='none')
            # plt.title('current screen')
            # plt.show()
            # display.clear_output(wait = True)
            # Select and perform an action
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            if reward == 2:
                aliens_killed += 1

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            sum_score += reward
            if t % 100 == 0:
                print("Time: ", t, " Reward: ", reward[0], "Score: ", sum_score[0], " Aliens Killed: ",aliens_killed)
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                # plot_durations()
                if reward == 2:
                    print("WIN WIN WIN\n", "WIN \n", "WIN \n", "WIN \n")
                    print("Score: ", sum_score.item(), " Aliens Killed: ", aliens_killed)
                    results.append([sum_score.item(), aliens_killed, 0])
                elif reward == -1:
                    print("LOSE LOSE LOSE\n", "LOSE \n", "LOSE \n", "LOSE \n")
                    print("Score: ", sum_score.item(), " Aliens Killed: ", aliens_killed)
                    results.append([sum_score.item(), aliens_killed, 1])
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    score, killed, won = results[-1]
    results = np.array(results, dtype=np.int32)
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    params_name = f'batch_{BATCH_SIZE}_linear_{LINEAR_INPUT_SCALAR}_kernel_{KERNEL}_score_{score}'
    run_name = '{}_{}_results'.format(time, params_name)
    np.savetxt(run_name, results)
    env.close()
    return score

if __name__ == '__main__':
    run_training_for_params()