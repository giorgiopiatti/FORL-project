import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium import spaces
from environment.briscola_base.utils import Roles
import torch.nn as nn
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from agents.heuristic_agent import HeuristicAgent
from environment.briscola_communication.actions import BriscolaCommsAction


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(
                probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(logits.device)
            logits = torch.where(self.masks, logits,
                                 torch.tensor(-1e8).to(logits.device))
            super(CategoricalMasked, self).__init__(
                probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p,
                              torch.tensor(0.0).to(self.masks.device))
        return -p_log_p.sum(-1)


class AgentRNN(nn.Module):
    def __init__(self, rnn_out_size, hidden_dim, num_actions, current_round_shape, previous_round_shape):
        self.rnn_out_size = rnn_out_size
        super().__init__()
        self.lstm = nn.LSTM(np.prod(previous_round_shape), rnn_out_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.prod(current_round_shape) +
                       rnn_out_size, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(
                nn.Linear(hidden_dim, num_actions), std=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.prod(current_round_shape) +
                       rnn_out_size, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1)
        )

        self.offset_round = current_round_shape[-1]

    def get_states(self, x, lstm_state, done):
        x_features_round = x[:, :, :self.offset_round]
        x_previous_round = x[:, :,  self.offset_round:]

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        x_previous_round = x_previous_round.reshape(
            (-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))

        out_lstm = []
        for xr, d in zip(x_previous_round, done):
            o, lstm_state = self.lstm(
                xr.unsqueeze(0),  ((1.0 - d).view(1, -1, 1) * lstm_state[0], (1.0 - d).view(1, -1, 1) * lstm_state[1]))
            out_lstm += [o]

        out_lstm = torch.flatten(torch.cat(out_lstm), 0, 1)
        new_hidden = torch.cat([x_features_round.squeeze(1),
                                out_lstm], dim=1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, action_mask, lstm_state, done, action=None,  deterministic=False):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        action_mask = action_mask.squeeze()
        logits = self.actor(hidden)
        probs = CategoricalMasked(logits=logits, masks=action_mask)
        if action is None and not deterministic:
            action = probs.sample()
        if action is None and deterministic:
            if len(action_mask.shape) == 1:
                action_mask = action_mask.unsqueeze(0)
            logits[~action_mask] = -torch.inf
            action = logits.argmax(axis=-1)
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(
                probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(logits.device)
            logits = torch.where(self.masks, logits,
                                 torch.tensor(-1e8).to(logits.device))
            super(CategoricalMasked, self).__init__(
                probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p,
                              torch.tensor(0.0).to(self.masks.device))
        return -p_log_p.sum(-1)


class AgentNN(nn.Module):
    def __init__(self, hidden_dim, num_actions, observation_shape):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_shape, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_shape, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, num_actions), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action_mask, action=None, deterministic=False):
        logits = self.actor(x)
        probs = CategoricalMasked(logits=logits, masks=action_mask)
        if action is None and not deterministic:
            action = probs.sample()
        if action is None and deterministic:
            logits[~action_mask] = -torch.inf
            action = logits.argmax(axis=-1)
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
