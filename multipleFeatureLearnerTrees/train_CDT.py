"""
PPO with cascading decision tree (CDT) as policy function approximator

Taken from the following:
https://github.com/quantumiracle/Cascading-Decision-Tree/blob/d660c442175c3a05bc70c1a45eb3eb9d91242260/src/cdt/deprecated/cdt_rl_train.py

Modified loss function
Added wrapper function which strings together CDTs into a boosted forest
"""
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.distributions import Categorical
import numpy as np
import pandas as pd
from .CDT_hierarchical_fl import CDT
import sys
sys.path.append("..")
# from rl import StateNormWrapper


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, learner_args,
                 sample_weights=None, state_mask=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_path = learner_args.get('model_path', None)
        self.device = learner_args.get('device', None)
        self.learning_rate = learner_args.get('lr', 0.0005)
        self.gamma = learner_args.get('gamma', 0.98)
        self.K_epoch = learner_args.get('K_epoch', 3)
        self.beta = learner_args.get('beta', 1.0)
        self.weight_decay = learner_args.get('weight_decay', 0.1)
        self.exp_gamma = learner_args.get('exp_scheduler_gamma', 0.98)
        self.learn_probabilities = learner_args.get('learn_probabilities', 0)
        self.loss_batch_size = learner_args.get('loss_batch_size', 1)

        self.sample_weights = (
            self._to_tensor(sample_weights)
            if not isinstance(sample_weights, type(None)) else None
        )
        if not isinstance(state_mask, type(None)):
            self.state_mask = self._to_tensor(state_mask, dtype=torch.bool)
            args = learner_args.copy()
            args['input_dim'] = state_dim
            self.cdt = CDT(args).to(self.device)
        else:
            self.state_mask = None
            self.cdt = CDT(learner_args).to(self.device)

        self.softmax = nn.Softmax(dim=1)
        self.calc_probs = lambda x: self.cdt.forward(x)[1]

        if learner_args.get("momentum") is not None:
            momentum = learner_args.get("momentum")
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=momentum
            )
        else:
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.exp_gamma
        )
        
        self.train_losses = []
        self.val_losses = []

    def make_batch(self):
        batch = (
            self.train[["s", "rs", "s'"]].copy()
        )
        return batch
    
    def calc_train_loss(self):
        total_train_loss = None
        for idx, row in self.train.iterrows():
            s, rs, s_prime = (
                self._to_tensor([row["s"]]),
                self._to_tensor([row["rs"]]),
                self._to_tensor([row["s'"]])
            )
            train_loss = self._to_numpy(
                self.calc_loss(s, rs, s_prime, idx)
            )
            if isinstance(total_train_loss, type(None)):
                total_train_loss = train_loss
            else:
                total_train_loss += train_loss

        self.train_losses.append((total_train_loss / len(self.train))[0])

    def calc_val_loss(self):
        total_val_loss = None
        for idx, row in self.val.iterrows():
            s, rs, s_prime = (
                self._to_tensor([row["s"]]),
                self._to_tensor([row["rs"]]),
                self._to_tensor([row["s'"]])
            )
            val_loss = self._to_numpy(
                self.calc_loss(s, rs, s_prime)
            )
            if isinstance(total_val_loss, type(None)):
                total_val_loss = val_loss
            else:
                total_val_loss += val_loss

        self.val_losses.append((total_val_loss / len(self.val))[0])

    def train_net(self):
        batch = self.make_batch()
        loss_batch = []
        for epoch in range(self.K_epoch):
            for i, row in batch.iterrows():
                s, rs, s_prime = (
                    self._to_tensor([row["s"]]),
                    self._to_tensor([row["rs"]]),
                    self._to_tensor([row["s'"]])
                )
                loss = self.calc_loss(s, rs, s_prime, i=i)
                loss_batch.append(loss)
                
                if i % self.loss_batch_size == 0 or i == len(batch) - 1:
                    self.optimizer.zero_grad()
                    torch.cat(loss_batch, 0).mean().backward()
                    loss_batch = []
                    self.optimizer.step()
                    
            self.calc_train_loss()
            self.calc_val_loss()

    def calc_loss(self, s, rs, s_prime, i=None):
        if isinstance(self.state_mask, torch.Tensor):
            # apply estimator level state mask
            s = s.masked_select(self.state_mask).view(s.shape[0], -1)
            s_prime = s_prime.masked_select(
                self.state_mask
            ).view(s.shape[0], -1)

        # get prediction and target
        q_vals = self.calc_probs(s)
        td_target = self.softmax(rs) + self.gamma * self.calc_probs(s_prime)
        td_target = td_target / torch.sum(td_target, dim=1)
        max_target_mask = torch.eq(td_target, td_target.max(dim=1)[0]).float()

        # calc cross entropy loss using target mask or real probabilities
        if not self.learn_probabilities:
            loss = self.softXEnt(q_vals, max_target_mask)
        else:
            loss = self.softXEnt(q_vals, td_target)

        # orient loss so minimum is 0, only effects learn_probabilities
        loss -= self.softXEnt(td_target, td_target)

        # mask loss for largest decision signal in this sample
        loss = torch.sum(loss * max_target_mask, dim=1)
        if isinstance(self.sample_weights, torch.Tensor) and i is not None:
            # apply sample weights
            loss *= self.sample_weights[i]

        return loss

    def softXEnt(self, input, target):
        # https://www.desmos.com/calculator/zlzedjdfif
        # log will only be nan when input is 0 or 1
        logprobs = torch.log(input + 1e-44)
        inv_logprobs = torch.log(1 - input + 1e-44)
        loss_per_action = - (
            (target * logprobs) + ((1-target) * inv_logprobs)
        )
        return loss_per_action

    def choose_action(self, s):
        s = torch.from_numpy(s).unsqueeze(0).float().to(self.device)
        if isinstance(self.state_mask, torch.Tensor):
            s = s.masked_select(self.state_mask).view(s.shape[0], -1)
        # ensure input state shape is correct
        probs = self.calc_probs(s).squeeze()
        a = torch.argmax(probs, dim=-1).item()
        return a, probs

    def load_model(self):
        self.load_state_dict(torch.load(self.model_path))

    def _to_tensor(self, arr, dtype=torch.float):
        return torch.tensor(arr, dtype=dtype).to(self.device)
    
    def _to_numpy(self, torch_tensor):
        return np.copy(
            torch_tensor.detach().cpu().numpy()
        )


def run(
    train_env, learner_args, reuse_data=None, train=False, test=False,
    train_ratio=0.80, state_mask=None
):
    if not isinstance(state_mask, type(None)):
        state_dim = sum(state_mask)
    else:
        state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    learner_args["input_dim"] = state_dim
    learner_args["output_dim"] = action_dim

    print(state_dim, action_dim)
    model = PPO(  # model
        state_dim=state_dim, action_dim=action_dim, learner_args=learner_args, state_mask=state_mask
    )
    
    if isinstance(reuse_data, type(None)):
        data = []

        s = train_env.reset()
        done = False
        while not done:
            if train:
                a, q_vals = None, None
                s_prime, rs, done, _ = train_env.step_all()
                data.append([s, rs, s_prime, q_vals, a])
            else:
                q_vals = model.calc_mean_of_q_vals(s)
                a = np.argmax(q_vals)
                s_prime, r, done, a = train_env.step(a)
                data.append([s, r, s_prime, q_vals, a])
            if done:
                break
            s = s_prime

        data = pd.DataFrame(data, columns=["s", "rs", "s'", "Qs", "a"])
    else:
        data = reuse_data

    if train:
        model.train = data.iloc[: int(len(data) * train_ratio)]
        model.val = data.iloc[int(len(data) * train_ratio): -1]
        model.train_net()

    train_env.close()
    return model, data, data.iloc[: int(len(data) * train_ratio)], data.iloc[int(len(data) * train_ratio): -1]