from market.agent import TradingAgent
import numpy as np
import random

# Torch and RL-specific imports.
import torch
from torch import nn, optim
from torch.nn import functional as F
from util import ReplayBuffer


class QNetwork(nn.Module):
    """ Defines a network that estimates the Q value of each action, given a state. """
    def __init__(self, args, features, actions, net_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(features, net_size),
            nn.ReLU(),
            nn.Linear(net_size, net_size),
            nn.ReLU(),
            nn.Linear(net_size, actions),
        )


    def forward(self, x):
        return self.net(x)

class DDQNAgent(TradingAgent):
    def __init__(self, args, tag=''):

        super().__init__(args.symbol, args.latency, args.interval, lot=100, tag=tag, offset=60e9)
        self.args = args

        # putting all args here so I can see what I am doing

        self.features = 4
        self.actions = 7

        self.max_shares = 1000 #can't have more than 1k shares, defensive against downtrends

        self.result_ema = 1.0
        
        self.symbol = args.symbol  
        self.trips = args.trips          
        self.train_dates = args.train_dates    
        self.train_start = args.train_start    
        self.train_end = args.train_end

        self.latency = args.latency         
        self.interval = args.interval        
        self.trans_cost = args.trans_cost
        
        self.batch_size = args.batchsize
        self.gamma = args.gamma
        self.lr = args.lr
        self.netsize = args.netsize
        
        self.start_e = args.start_e
        self.end_e = args.end_e
        self.explore_frac = args.explore_frac
        self.train_freq = args.train_freq
        self.target_freq = args.target_freq  


        self.ep_steps = int((self.train_end - self.train_start) / self.interval)
        self.all_eps = self.trips * len(self.train_dates)
        self.total_steps = self.all_eps * self.ep_steps

        self.online = QNetwork(self.args, self.features, self.actions, net_size=self.netsize)
        self.target = QNetwork(self.args, self.features, self.actions, net_size=self.netsize)

        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.lr)

        # Loss Functions in Deep Learning: A Comprehensive Review. (n.d.). 
        # Retrieved May 8, 2026, from https://arxiv.org/html/2504.04242v1

        self.criterion = nn.HuberLoss(delta=1.0) 

        self.re_buff = ReplayBuffer(maxlen=100_000, obs_shape=(self.features,), act_shape=(), discrete_actions=True)

        self.last_f = None
        self.last_a = None
        self.losses_updated = False

        # Initialize simulation attributes that should not reset each day.
        self.episode, self.global_step, self.eval = -1, -1, False

    def epsilon(self):
        # no exploration during testing
        if self.eval:
           return 0.0

        # linear anealing that flattens at end_e (clamps frac to 1 so final e is end_e)
        # Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., 
        # & Riedmiller, M. (2013). Playing Atari with Deep Reinforcement Learning 
        # (arXiv:1312.5602). arXiv. https://doi.org/10.48550/arXiv.1312.5602

        # Linear interpolation. (2026). In Wikipedia. https://en.wikipedia.org/
        # w/index.php?title=Linear_interpolation&oldid=1352338849
        frac = min(1.0, self.global_step / (self.args.explore_frac * self.total_steps))
        return self.args.start_e + frac * (self.args.end_e - self.args.start_e)
    
    def message (self, ct, msg):
        self.ct = ct
        self.handle(ct, msg)

        if msg['type'] == 'lob':

            self.episode_step += 1
            self.global_step += 1

            features = self.get_features()

            if self.last_f is not None:
               reward = self.get_reward()
               self.re_buff.add(self.last_f, self.last_a, features, reward)

            if self.eval or random.random() >= self.epsilon():
               with torch.no_grad():
                    q_val = self.online(torch.tensor(features, dtype=torch.float32).unsqueeze(0))
                    action = q_val.argmax(dim=1).item()
            else:
               action = random.randint(0, self.actions - 1)

            for order in self.make_order(action):
                yield order

            buff_size = self.re_buff.s.shape[0] if self.re_buff.full else self.re_buff.n

            if (not self.eval) and (buff_size >= self.batch_size) and (self.global_step % self.train_freq == 0):
                loss = self.train()
                self.losses.append(loss)
                self.losses_updated = True

            if not self.eval and self.global_step % self.target_freq == 0:
                self.target.load_state_dict(self.online.state_dict())

            self.last_f = features
            self.last_a = action
            self.last_portval = self.portval


    def get_reward(self):
        """
        #tracking ema of port profit to stay in appropriate range, avoid
        # explodig gradiants, avoid anomalies & outliers
        raw_result = self.portval - self.last_portval
        self.result_ema = 0.99 * self.result_ema + 0.01 * abs(raw_result)
        
        result_scaled = raw_result / max(self.result_ema, 1e-6) #0 division
        return float(np.clip(result_scaled, -3.0, 3.0)) #avoid anomilies that are too many stds away
        """

        raw_result = self.portval - self.last_portval                           
        scaled = raw_result / 100.0                                         
        penalty = 0.5 * (self.held / self.max_shares) ** 2                      
        return float(scaled - penalty)

        
    def make_order(self, action):

        # cancel stale orders
        if action != 0 and len(self.orders) > 0:
           yield from self.cancel_all()

        # stay within share cap
        can_buy  = self.held < self.max_shares
        can_sell = self.held > -self.max_shares


        # broke the market once, I didn't even know this could be true
        dont_break_the_market = (len(self.snap['bid']) > 0 and len(self.snap['ask']) > 0)

        # action 0 does absolutley nothing (doesn't clear orders), action 4 clears and returns
        if action == 0 or action == 4 or not dont_break_the_market:
           return
    
        elif action == 1: # capture spread
            if can_buy:  yield self.place(+self.lot, self.bid)
            if can_sell: yield self.place(-self.lot, self.ask)
    
        elif action == 2: #bid
            if can_buy:  yield self.place(+self.lot, self.bid)
    
        elif action == 3: #ask
            if can_sell: yield self.place(-self.lot, self.ask)
    
        elif action == 5: #market buy but dont break the book!!!!
            total_ask_depth = sum(a.quantity for a in self.snap['ask'])

            if can_buy and total_ask_depth > self.lot: 
                yield self.place(+self.lot) 
    
        elif action == 6: #market sell but dont break the book!!!!
            total_bid_depth = sum(b.quantity for b in self.snap['bid'])
            
            if can_sell and total_bid_depth > self.lot:
                yield self.place(-self.lot)

    def get_features(self):
        LOB_spread = (self.ask - self.bid) / max(self.mid, 1)  #avoid 0 division
        LOB_imb = self.bidq / (self.bidq + self.askq) #normalized so market scale doesn't matter
        
        current_shares = np.clip(self.held / self.max_shares, -1, 1)
        time_till_trade_end = (self.end - self.ct) / (self.end - self.start) if (self.end - self.start) > 0 else 0
        
        return np.array([LOB_spread, LOB_imb, current_shares, time_till_trade_end], dtype=np.float32) 

    def train(self):
        batch = self.re_buff.sample(self.batch_size)

        q = self.online(batch.observations).gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
             next_a = self.online(batch.next_observations).argmax(dim=1)
             next_q = self.target(batch.next_observations).gather(1, next_a.unsqueeze(1)).squeeze(1)
             eval = batch.rewards + self.gamma * next_q

        loss = self.criterion(q, eval)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def reset (self):
        """ Reset any attributes that should not carry across episodes/days. """
        super().reset()

        # Increment episode and reset losses and total rewards for reporting purposes.
        self.episode_step, self.losses = -1, []

        self.episode += 1

        self.last_f = None
        self.last_a = None

        self.last_portval = None
        self.losses_updated = False

    def report_loss (self):
        """ Returns the episode step, global step, actor loss, and critic loss for logging. """
        # Note: this RL agent has no critic (assuming normal DDQN).
        # Only return a loss entry to log if we've just trained.
        # Otherwise, it will be the same as many prior entries.
        if self.eval or not self.losses_updated: return
        loss = np.nan if len(self.losses) == 0 else self.losses[-1]
        return self.episode, self.episode_step, self.global_step, loss, np.nan

    def finalize_episode(self, ct, msg):
        return super().finalize_episode(ct,msg)

