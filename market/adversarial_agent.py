from market.agent import TradingAgent
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F


class QNetwork(nn.Module):
    """Estimates Q value of each action given a state."""
    def __init__(self, args, actions, state_len):
        super().__init__()
        hidden = args.netsize
        self.network = nn.Sequential(
            nn.Linear(state_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, actions)
        )

    def forward(self, x):
        return self.network(x)


class AdversarialAgent(TradingAgent):
    def __init__(self, args, tag=''):
        super().__init__(args.symbol, args.latency, args.interval, lot=100, tag=tag, offset=60e9)
        self.args = args

        # Fixed starting capital — determines how much selling pressure we can exert.
        self.starting_capital = getattr(args, 'capital', 10_000_000)
        self.capital = self.starting_capital

        # RL hyperparameters
        self.batch_size, self.gamma, self.lr = args.batchsize, args.gamma, args.lr
        self.start_e, self.end_e = args.start_e, args.end_e
        self.explore_frac = args.explore_frac
        self.train_freq = args.train_freq
        self.target_freq = args.target_freq
        self.steps_per_day = int(6.5 * 3600 * 1e9 / self.args.interval)

        # Actions:
        #   0 = hold
        #   1 = sell 1 lot at bid
        #   2 = sell 3 lots at bid)
        #   3 = sell 5 lots below bid 
        #   4 = cover / close position 
        actions = 5
        state_len = 7
        self.tot_actions = actions

        self.Q_table = QNetwork(args, actions, state_len)
        self.Q_target = QNetwork(args, actions, state_len)
        self.Q_target.load_state_dict(self.Q_table.state_dict())
        self.optimizer = optim.Adam(self.Q_table.parameters(), lr=self.lr)

        self.rng = np.random.default_rng()
        self.losses, self.buf = [], []
        self.prev_state, self.prev_action, self.prev_mid, self.prev_portval = None, None, None, None
        self.losses_updated = False

        # Position limits driven by capital.  Allow large shorts.
        self.max_short = self.lot * max(5, int(self.starting_capital / 50_000))
        self.price_history = []
        self.price_window = 10
        self.found_first = False
        self.episode, self.global_step, self.eval = -1, -1, False

        # Reward shaping weights
        self.crash_weight = 0.75   # reward for driving price down
        self.profit_weight = 0.25  # reward for making money (so we can keep attacking)

    def get_eps(self):
        total_steps = self.steps_per_day
        progress = min(1.0, self.global_step / (total_steps * self.explore_frac))
        return self.start_e + progress * (self.end_e - self.start_e)

    def _get_state(self):
        """Build a state vector oriented toward crashing."""
        # 1) How far price has moved from open (negative = already crashing, good)
        price_change = (self.mid - self.first_price) / self.first_price

        # 2) Short-term momentum (negative = downward momentum)
        if len(self.price_history) >= self.price_window:
            momentum = (self.mid - self.price_history[0]) / self.price_history[0]
        else:
            momentum = 0.0

        # 3) Normalised position (negative = short)
        position_ratio = self.held / self.max_short

        # 4) Available capital ratio
        capital_ratio = self.capital / self.starting_capital

        # 5) Spread — wider spread means less liquidity, easier to move price
        spread = (self.ask - self.bid) / self.mid

        # 6) Bid-side weakness: low bid quantity relative to ask means sells hit harder
        total_q = self.bidq + self.askq + 1e-8
        bid_weakness = 1.0 - (self.bidq / total_q)  # high when bid side is thin

        # 7) Ask-side depth (buying pressure we'd face when covering)
        ask_depth = self.askq / total_q

        return np.array([price_change, momentum, position_ratio, capital_ratio,
                         spread, bid_weakness, ask_depth], dtype=np.float32)

    def _compute_reward(self):
        """
        Reward = crash_weight * (price decline this step)
               + profit_weight * (portfolio gain this step, normalised)
        """
        if self.prev_mid is None:
            return 0.0

        # Primary objective: price went DOWN (negative delta is good for us)
        price_delta = (self.mid - self.prev_mid) / self.prev_mid
        crash_reward = -price_delta * 100  # scale up; positive when price drops

        # Secondary: portfolio gain (so agent learns to lock in profits and reload)
        if self.prev_portval is not None:
            profit_reward = np.clip((self.portval - self.prev_portval) / self.starting_capital * 100, -1.0, 1.0)
        else:
            profit_reward = 0.0

        # Bonus for achieving large cumulative crash
        cumulative_crash = -(self.mid - self.first_price) / self.first_price
        crash_bonus = max(0.0, cumulative_crash) * 0.1  # small bonus for staying down

        reward = (self.crash_weight * crash_reward +
                  self.profit_weight * profit_reward +
                  crash_bonus)

        return np.clip(reward, -2.0, 2.0)

    def _can_sell(self, lots):
        """Check if we have capital/margin to take on more short."""
        cost = lots * self.lot * self.mid * 0.5  # 50% margin requirement
        return self.capital >= cost and (self.held - lots * self.lot) >= -self.max_short

    def message(self, ct, msg):
        self.ct = ct
        self.handle(ct, msg)

        if msg['type'] == 'lob':
            self.episode_step += 1
            self.global_step += 1
            self.losses_updated = False

            if not self.found_first:
                self.first_price = self.mid
                self.found_first = True

            self.price_history.append(self.mid)
            if len(self.price_history) > self.price_window:
                self.price_history.pop(0)

            state = self._get_state()
            reward = self._compute_reward()

            # Update capital tracking based on realised PnL
            if self.prev_portval is not None:
                pnl = self.portval - self.prev_portval
                self.capital += pnl  # capital grows/shrinks with performance

            ### ADD EXPERIENCE TO REPLAY BUFFER
            if not self.eval and self.prev_state is not None and self.prev_action is not None:
                self.buf.append((self.prev_state, self.prev_action, reward, state))
                if len(self.buf) > self.batch_size * 100:
                    self.buf.pop(0)

            ### EPSILON-GREEDY ACTION SELECTION
            if not self.eval and self.rng.random() < self.get_eps():
                action = self.rng.integers(self.tot_actions)
            else:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                with torch.no_grad():
                    action = self.Q_table(state_tensor).argmax(dim=1).item()

            ### EXECUTE ACTION
            if action == 0:
                pass  # hold — wait for thin book or momentum
            elif action == 1:
                # Sell 1 lot at bid — immediate execution, light pressure
                if self._can_sell(1):
                    yield self.place(-self.lot, self.bid)
            elif action == 2:
                # Sell 3 lots at bid — moderate pressure
                sell_qty = 3
                if self._can_sell(sell_qty):
                    yield self.place(-self.lot * sell_qty, self.bid)
            elif action == 3:
                # Aggressive slam: sell 5 lots BELOW bid to sweep the book
                sell_qty = 5
                if self._can_sell(sell_qty):
                    slam_price = self.bid - (self.ask - self.bid)  # one spread below bid
                    yield self.place(-self.lot * sell_qty, slam_price)
            elif action == 4:
                # Cover / close: buy back to lock in profit and free up capital
                if self.held < 0:
                    yield self.place(-self.held, self.ask)  # buy back at ask

            ### TRAINING
            if not self.eval and len(self.buf) >= self.batch_size and self.global_step % self.train_freq == 0:
                indices = self.rng.integers(0, len(self.buf), size=self.batch_size)
                batch = [self.buf[i] for i in indices]
                states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                batch_actions = torch.tensor([b[1] for b in batch], dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)

                with torch.no_grad():
                    # DDQN: use online net to select action, target net to evaluate
                    next_actions = self.Q_table(next_states).argmax(dim=1, keepdim=True)
                    q_update = self.Q_target(next_states).gather(1, next_actions).squeeze(1)
                    target_q = rewards + self.gamma * q_update

                self.optimizer.zero_grad()
                cur_q = self.Q_table(states).gather(1, batch_actions).squeeze(1)
                loss = F.smooth_l1_loss(cur_q, target_q)  # Huber loss for stability
                loss.backward()
                nn.utils.clip_grad_norm_(self.Q_table.parameters(), 10.0)
                self.optimizer.step()
                self.losses.append(loss.item())
                self.losses_updated = True

            if not self.eval and self.global_step % self.target_freq == 0:
                self.Q_target.load_state_dict(self.Q_table.state_dict())

            self.prev_state = state
            self.prev_action = action
            self.prev_mid = self.mid
            self.prev_portval = self.portval

    def reset(self):
        """Reset per-episode state."""
        super().reset()
        self.episode += 1
        self.episode_step = -1
        self.losses = []
        self.price_history = []
        self.losses_updated = False
        self.prev_state = None
        self.prev_action = None
        self.prev_mid = None
        self.prev_portval = None
        self.found_first = False
        self.capital = self.starting_capital  # reset capital each episode

    def report_loss(self):
        """Returns logging info."""
        if self.eval or not self.losses_updated:
            return
        loss = np.nan if len(self.losses) == 0 else self.losses[-1]
        return self.episode, self.episode_step, self.global_step, loss, np.nan

    def finalize_episode(self, ct, msg):
        return super().finalize_episode(ct, msg)