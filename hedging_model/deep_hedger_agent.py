from collections import deque
import sys
from pathlib import Path

_HEDGING_DIR = Path(__file__).parent
if str(_HEDGING_DIR) not in sys.path:
    sys.path.insert(0, str(_HEDGING_DIR))
    
import numpy as np
import torch

from market.agent import TradingAgent

from .configs import HedgerConfig
from .features import FeatureExtractor
from .hedger import LSTMHedger


class DeepHedgerAgent(TradingAgent):

    def __init__(
        self,
        symbol,
        minlat,
        interval,
        hedger_checkpoint_path,
        cost_rate_bps=10.0,
        position_scale=100,
        offset=0.0,
        tag='',
    ):
        super().__init__(symbol, minlat, interval, lot=None, offset=offset, tag=tag)

        self.cost_rate = cost_rate_bps / 10_000.0
        self.position_scale = position_scale

        bundle = torch.load(
            hedger_checkpoint_path, weights_only=False, map_location="cpu"
        )
        self.hedger_cfg = bundle["hedger_config"]
        self.fx = FeatureExtractor(self.hedger_cfg)
        self.hedger = LSTMHedger(self.hedger_cfg, n_features=self.fx.n_features)
        self.hedger.load_state_dict(bundle["model_state_dict"])
        self.hedger.eval()

        self.hidden_state = None
        self.current_target = 0.0
        self.last_mid = None
        self.recent_returns = deque(maxlen=self.hedger_cfg.return_window)
        self.step = 0
        self.next_decision_time = None
        self.spot_anchor = None

    def reset(self):
        super().reset()
        self.hidden_state = None
        self.current_target = 0.0
        self.last_mid = None
        self.recent_returns = deque(maxlen=self.hedger_cfg.return_window)
        self.step = 0
        self.next_decision_time = self.start
        self.spot_anchor = None

    def message(self, ct, msg):
        super().message(ct, msg)

        if msg['type'] != 'lob':
            return

        if self.mid is None:
            return

        if ct < self.next_decision_time:
            return

        if self.step >= self.hedger_cfg.n_steps:
            return

        mid_dollars = self.mid / 100.0

        if self.spot_anchor is None:
            self.spot_anchor = mid_dollars

        mid_rescaled = mid_dollars * (100.0 / self.spot_anchor)

        if self.last_mid is not None and self.last_mid > 0:
            self.recent_returns.append(np.log(mid_rescaled / self.last_mid))
        self.last_mid = mid_rescaled
        
        target_fraction = self._policy(mid_rescaled)
        target_shares = int(round(target_fraction * self.position_scale))
        
        for m in self.adjust(target_shares):
            yield m

        self.current_target = target_fraction
        self.step += 1
        self.next_decision_time = ct + self.interval

    def _policy(self, mid):
        tau = max(self.hedger_cfg.T - self.step * self.hedger_cfg.dt, 0.0)

        rr = list(self.recent_returns)
        pad = self.hedger_cfg.return_window - len(rr)
        if pad > 0:
            rr = [0.0] * pad + rr

        with torch.no_grad():
            feats = self.fx(
                spot=torch.tensor([mid], dtype=torch.float32),
                time_remaining=torch.tensor([tau], dtype=torch.float32),
                position=torch.tensor([self.current_target], dtype=torch.float32),
                cost=torch.tensor([self.cost_rate], dtype=torch.float32),
                recent_returns=torch.tensor(rr, dtype=torch.float32).unsqueeze(0),
            ).unsqueeze(1)

            pos, self.hidden_state = self.hedger(feats, self.hidden_state)
            return pos.squeeze().item()
