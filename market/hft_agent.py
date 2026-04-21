import numpy as np
from collections import deque

from market.agent import TradingAgent

class HFTAgent(TradingAgent):
    """ 
    HFT market maker with online ML.
    
    Every tick: observe book → train model → place skewed quotes.
    Profits from spread capture + ML directional lean.
    """

    def __init__ (self, symbol, minlat=1_000, interval=100_000_000, lot=50, levels=5, base_spread=1, inventory_penalty=0.01,
                  ml_scale=50.0, learning_rate=0.01, max_inventory=500, tag=''):
        super().__init__(symbol, minlat, interval, lot=lot, tag=tag, offset=1e9)
        self.levels = levels # how many book levels to look up
        self.base_spread = base_spread #half-spread in cents
        self.inventory_penalty = inventory_penalty # cents of skew per share held
        self.ml_scale = ml_scale # max ml skew ($+-0.5)
        self.max_inventory = max_inventory # hard position limit (+=500 shares)

        # Linear model: prediction = dot(weights, features) + bias
        # 5 weights, one per feature.
        self.weights = np.zeros(5)
        self.bias = 0.0
        self.lr = learning_rate

        # Rolling window of last 200 mid-prices and imbalances.
        # (200 = ~20s history at 100ms polling)
        self.mid_prices = deque(maxlen=200)
        self.imbalances = deque(maxlen=200)
        self.feature_buffer = deque(maxlen=200)
        self.tick_count = 0
        self.trades_made = 0

 
    def message(self, ct, msg):
        """
        Called by exchange every tick with LOB/order update
        - lob: run strategy with new order book
        - executed: count fulfilled order
        """
        super().message(ct, msg)

        if msg['type'] != 'lob':
            if msg['type'] != 'executed':
                self.trades_made += 1
            return
        
        # run full strategy
        # no book data yet
        if not self.snap['bid'] or not self.snap['ask']:
            return
        if self.bid is None or self.ask is None:
            return
        
        # market state
        mid = self.mid

        # volume imbalance, e.g. positive = more bid volume = buying pressure = price likely rises
        bid_vol = sum(self.snap['bid'][i].quantity for i in range(min(self.levels, len(self.snap['bid']))))
        ask_vol = sum(self.snap['ask'][i].quantity for i in range(min(self.levels, len(self.snap['ask']))))
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)

        # Store in rolling buffers
        self.mid_prices.append(mid)
        self.imbalances.append(imbalance)
        self.tick_count += 1

        if self.tick_count < 5: # warmup 
            return
        
        # extract features and store for future labeling
        features = self.get_features()
        self.feature_buffer.append((features.copy(), mid))

        self.train()

        # predict and trade
        signal = float(np.dot(self.weights, features) + self.bias)
        yield from self.act(signal)
    
    def get_features(self) -> np.ndarray:
        """
        Online ML
        5 microstructure features:
            [0] Current volume imbalance — order flow pressure
            [1] 1-tick return — immediate momentum
            [2] 5-tick return — short-term trend (directional moves)
            [3] 10-tick avg imbalance — smoothed pressure
            [4] 20-tick volatility — regime indicator
        """
        mids = list(self.mid_prices)
        imbs = list(self.imbalances)
        mid = mids[-1]

        f = np.zeros(5)
        # Feature 0: Volume imbalance right now
        # Range ~[-1, +1]. Positive = more bids = bullish.
        f[0] = imbs[-1]
        # Feature 1: 1-tick return (% change from last tick)
        # Captures immediate momentum/mean-reversion
        f[1] = (mids[-1] - mids[-2]) / (mid + 1e-9) if len(mids) >= 2 else 0
        # Feature 2: 5-tick return (% change from 5 ticks ago)
        # Captures short-term trends
        f[2] = (mids[-1] - mids[-5]) / (mid + 1e-9) if len(mids) >= 5 else 0
        # Feature 3: Average imbalance over last 10 ticks
        f[3] = np.mean(imbs[-10:]) if len(imbs) >= 10 else imbs[-1]
        # Feature 4: Normalized price volatility over 20 ticks
        f[4] = np.std(mids[-20:]) / (mid + 1e-9) if len(mids) >= 20 else 0
        return f
    
    def train(self) -> None:
        """
        Train using features from 5 ticks ago labeled with current price.
        """
        # Need at least 6 entries: features from 5 ticks ago + current price
        if len(self.feature_buffer) < 6:
            return

        # Get features from 5 ticks ago
        past_features, past_mid = self.feature_buffer[-6]
        current_mid = self.mid_prices[-1]

        # Label: did price go up since those features were observed?
        label = 1.0 if current_mid > past_mid else -1.0

        # Prediction using past features
        pred = float(np.dot(self.weights, past_features) + self.bias)

        # Loss update (only when wrong or not confident)
        margin = label * pred
        if margin < 1.0:
            self.weights -= self.lr * (-label * past_features + 0.001 * self.weights)
            self.bias -= self.lr * (-label)

    def act(self, signal: float):
        """
        Decide between:
        1. Aggressive order (cross spread) — when signal is very strong
        2. Passive quotes (provide liquidity) — normal operation
        3. Do nothing — if price hasn't moved enough to justify repricing
        """
        inventory = self.held

        # Aggressive: Cross the spread when signal is very strong 
        if abs(signal) > 1.5:
            size = max(1, self.lot // 2)
            if signal > 1.5 and inventory < self.max_inventory:
                yield self.place(size, self.ask + 10) # lift ask
            elif signal < -1.5 and inventory > -self.max_inventory:
                yield self.place(-size, self.bid - 10) # hit bid
            return
        
        # Passive: cancel old quotes, place new ones
        yield from self.cancel_all()

        # ML skew: shift quotes in predicted direction
        ml_skew = np.tanh(signal) * self.ml_scale

        # Inventory skew: push quotes to flattening position
        inv_skew = -self.inventory_penalty * inventory

        total_skew = ml_skew + inv_skew

        # Quote prices (mid += spread + combined skew)
        bid_price = int(self.mid - self.base_spread + total_skew)
        ask_price = int(self.mid + self.base_spread + total_skew)
        bid_price = max(1, bid_price)
        ask_price = max(bid_price + 1, ask_price)

        # Asymmetric sizing (reduce on overloaded size)
        inv_ratio = np.clip(inventory / self.max_inventory, -1.0, 1.0)
        bid_size = max(0, int(self.lot * (1.0 - inv_ratio)))
        ask_size = max(0, int(self.lot * (1.0 + inv_ratio)))

        # Hard inventory cutoff
        if inventory >= self.max_inventory:
            bid_size = 0
        if inventory <= -self.max_inventory:
            ask_size = 0

        # Place orders 
        if bid_size > 0:
            yield self.place(bid_size, bid_price)
        if ask_size > 0:
            yield self.place(-ask_size, ask_price)
