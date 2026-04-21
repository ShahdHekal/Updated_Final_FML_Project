from random import random, randint
from .agent import TradingAgent


class AdversarialAgent(TradingAgent):
    def __init__(self, symbol, minlat, interval, lot=None, offset=0.0, tag=''):
        super().__init__(symbol, minlat, interval, lot=lot, offset=offset, tag=tag)
        self.next_trade = None
        self.action_count = 0

    def reset(self):
        self.next_trade = self.start
        self.action_count = 0

    def message(self, ct, msg):
        # First do normal trading-agent bookkeeping
        super().message(ct, msg)

        # Wait until our next eligible action time
        if self.next_trade is None or ct < self.next_trade:
            return

        # Schedule next opportunity
        self.next_trade = ct + self.interval

        # 1. Occasional random market order
        if random() < 0.10:
            q = randint(1, 10)
            if random() < 0.5:
                yield self.place(q)      # market buy
            else:
                yield self.place(-q)     # market sell
            self.action_count += 1

        # 2. Small extra sell pressure
        if random() < 0.03:
            q = randint(1, 10)
            yield self.place(-q)         # market sell
            self.action_count += 1

        # 3. Larger sell burst
        if random() < 0.01:
            q = randint(10, 20)
            yield self.place(-q)         # market sell
            self.action_count += 1

        # 4. Small spoof-like ask limit order
        if random() < 0.02 and len(self.orders) < 2:
            q = randint(5, 15)
            price = self.ask + 1 if self.ask is not None else None
            if price is not None:
                yield self.place(-q, p=price)   # limit sell
                self.action_count += 1

        # 5. Rare cancellation of resting orders
        if random() < 0.01 and len(self.orders) > 0:
            for m in self.cancel_all():
                yield m
            self.action_count += 1