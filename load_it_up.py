import os
import pandas as pd
import numpy as np

def load_run(run_dir):
    path = os.path.join(run_dir, "messages.parquet")
    return pd.read_parquet(path)

class Tokenizer:
    """
    We are making discrete the componenets of our state space, order types
    are already discrete, but price, volume, and time bet events / LOB message
    are all continuous. This class tranforms them to tokens to discretsize 
    (i spelt that wrong) everything so we can represent each message as a 
    sequence of 5 cells
    """
    def __init__(self, size_bins=32, time_bins=32, price_window=64):
        self.size_bins = size_bins
        self.time_bins = time_bins
        
        self.price_window = price_window
        
        self.size_edges = None
        self.time_edges = None

        
    def fit(self, df):
        #quantile binning (we're specifically using quantile binning from project 7
        #since we don't want to impose equal arbitrary bin widths on the data)
        #linear quantile binning for size edges
        _, self.size_edges = pd.qcut(df["size"], self.size_bins, retbins=True, duplicates="drop")

        #logarithmic quantile bining because the scale of time increases by orders of magnitude
        deltas = df["timestamp"].diff().dropna()
        #filter zeros for geomspace  we'll handle 0 deltas at tokenize time
        positive_deltas = deltas[deltas > 0]
        self.time_edges = np.geomspace(positive_deltas.min(), positive_deltas.max(), self.time_bins + 1)

        # ensure lower thresholds are 0
        self.size_edges[0] = 0
        self.time_edges[0] = 0

        self.size_bins = len(self.size_edges) - 1
        
        
        self.event_offset = 0
        self.side_offset = 5
        self.price_offset = self.side_offset + 2
        self.size_offset = self.price_offset + (2 * self.price_window + 1)
        self.time_offset = self.size_offset + self.size_bins
        self.event_size = self.time_offset + self.time_bins
        
        return self

    def tokenize(self, df):

        #we initially had 5 separate columns, where each feature was
        #its own vector, but then we couldn't figure out how to forward pass that and take its loss
        #bc it was 5 different streams. now it's all flat instead of a tensor
        all_data = np.empty(5 * len(df), dtype=np.int64)
        
        #state space of the data is feature-size sequence in a flat stream
        #with features as follows  event type: 
        #limit order placed, market order placed, order cancel, order reduce order; 
        #side:buy or sell, price, volume, and time bet event/message (event / trade latency).
        
        all_data[0::5] = df["event_type"].values
        all_data[1::5] = df["side"].values + self.side_offset

        #squashing stock price into our bins so all extremes beyond a certain point
        #are just in the last bin. stock price is binned by ticks, so the difference
        #between bins is a tick, bar from the extreme which is everything beyond a certain
        #price point (which should capture our entire price space for a stable market, we're 
        #binning this way to create an ideal space for stable markets specifically)
        offset = (df["price"] - df["fundamental"]).astype(np.int64).values
        offset = np.clip(offset, -self.price_window, self.price_window)

        #adding in the window to make sure everything is positive (range[0, price_window))
        all_data[2::5] = offset + self.price_window + self.price_offset
        all_data[3::5] = np.nan_to_num(pd.cut(df["size"], bins=self.size_edges,
                   labels=False, include_lowest=True), nan=0).astype(np.int64) + self.size_offset
        deltas = np.diff(df["timestamp"].values, prepend=df["timestamp"].iloc[0])
        all_data[4::5] = np.nan_to_num(pd.cut(deltas, bins=self.time_edges,
                   labels=False, include_lowest=True), nan=0).astype(np.int64) + self.time_offset

        return all_data

    def build(run_dirs, tokenizer):
        out = []
        for d in run_dirs:
            df = load_run(d)
            # model is only learning the market (agent blind), but we're including metadata in an adjacent
            # df to find our sources of entropy later for analysis
            out.append((tokenizer.tokenize(df), df[["agent_id", "agent_class"]]))
        return out
