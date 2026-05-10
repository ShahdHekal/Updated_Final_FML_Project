import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def train(model, train_arrays, seq_len=1000, batch_size=64, lr=1e-3, epochs=10, device="cuda"):
    """
    seq length, how many tokens per batch, and event is 5 tokens, so this is just
    200 events per batch, and 12,800 events consumed in parallel. per training day,
    and as an exampl, according to message size from my last run (373071) that's about 30-ish
    parallel processes. not bad.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    chunks = torch.cat([torch.from_numpy(t[:(len(t) // seq_len) * seq_len]).reshape(-1, seq_len)
                        for t in train_arrays], dim=0)
                        
    loader = DataLoader(TensorDataset(chunks), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total = 0
        
        for (batch,) in loader:
            loss = model.loss(batch.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch}  loss {total / len(loader)}") #for tracking purposes
    return model


def test(model, token_arrays, seq_len=1000, batch_size=64, device="cuda"):
    model.eval()
    results = []
    
    for t in token_arrays:
        chunks = torch.from_numpy(t[:(len(t) // seq_len) * seq_len]).reshape(-1, seq_len).to(device)

        per_token_chunks = []
        with torch.no_grad():
            #calculating entropy / residuals between two populations based on features
            #(so the degree of surprise / hidden structure in each feature.
            for i in range(0, chunks.shape[0], batch_size):
                batch = chunks[i:i+batch_size].to(device)
                pt = model.loss(batch, reduction="none").cpu().numpy().flatten()
                per_token_chunks.append(pt)
        per_token = np.concatenate(per_token_chunks)
        
        #residuals between events     
        n = (len(per_token) // 5) * 5
        per_event = per_token[:n].reshape(-1, 5).sum(axis=1)
                
        results.append({
            "total": float(per_event.mean()), #total, describes hidden structure between two different configurations/markets/models
            "per_event": per_event,
            "per_token": per_token,
        })
    return results


def bootstrap(test_output, baseline_results=None, n=1000):
    """
    bootstrapping to get a confidence percentile
    """
    # test output on any configuration that isn't HT-Stable
    test_means = np.array([r["total"] for r in test_output])
    # test output on HT-Stable (filtering out noise too)
    base_means = np.array([r["total"] for r in baseline_results])

    #calculating the residual 1000 times from random entropy samples
    #IMPORTANAT: literally speaking, this is the difference between the cross
    #Entropy of two distributions. The cross entropy of a distribution is just 
    #the shannon entropy + "code waste" or the KL divergence, the gap between our model and the truth.
    #residual = shannon entroy test + KL divergence test - shannon entropy baseline - KL divergence baseline.
    #for (cross entropy baseline = SE baseline + KL DIV), we will only compute IN SAMPLE
    #CROSS ENTROPY as the baseline. this effectively gives us KL \approx 0. Thus, we get
    #residual structure = shannon entropy test + KL div test - shannon entropy baseline
      
    samples = np.array([np.random.choice(test_means, len(test_means), replace=True).mean()
               - np.random.choice(base_means, len(base_means), replace=True).mean()
               for _ in range(n)])

    #for the mean of 1000 random residual samples, how likely is this result (for a 95% confidence interval with 5%/2 tails)
    return samples.mean(), np.percentile(samples, [2.5, 97.5])

def decompose_by_factor(cell_results, baseline_results):
    #mean of our entropy by factors, if there is hidden structure or pattern, is it
    #driven by a specific factor?
    factors = ["event_type", "side", "price", "size", "time"]
    return {
        f: np.concatenate([r["per_token"][i::5] for r in cell_results]).mean()
           - np.concatenate([r["per_token"][i::5] for r in baseline_results]).mean()
        for i, f in enumerate(factors)
    }


def cross_reference_agents(cell_results, metadata, top_k_pct=5):
    #who of our agents is most surprising? who does the underlying
    #hidden structure / pattern belong to?
    
    all_nll = np.concatenate([r["per_event"] for r in cell_results])
    threshold = np.percentile(all_nll, 100 - top_k_pct)
    surprising = []
    for r, m in zip(cell_results, metadata):
        mask = r["per_event"] >= threshold
        n_events = len(r["per_event"])
        surprising.append(m["agent_class"].iloc[1:n_events + 1][mask])
    return pd.concat(surprising).value_counts(normalize=True)
