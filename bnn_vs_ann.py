#!/usr/bin/env python3
"""BNN vs ANN Head-to-Head — Single CL Session

Compares a biological neural network (Cortical Labs CL SDK simulator) against
a conventional artificial neural network (NumPy MLP) across three experiments:

1. Learning Speed — 4-class pattern classification
2. Adaptation — shuffle class mappings, measure re-learning
3. Noise Robustness — increasing Gaussian noise, compare degradation
"""

import cl
import numpy as np
import time

NUM_PATTERNS = 4
NUM_CHANNELS = 8
TRIALS_PER_EPOCH = 16
MAX_EPOCHS = 15
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TARGET_ACC = 0.75
np.random.seed(42)


def make_patterns():
    pats = []
    for i in range(NUM_PATTERNS):
        p = np.random.uniform(0.05, 0.2, NUM_CHANNELS)
        hot = np.random.choice(NUM_CHANNELS, size=2, replace=False)
        p[hot] = np.random.uniform(0.7, 1.0, 2)
        pats.append(p)
    return np.array(pats)


def add_noise(p, n):
    return np.clip(p + np.random.normal(0, n, p.shape), 0, 1)


PATTERNS = make_patterns()


class MLP:
    def __init__(self, lr=0.1):
        self.W1 = np.random.randn(NUM_CHANNELS, 16) * 0.3
        self.b1 = np.zeros(16)
        self.W2 = np.random.randn(16, NUM_PATTERNS) * 0.3
        self.b2 = np.zeros(NUM_PATTERNS)
        self.lr = lr

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        z2 = self.a1 @ self.W2 + self.b2
        e = np.exp(z2 - z2.max())
        self.probs = e / e.sum()
        return self.probs

    def train(self, x, t):
        self.forward(x)
        oh = np.zeros(NUM_PATTERNS)
        oh[t] = 1
        dz2 = self.probs - oh
        self.W2 -= self.lr * np.outer(self.a1, dz2)
        self.b2 -= self.lr * dz2
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        self.W1 -= self.lr * np.outer(x, dz1)
        self.b1 -= self.lr * dz1
        return int(np.argmax(self.probs) == t)

    def predict(self, x):
        self.forward(x)
        return np.argmax(self.probs)

    def reset(self):
        self.__init__(self.lr)


class BNN:
    def __init__(self):
        self.W = np.random.uniform(0.1, 0.5, (64, NUM_PATTERNS))
        self.lr = 0.03

    def stim_and_read(self, neurons, pattern):
        for ch in range(NUM_CHANNELS):
            if pattern[ch] > 0.3:
                amp = max(20, int(pattern[ch] * 200) // 20 * 20)
                neurons.stim(ch, cl.StimDesign(amp, -1, amp, 1))
        sc = np.zeros(64)
        for tick in neurons.loop(ticks_per_second=1000, stop_after_ticks=30):
            for s in tick.analysis.spikes:
                if s.channel < 64:
                    sc[s.channel] += 1
        return sc

    def train(self, neurons, pattern, target):
        sc = self.stim_and_read(neurons, pattern)
        pred = np.argmax(sc @ self.W)
        ok = int(pred == target)
        if ok:
            self.W[sc > 0, target] += self.lr
        else:
            self.W[:, pred] -= self.lr * 0.3 * (sc / (sc.max() + 1e-8))
            self.W[sc > 0, target] += self.lr * 0.5
        self.W = np.clip(self.W, 0.01, 2.0)
        return ok

    def predict(self, neurons, pattern):
        sc = self.stim_and_read(neurons, pattern)
        return np.argmax(sc @ self.W)

    def reset(self):
        self.W = np.random.uniform(0.1, 0.5, (64, NUM_PATTERNS))


def main():
    print("=" * 60)
    print("BNN vs ANN HEAD-TO-HEAD")
    print("=" * 60)

    for i, p in enumerate(PATTERNS):
        print(f"  Class {i}: hot channels = {np.where(p > 0.5)[0].tolist()}")

    ann = MLP()
    bnn = BNN()

    with cl.open() as neurons:
        # EXP 1: Learning Speed
        print(f"\n--- EXPERIMENT 1: LEARNING SPEED ---")
        ann_lc, bnn_lc = [], []
        ann_t = bnn_t = None
        t0 = time.time()
        for epoch in range(MAX_EPOCHS):
            ac = bc = 0
            order = list(range(NUM_PATTERNS)) * (TRIALS_PER_EPOCH // NUM_PATTERNS)
            np.random.shuffle(order)
            for tc in order:
                p = add_noise(PATTERNS[tc], 0.05)
                ac += ann.train(p, tc)
                bc += bnn.train(neurons, p, tc)
            aa, ba = ac / TRIALS_PER_EPOCH, bc / TRIALS_PER_EPOCH
            ann_lc.append(aa)
            bnn_lc.append(ba)
            if aa >= TARGET_ACC and ann_t is None:
                ann_t = epoch + 1
            if ba >= TARGET_ACC and bnn_t is None:
                bnn_t = epoch + 1
            print(f"  Epoch {epoch+1:2d}: ANN={aa:.0%}  BNN={ba:.0%}")
            if ann_t and bnn_t and epoch >= max(ann_t, bnn_t) + 1:
                break
        print(f"  ({time.time()-t0:.0f}s)")

        # EXP 2: Adaptation
        print(f"\n--- EXPERIMENT 2: ADAPTATION ---")
        SHUFFLED = PATTERNS.copy()
        np.random.shuffle(SHUFFLED)
        ann_ac, bnn_ac = [], []
        ann_at = bnn_at = None
        for epoch in range(10):
            ac = bc = 0
            order = list(range(NUM_PATTERNS)) * (TRIALS_PER_EPOCH // NUM_PATTERNS)
            np.random.shuffle(order)
            for tc in order:
                p = add_noise(SHUFFLED[tc], 0.05)
                ac += ann.train(p, tc)
                bc += bnn.train(neurons, p, tc)
            aa, ba = ac / TRIALS_PER_EPOCH, bc / TRIALS_PER_EPOCH
            ann_ac.append(aa)
            bnn_ac.append(ba)
            if aa >= TARGET_ACC and ann_at is None:
                ann_at = epoch + 1
            if ba >= TARGET_ACC and bnn_at is None:
                bnn_at = epoch + 1
            print(f"  Epoch {epoch+1:2d}: ANN={aa:.0%}  BNN={ba:.0%}")

        # EXP 3: Noise Robustness
        print(f"\n--- EXPERIMENT 3: NOISE ROBUSTNESS ---")
        ann.reset()
        bnn.reset()
        for _ in range(8):
            order = list(range(NUM_PATTERNS)) * 4
            np.random.shuffle(order)
            for tc in order:
                ann.train(add_noise(PATTERNS[tc], 0.05), tc)
                bnn.train(neurons, add_noise(PATTERNS[tc], 0.05), tc)
        ann_n, bnn_n = [], []
        for noise in NOISE_LEVELS:
            ac = bc = 0
            for tc in list(range(NUM_PATTERNS)) * 4:
                p = add_noise(PATTERNS[tc], noise)
                ac += int(ann.predict(p) == tc)
                bc += int(bnn.predict(neurons, p) == tc)
            ann_n.append(ac / 16)
            bnn_n.append(bc / 16)
            print(f"  Noise {noise:.1f}: ANN={ac/16:.0%}  BNN={bc/16:.0%}")

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"""
+------------------------+-----------+-----------+
| Metric                 |    ANN    |    BNN    |
+------------------------+-----------+-----------+
| Epochs to {TARGET_ACC:.0%}          | {str(ann_t or 'N/A'):>9} | {str(bnn_t or 'N/A'):>9} |
| Final learn accuracy   | {ann_lc[-1]:>8.0%} | {bnn_lc[-1]:>8.0%} |
| Re-adapt epochs        | {str(ann_at or 'N/A'):>9} | {str(bnn_at or 'N/A'):>9} |
| Noise 0.3 accuracy     | {ann_n[3]:>8.0%} | {bnn_n[3]:>8.0%} |
| Noise 0.7 accuracy     | {ann_n[5]:>8.0%} | {bnn_n[5]:>8.0%} |
| Noise 1.0 accuracy     | {ann_n[6]:>8.0%} | {bnn_n[6]:>8.0%} |
+------------------------+-----------+-----------+
""")


if __name__ == "__main__":
    main()
