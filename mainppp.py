from datasets import load_dataset
import tqdm
import re
import string
import random
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


"""model = sklearn.linear_model.LinearRegression()

print(string.printable)

# Load a specific split
train = load_dataset("ronantakizawa/github-top-code", split="train")
test = load_dataset("ronantakizawa/github-top-code", split="test")

random.seed(0)

xs = []
ys = []
#used_chars = set()
for code in tqdm.tqdm(range(10000)):
    stri = list(re.sub('[^!-~\\s]|[　]', '', train[code]["content"]))
    s = random.randint(1, len(stri))
    s2 = (len(stri) - s) / len(stri)
    for j in random.choices(list(range(len(stri))), k=s):
        stri[j] = string.printable[random.randint(0, len(string.printable)-1)]
    x = np.array([stri.count(c) / len(stri) for c in string.printable])
    y = s2
    xs.append(x)
    ys.append(y)

model.fit(xs, ys)
g = model.predict(xs)
print(np.corrcoef(ys, g)[0, 1])
print(spearmanr(ys, g).statistic)

plt.scatter(ys, g, s=0.25)
plt.show()

print(model.coef_)

np.savez("coef.npz", coef=model.coef_, intercept=model.intercept_)
"""

## MANY FUNCTIONS

def mean(x, hidden=None):
    if(hidden is None):
        return (x, x)
    a = x * 0.1 + hidden * 0.9
    return (a, a)

def mean2(x, hidden=None):
    if(hidden is None):
        return (x, x)
    a = x * 0.01 + hidden * 0.99
    return (a, a)

def mean3(x, hidden=None):
    if(hidden is None):
        return (x, x)
    a = x * 0.001 + hidden * 0.999
    return (a, a)

def mean4(x, hidden=None):
    if(hidden is None):
        return (x, x)
    a = x * 0.5 + hidden * 0.5
    return (a, a)

def old(x, hidden=None):
    if(hidden is None):
        return (x, x)
    return (hidden, x)

def mean5(x, hidden=None):
    if(hidden is None):
        return (x, (x, 1))
    a = x * (1 / (hidden[1] + 1)) + hidden[0] * (hidden[1] / (hidden[1] + 1))
    return (a, (a, hidden[1] + 1))

def sum_(x, hidden=None):
    if(hidden is None):
        return (x, x)
    return (hidden + x, hidden + x)

def divstep(x, hidden=None):
    if(hidden is None):
        return (x, 1)
    return (x / (hidden + 1), hidden + 1)

def multstep(x, hidden=None):
    if(hidden is None):
        return (x, 1)
    return (x * (hidden + 1), hidden + 1)

def diff(x, hidden=None):
    if(hidden is None):
        return (x * 0, x)
    return (hidden - x, x)

def ada(x, hidden=None):
    if(hidden is None):
        return (x * 0, (x, x ** 2))
    a = hidden[0] * 0.9 + x * 0.1
    b = hidden[1] * 0.999 + (x ** 2) * 0.001
    return ((x - a) / np.sqrt(b + 1e-12), (a, b))

def ada2(x, hidden=None):
    if(hidden is None):
        return (x * 0, (x, x ** 2))
    a = hidden[0] * 0.9 + x * 0.1
    b = hidden[1] * 0.95 + (x ** 2) * 0.05
    return ((x - a) / np.sqrt(b + 1e-12), (a, b))

funcs_one_recurrent = [
    mean,
    mean2,
    mean3,
    mean4,
    old,
    mean5,
    sum_,
    divstep,
    multstep,
    diff,
    ada,
    ada2,
]

funcs_one_single = [
    lambda x: x,
    lambda x: x + 1,
    lambda x: x - 1,
    lambda x: x ** 2,
    lambda x: x * 2,
    lambda x: x / 2,
    lambda x: -x,
    lambda x: -x ** 2,
    lambda x: np.exp(-x ** 2),
    lambda x: x * 10,
    lambda x: x * 0.1,
    lambda x: np.max(x, 0),
    lambda x: np.sqrt(np.abs(x)),
    lambda x: np.mean(x) + x * 0,
    lambda x: np.std(x) + x * 0,
    lambda x: np.max(x) + x * 0,
    lambda x: np.min(x) + x * 0,
    lambda x: (x - np.mean(x)) / (np.std(x) + 1e-12),
    lambda x: np.concatenate((x[1:], x[:1])),
    lambda x: np.concatenate((x[:-1], x[:1])),
    lambda x: np.concatenate((x[1:], x[:1])) * 0.25 + x * 0.5 + np.concatenate((x[:-1], x[:1])) * 0.25,
    lambda x: ((np.concatenate((x[1:], x[:1])) - x) ** 2 + (x - np.concatenate((x[:-1], x[:1]))) ** 2) * 0.5,
    lambda x: np.concatenate((x[1:], x[:1])) - x,
    lambda x: np.concatenate((x[:-1], x[:1])) - x,
    lambda x: np.concatenate((x[2:], x[:2])),
    lambda x: np.concatenate((x[:-2], x[:2])),
    lambda x: np.concatenate((x[4:], x[:4])),
    lambda x: np.concatenate((x[:-4], x[:4])),
    lambda x: np.concatenate((x[8:], x[:8])),
    lambda x: np.concatenate((x[:-8], x[:8])),
    lambda x: np.concatenate((x[16:], x[:16])),
    lambda x: np.concatenate((x[:-16], x[:16])),
    lambda x: np.concatenate((x[32:], x[:32])),
    lambda x: np.concatenate((x[:-32], x[:32])),
    lambda x: np.concatenate((x[64:], x[:64])),
    lambda x: np.concatenate((x[:-64], x[:64])),
    lambda x: np.sort(x),
    lambda x: np.argsort(x),
    lambda x: np.fft.ifft(np.abs(np.fft.fft(x + 0j) ** 2) + 0j).real,
    lambda x: np.fft.ifft(np.fft.fft(x + 0j) * np.fft.fft(x + 0j).conj() + 0j).real,
    lambda x: np.fft.fft(x + 0j).real,
    lambda x: np.fft.fft(x + 0j).imag,
    lambda x: np.tanh(x),
    lambda x: np.tanh(x) * 0.5 + 0.5,
    lambda x: np.cumsum(x),
    lambda x: np.flip(x),
    lambda x: np.flip(np.cumsum(np.flip(x))),
    lambda x: np.cumsum(x) - np.flip(np.cumsum(np.flip(x))),
    lambda x: x[0] + x * 0,
    lambda x: x[1] + x * 0,
    lambda x: x[2] + x * 0,
    lambda x: x[4] + x * 0,
    lambda x: x[8] + x * 0,
    lambda x: np.sign(x) * np.abs(x) ** (1/3),
]

def mean_weighted(x, y, hidden=None):
    if(hidden is None):
        return (x, x)
    weight = np.tanh(y) * 0.5 + 0.5
    a = x * (1 - weight) + hidden * weight
    return (a, a)

def mean_weighted2(x, y, hidden=None):
    if(hidden is None):
        return (x, x)
    weight = np.tanh(np.mean(y)) * 0.5 + 0.5
    a = x * (1 - weight) + hidden * weight
    return (a, a)

def mean_weighted3(x, y, hidden=None):
    if(hidden is None):
        return (x, x)
    weight = np.mean(np.tanh(y)) * 0.5 + 0.5
    a = x * (1 - weight) + hidden * weight
    return (a, a)

funcs_two_recurrent = [
    mean_weighted,
    mean_weighted2,
    mean_weighted3,
]

funcs_two_single = [
    lambda x, y: x + y,
    lambda x, y: x * 2 - y,
    lambda x, y: x * 3 - y * 2,
    lambda x, y: x * 4 - y * 3,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / np.sqrt(y ** 2 + 1e-12),
    lambda x, y: np.fft.ifft(np.fft.fft(x + 0j) * np.fft.fft(y + 0j)).real,
    lambda x, y: np.fft.ifft(np.fft.fft(x + 0j) * np.fft.fft(y + 0j).conj()).real,
    lambda x, y: x * (np.tanh(y) + 1),
    lambda x, y: x * np.tanh(y),
    lambda x, y: (x - y) ** 2,
    lambda x, y: (x + y) / 2,
    lambda x, y: x ** 2 + y ** 2,
    lambda x, y: (x ** 2 + y ** 2) ** 0.5,
    lambda x, y: np.sin(x * y),
    lambda x, y: np.take(x, np.argsort(y)),
    lambda x, y: np.take(x, np.argsort(np.argsort(y))),
    lambda x, y: np.maximum(x, y),
    lambda x, y: np.minimum(x, y),
    lambda x, y: np.exp(-(x * y) ** 2),
    lambda x, y: np.fft.ifft(np.fft.fft(x + 0j) ** 2 / (np.fft.fft(y + 0j) + 1e-12)).real,
]

def rls(x, y, z, hidden=None):
    if(hidden is None):
        W = np.zeros((len(x), len(x)))
        P = 1e3 * np.eye(len(x))
    else:
        W = hidden[0]
        P = hidden[1]
    x = x.reshape(-1, 1)  # (d,1)
    y = y.reshape(-1, 1)  # (m,1)
    Px = P @ x
    denom = 0.999 + (x.T @ Px)[0, 0]
    K = Px / denom  # (d,1)
    y_pred = W @ x
    e = y - y_pred  # (m,1)
    W += e @ K.T
    P = (P - K @ x.T @ P) / 0.999
    return (W @ z, (W, P))

def rls2(x, y, z, hidden=None):
    if(hidden is None):
        W = np.zeros((len(x), len(x)))
        P = 1e2 * np.eye(len(x))
    else:
        W = hidden[0]
        P = hidden[1]
    x = x.reshape(-1, 1)  # (d,1)
    y = y.reshape(-1, 1)  # (m,1)
    Px = P @ x
    denom = 0.95 + (x.T @ Px)[0, 0]
    K = Px / denom  # (d,1)
    y_pred = W @ x
    e = y - y_pred  # (m,1)
    W += e @ K.T
    P = (P - K @ x.T @ P) / 0.95
    return (W @ z, (W, P))

def fftattn(x, y, z, hidden=None):
    return (np.fft.ifft(np.fft.fft(x) * np.fft.fft(y) / (np.fft.fft(z) + 1e-12)).real, None)

def interp(x, y, z, hidden=None):
    s = np.argsort(x)
    x, y = x[s], y[s]
    out = np.interp(z, x, y)
    return (out, None)

def sortten(x, y, z, hidden=None):
    return (np.take(np.take(x, np.argsort(y.flatten())), np.argsort(np.argsort(z.flatten()))).reshape(x.shape), None)

def fftattn_mean(x, y, z, hidden=None):
    if(hidden is None):
        filt = np.fft.fft(y) / (np.fft.fft(z) + 1e-12)
        return (np.fft.ifft(np.fft.fft(x) * filt).real, filt)
    filt = np.fft.fft(y) / (np.fft.fft(z) + 1e-12)
    return (np.fft.ifft(np.fft.fft(x) * (filt * 0.05 + hidden * 0.95)).real, (filt * 0.05 + hidden * 0.95))

def fftattn_mean2(x, y, z, hidden=None):
    if(hidden is None):
        return (np.fft.ifft(np.fft.fft(x) * np.fft.fft(y) / (np.fft.fft(z) + 1e-12)).real, (y, z))
    return (np.fft.ifft(np.fft.fft(x) * np.fft.fft(y * 0.1 + hidden[0] * 0.9) / (np.fft.fft(z * 0.05 + hidden[1] * 0.95) + 1e-12)).real, (y * 0.1 + hidden[0] * 0.9, z * 0.05 + hidden[1] * 0.95))

def swish(x): return x / (1 + np.exp(-x))
def dswish(x): s = 1/(1+np.exp(-x)); return s + x*s*(1-s)

def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (np.linalg.norm(G) + 1e-7)
    if X.shape[0] > X.shape[1]: X, t = X.T, True
    else: t = False
    for _ in range(steps):
        A = X @ X.T
        X = a*X + b*A@X + c*(A@A)@X
    return X.T if t else X

def muon_step(G, m, lr=1e-2, mu=0.95):
    m = mu*m + G
    G_nes = mu*m + G
    O = newton_schulz(G_nes)
    scale = max(1, G_nes.shape[0]/G_nes.shape[1])**0.5
    return -lr*scale*O, m

def forward(x, params):
    a = x
    for i,(W,b) in enumerate(params):
        z = W@a+b; a = swish(z) if i<len(params)-1 else z
    return a

def backward(x, y, params):
    acts, zs, a = [x], [], x
    for i,(W,b) in enumerate(params):
        z = W@a+b; zs.append(z)
        a = swish(z) if i<len(params)-1 else z; acts.append(a)
    delta = acts[-1]-y
    gWs, gbs = [], []
    for i in reversed(range(len(params))):
        gWs.insert(0, np.outer(delta, acts[i]))
        gbs.insert(0, delta)
        if i>0: delta = (params[i][0].T@delta)*dswish(zs[i-1])
    return gWs, gbs

def rls_mlp(x, y, z, hidden=None, n_layers=4):
    d = len(x)
    if hidden is None:
        params  = [(np.eye(d), np.zeros(d))] * n_layers  # 単位行列で初期化
        moments = [np.zeros((d,d))] * n_layers
    else:
        params, moments = hidden
    gWs, gbs = backward(x, y, params)
    new_params, new_moments = [], []
    for (W,b),mW,gW,gb in zip(params,moments,gWs,gbs):
        dW,mW = muon_step(gW,mW)
        new_params.append((W+dW, b-1e-2*gb))
        new_moments.append(mW)
    return forward(z, new_params), (new_params, new_moments)

def rls_mlp2(x, y, z, hidden=None, n_layers=4):
    d = len(x)
    if hidden is None:
        params  = [(np.eye(d), np.zeros(d))] * n_layers  # 単位行列で初期化
        moments = [None] * n_layers
        moments2 = [None] * n_layers
    else:
        params, moments, moments2 = hidden
    gWs, gbs = backward(x, y, params)
    new_params, new_moments, new_moments2 = [], [], []
    for (W,b),mW,mW2,gW,gb in zip(params,moments,moments2,gWs,gbs):
        if mW is None: mW = (gW, gb)
        if mW2 is None: mW2 = (gW**2, gb**2)
        mW = (mW[0]*0.9 + gW*0.1, mW[1]*0.9 + gb*0.1)
        mW2 = (mW2[0]*0.99 + gW**2*0.01, mW2[1]*0.99 + gb**2*0.01)
        wt = mW[0] / (np.sqrt(mW2[0]) + 1e-12)
        wb = mW[1] / (np.sqrt(mW2[1]) + 1e-12)
        new_params.append((W-1e-2*wt, b-1e-2*wb))
        new_moments.append(mW)
        new_moments2.append(mW2)
    return forward(z, new_params), (new_params, new_moments, new_moments2)

def causal_swa(x, y, z, hidden=None, window=64):
    n = len(x)
    if hidden is None:
        K_buf = np.zeros((window, n))
        V_buf = np.zeros((window, n))
        t = 0
    else:
        K_buf, V_buf, t = hidden
    idx = t % window
    K_buf = K_buf.copy()
    V_buf = V_buf.copy()
    K_buf[idx] = y 
    V_buf[idx] = z
    w = min(t + 1, window)
    Q = x 
    scores = np.array([Q @ K_buf[(t - w + 1 + i) % window] for i in range(w)]) 
    scores = scores / np.sqrt(n)
    scores = scores - scores.max()
    weights = np.exp(scores)
    weights = weights / (weights.sum() + 1e-12)
    out = sum(
        weights[i] * V_buf[(t - w + 1 + i) % window]
        for i in range(w)
    )
    return (out, (K_buf, V_buf, t + 1))

funcs_three_recurrent = [
    rls,
    rls2,
    fftattn,
    interp,
    sortten,
    fftattn_mean,
    causal_swa,
    rls_mlp,
    rls_mlp2,
]

funcs_three_single = [
    lambda x, y, z: x + y + z,
    lambda x, y, z: x + y - z,
    lambda x, y, z: x * y * z,
    lambda x, y, z: x * y / (np.sqrt(z ** 2) + 1e-12),
    lambda x, y, z: x / np.sqrt(y ** 2 + z ** 2 + 1e-12),
    lambda x, y, z: np.fft.ifft(np.fft.fft(x + 0j) * np.fft.fft(y + 0j) / (np.fft.fft(z + 0j) + 1e-12)).real,
    lambda x, y, z: np.fft.ifft(np.fft.fft(x + 0j) * (np.fft.fft(y + 0j) / (np.fft.fft(z + 0j) + 1e-12)).conj()).real,
    lambda x, y, z: x * (np.tanh(y) + 1) * (np.tanh(z) + 1),
    lambda x, y, z: x * np.tanh(y) * np.tanh(z),
    lambda x, y, z: (x - y) ** 2 + (y - z) ** 2 + (z - x) ** 2,
    lambda x, y, z: (x + y + z) / 3,
    lambda x, y, z: x ** 2 + y ** 2 + z ** 2,
    lambda x, y, z: (x ** 2 + y ** 2 + z ** 2) ** 0.5,
    lambda x, y, z: (x ** 2 + y ** 2 + z ** 2) / 3,
    lambda x, y, z: ((x ** 2 + y ** 2 + z ** 2) / 3) ** 0.5,
    lambda x, y, z: ((x - y) ** 2 + (y - z) ** 2 + (z - x) ** 2) ** 0.5,
    lambda x, y, z: (((x - y) ** 2 + (y - z) ** 2 + (z - x) ** 2) / 3) ** 0.5,
    lambda x, y, z: ((x - y) ** 2 + (y - z) ** 2 + (z - x) ** 2) / 3,
    lambda x, y, z: np.sign(x * y * z) * np.abs(x * y * z) ** (1/3),
    lambda x, y, z: np.sign((x - y) * (y - z) * (z - x)) * np.abs((x - y) * (y - z) * (z - x)) ** (1/3),
    lambda x, y, z: np.sin(x * y * z),
    lambda x, y, z: np.take(np.take(x, np.argsort(y)), np.argsort(np.argsort(z))),
    lambda x, y, z: np.take(np.take(x, np.argsort(y)), np.argsort(z)),
    lambda x, y, z: np.maximum(np.minimum(x, y), z),
    lambda x, y, z: np.exp(-(x * (y @ z) / len(x)) ** 2),
    lambda x, y, z: np.exp(-(x * np.std(y) + np.mean(z)) ** 2),
    lambda x, y, z: np.tanh(x) * np.std(y) + np.mean(z),
    lambda x, y, z: np.tanh(x) * np.mean(y) + np.mean(z),
    lambda x, y, z: x * np.std(y) + np.mean(z),
    lambda x, y, z: x * np.mean(y) + np.mean(z),
    lambda x, y, z: x * 0.5 - y * 1 + z * 2.5,
    lambda x, y, z: x * 1 - y * 2 + z * 4,
    lambda x, y, z: x * 2 - y * 3 + z * 6,
    lambda x, y, z: x * 3 - y * 4 + z * 8,
    lambda x, y, z: x * 4 - y * 5 + z * 10,
    lambda x, y, z: x * 5 - y * 6 + z * 12,
]

import time

for i in range(len(funcs_one_single)):
    randt = np.random.normal(0, 1, (128))
    t = time.perf_counter()
    for j in range(2048):
        randt2 = funcs_one_single[i](randt)
    print(time.perf_counter() - t)

for i in range(len(funcs_one_recurrent)):
    randt = np.random.normal(0, 1, (128))
    randt2 = None
    t = time.perf_counter()
    for j in range(2048):
        _, randt2 = funcs_one_recurrent[i](randt, randt2)
    print(time.perf_counter() - t)

for i in range(len(funcs_two_single)):
    randt = np.random.normal(0, 1, (128))
    t = time.perf_counter()
    for j in range(2048):
        randt2 = funcs_two_single[i](randt, randt)
    print(time.perf_counter() - t)

for i in range(len(funcs_two_recurrent)):
    randt = np.random.normal(0, 1, (128))
    randt2 = None
    t = time.perf_counter()
    for j in range(2048):
        _, randt2 = funcs_two_recurrent[i](randt, randt, randt2)
    print(time.perf_counter() - t)

for i in range(len(funcs_three_single)):
    randt = np.random.normal(0, 1, (128))
    t = time.perf_counter()
    for j in range(2048):
        randt2 = funcs_three_single[i](randt, randt, randt)
    print(time.perf_counter() - t)

for i in range(len(funcs_three_recurrent)):
    randt = np.random.normal(0, 1, (128))
    randt2 = None
    t = time.perf_counter()
    for j in range(2048):
        _, randt2 = funcs_three_recurrent[i](randt, randt, randt, randt2)
    print(time.perf_counter() - t)

