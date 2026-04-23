import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

# ===== device =====
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("using device:", device)

# ===== データ =====
def make_data(n=512, dim=32):
    x_scalar = (torch.rand(n, 1, device=device) * 2 * math.pi - math.pi) * 16
    x = x_scalar.repeat(1, dim)
    y = torch.sin(x_scalar)
    return x, y

def make_test_data(n=512, dim=32):
    x_scalar = (torch.rand(n, 1, device=device) * 2 * math.pi + math.pi) * 16
    x = x_scalar.repeat(1, dim)
    y = torch.sin(x_scalar)
    return x, y

# ===== Interp Layer =====
class InterpLayer(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.mlp_q = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim)
        )
        self.mlp_y = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim)
        )
        self.mlp_z = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim)
        )

        self.log_tau = nn.Parameter(torch.tensor(-3.0))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        q = self.mlp_q(x)
        y = self.mlp_y(x)
        z = self.mlp_z(x)

        tau = torch.exp(self.log_tau)

        q_exp = q.unsqueeze(-1)
        y_exp = y.unsqueeze(-2)

        w = torch.exp(- (q_exp - y_exp)**2 / (tau**2 + 1e-6))
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)

        z_exp = z.unsqueeze(-2)
        interp_vals = (w * z_exp).sum(dim=-1)

        # residual + normalization
        out = self.norm(x + interp_vals)
        return out

# ===== スタックモデル =====
class DeepInterpModel(nn.Module):
    def __init__(self, dim=256, hidden=512, depth=4):
        super().__init__()
        self.layers = nn.ModuleList([
            InterpLayer(dim, hidden) for _ in range(depth)
        ])
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

# ===== 学習 =====
dim = 32
model = DeepInterpModel(dim=dim, depth=4).to(device)
opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

for step in range(4000):
    x, y = make_data(256, dim)
    pred = model(x)
    loss = ((pred - y)**2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 10 == 0:
        print(f"step {step} loss {loss.item():.4f}")

# ===== 評価 =====
x_train, y_train = make_data(512, dim)
x_test, y_test = make_test_data(512, dim)

with torch.no_grad():
    pred_train = model(x_train)
    pred_test = model(x_test)

print("train loss:", ((pred_train - y_train)**2).mean().item())
print("test loss:", ((pred_test - y_test)**2).mean().item())

# ===== 可視化 =====
xs_scalar = torch.linspace(-math.pi, 3*math.pi*16, 1000).unsqueeze(1)
xs = xs_scalar.repeat(1, dim).to(device)

ys = torch.sin(xs_scalar)

with torch.no_grad():
    preds = model(xs).cpu()

plt.plot(xs_scalar.numpy(), ys.numpy(), label="true sin")
plt.plot(xs_scalar.numpy(), preds.numpy(), label="model")
plt.axvline(math.pi*16, linestyle="--", label="train boundary")
plt.legend()
plt.show()