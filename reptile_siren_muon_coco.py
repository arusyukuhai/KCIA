"""
Reptile + SIREN + Muon  ―  COCO Implicit Neural Representation Meta-Learning
=============================================================================
アルゴリズム概要 (Nichol et al., 2018):
  1. メタパラメータ θ をコピーして φ を作る
  2. φ を support set で k ステップ適応
  3. θ ← θ + ε · (φ − θ)    ← Reptile 更新（2次微分不要）

MAML との違い:
  - query set / outer backward が不要 → メモリ効率◎
  - inner loop の勾配グラフを保持しなくてよい (create_graph=False)
  - 外部オプティマイザは「補間」なので Adam ではなく SGD 的に動く
    （ここでは外部も Muon を使い θ に直接補間する形で実装）

タスク定義:
  COCO 各画像 = 1 INR タスク
  SIREN : (x,y) ∈ [-1,1]² → (R,G,B) ∈ [-1,1]³

ディレクトリ構成:
  coco/images/train2017/  ← COCO train 画像
  coco/images/val2017/    ← COCO val 画像

依存:
  pip install torch torchvision Pillow matplotlib tqdm
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════
#  1.  SIREN
# ══════════════════════════════════════════════════════════════

class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0  = omega_0
        self.is_first = is_first
        self.linear   = nn.Linear(in_features, out_features)
        """self._init_weights()

    def _init_weights(self) -> None:
        fan_in = self.linear.in_features
        bound  = (1.0 / fan_in) if self.is_first \
                 else math.sqrt(6.0 / fan_in) / self.omega_0
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)"""

    def forward(self, x: Tensor) -> Tensor:
        y = self.linear(x)
        return torch.concatenate((torch.sin(self.omega_0 * x[..., ::2]) * torch.nn.functional.softplus(self.omega_0 * y[..., 1::2]) / self.omega_0, torch.cos(self.omega_0 * x[..., ::2]) * torch.nn.functional.softplus(self.omega_0 * y[..., 1::2]) / self.omega_0), dim=-1)


class SIREN(nn.Module):
    """
    (x, y) ∈ [-1,1]² → (R, G, B) ∈ [-1,1]³
    """
    def __init__(self,
                 in_features:     int   = 2,
                 hidden_features: int   = 256,
                 hidden_layers:   int   = 4,
                 out_features:    int   = 3,
                 first_omega_0:   float = 30.0,
                 hidden_omega_0:  float = 30.0):
        super().__init__()
        layers: List[nn.Module] = [
            SineLayer(in_features, hidden_features,
                      omega_0=first_omega_0, is_first=True)
        ]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features,
                                    omega_0=hidden_omega_0, is_first=False))
        final = nn.Linear(hidden_features, out_features)
        """bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
        nn.init.uniform_(final.weight, -bound, bound)
        nn.init.zeros_(final.bias)"""
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, coords: Tensor) -> Tensor:
        return self.net(coords)


# ══════════════════════════════════════════════════════════════
#  2.  Muon Optimizer
# ══════════════════════════════════════════════════════════════

def _newtonschulz5(G: Tensor, steps: int = 6, eps: float = 1e-7) -> Tensor:
    """直交極因子を Newton-Schulz 反復で計算。"""
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = (G.bfloat16() if G.is_cuda else G.float())
    X = X / (X.norm() + eps)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    2-D weight → Newton-Schulz 直交化 + Nesterov momentum
    1-D bias   → Nesterov SGD
    """
    def __init__(self, params,
                 lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 6,
                 weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                     nesterov=nesterov, ns_steps=ns_steps,
                                     weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu   = group['lr'], group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            wd       = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.clone()
                if wd:
                    g.add_(p, alpha=wd)
                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(p)
                buf: Tensor = state['buf']
                buf.mul_(mu).add_(g)
                g = g.add(buf, alpha=mu) if nesterov else buf.clone()
                if g.ndim >= 2:
                    g = _newtonschulz5(g, steps=ns_steps)
                    g = g * max(g.size(-2), g.size(-1)) ** 0.5
                p.add_(g, alpha=-lr)
        return loss


# ══════════════════════════════════════════════════════════════
#  3.  COCO INR Dataset
# ══════════════════════════════════════════════════════════════

class CocoINRDataset(Dataset):
    """
    COCO 画像を INR タスクとして返す。
    1 sample = 1 画像の全ピクセル (coord, rgb)。
    Reptile では support のみ使うので n_query は可視化専用。
    """
    def __init__(self,
                 image_dir:  str,
                 img_size:   Tuple[int, int] = (64, 64),
                 n_support:  int             = 1024,
                 max_images: Optional[int]   = None):
        self.img_size  = img_size
        self.n_support = n_support
        exts  = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        paths = sorted(p for p in Path(image_dir).iterdir()
                       if p.suffix.lower() in exts)
        if max_images:
            paths = paths[:max_images]
        if not paths:
            raise FileNotFoundError(f"No images in {image_dir}")
        self.paths = paths

        H, W = img_size
        ys = torch.linspace(-1, 1, H)
        xs = torch.linspace(-1, 1, W)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        self.all_coords: Tensor = torch.stack(
            [gx.reshape(-1), gy.reshape(-1)], dim=-1
        )  # (H*W, 2)

        self.to_tensor = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert('RGB')
        rgb = self.to_tensor(img).permute(1, 2, 0).reshape(-1, 3)  # (N,3)

        perm  = torch.randperm(rgb.size(0))
        s_idx = perm[:self.n_support]

        return {
            'support_coords': self.all_coords[s_idx],   # (n_support, 2)
            'support_rgb':    rgb[s_idx],                # (n_support, 3)
            'full_coords':    self.all_coords,           # (N, 2)  可視化用
            'full_rgb':       rgb,                       # (N, 3)  可視化用
            'path':           str(self.paths[idx]),
        }


def coco_collate(batch):
    out = {
        'support_coords': torch.stack([b['support_coords'] for b in batch]),
        'support_rgb':    torch.stack([b['support_rgb']    for b in batch]),
        'full_coords':    [b['full_coords'] for b in batch],
        'full_rgb':       [b['full_rgb']    for b in batch],
        'paths':          [b['path']        for b in batch],
    }
    return out


# ══════════════════════════════════════════════════════════════
#  4.  Reptile Inner Loop
# ══════════════════════════════════════════════════════════════

def reptile_inner_loop(
    meta_model:     SIREN,
    support_coords: Tensor,          # (n_support, 2)
    support_rgb:    Tensor,          # (n_support, 3)
    inner_lr:       float = 5e-3,
    n_inner_steps:  int   = 8,
    use_muon:       bool  = True,
) -> List[Tensor]:
    """
    φ = θ を k ステップ更新し、更新後パラメータのリストを返す。
    Reptile では create_graph 不要（二次微分なし）。
    """
    fast = copy.deepcopy(meta_model)
    fast.train()

    if use_muon:
        opt = Muon(fast.parameters(), lr=inner_lr,
                   momentum=0.9, nesterov=True, ns_steps=5)
    else:
        opt = torch.optim.SGD(fast.parameters(), lr=inner_lr,
                              momentum=0.9, nesterov=True)

    for _ in range(n_inner_steps):
        pred = fast(support_coords)
        loss = F.mse_loss(pred, support_rgb)
        opt.zero_grad()
        loss.backward()          # create_graph 不要
        opt.step()

    # 適応後パラメータを返す（メタ更新で θ との差分を使う）
    return [p.detach().clone() for p in fast.parameters()]


# ══════════════════════════════════════════════════════════════
#  5.  Reptile Meta-Update
# ══════════════════════════════════════════════════════════════

def reptile_meta_update(
    meta_model:    SIREN,
    adapted_params_list: List[List[Tensor]],
    meta_lr:       float = 0.1,
) -> None:
    """
    θ ← θ + meta_lr · mean(φ_i − θ)
         = (1 − meta_lr) · θ + meta_lr · mean(φ_i)

    adapted_params_list: タスクごとの適応後パラメータリスト
    """
    # adapted_params_list: List[List[Tensor]]
    #   外側のインデックス = タスク i
    #   内側のインデックス = パラメータ j
    # → パラメータ j ごとにタスク間で平均を取る
    n_tasks = len(adapted_params_list)
    with torch.no_grad():
        for j, meta_p in enumerate(meta_model.parameters()):
            # j 番目パラメータを全タスクからスタック → (n_tasks, *shape)
            stacked = torch.stack([adapted_params_list[i][j]
                                   for i in range(n_tasks)], dim=0)
            mean_adapted = stacked.mean(dim=0)          # (*shape)
            meta_p.add_(mean_adapted - meta_p, alpha=meta_lr)


# ══════════════════════════════════════════════════════════════
#  6.  Meta-Training Loop
# ══════════════════════════════════════════════════════════════

def meta_train(
    meta_model:    SIREN,
    train_loader:  DataLoader,
    n_epochs:      int   = 10,
    inner_lr:      float = 5e-3,
    n_inner_steps: int   = 8,
    meta_lr:       float = 0.1,
    use_muon_inner: bool = True,
    device:        str   = 'cpu',
    save_dir:      str   = 'checkpoints',
    log_every:     int   = 250,
) -> List[float]:
    """
    Reptile 外部ループ。
    外部には明示的なオプティマイザが不要。
    meta_lr は補間係数 ε ∈ (0, 1]。
    """
    os.makedirs(save_dir, exist_ok=True)
    meta_model.to(device)
    all_losses: List[float] = []
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        epoch_losses: List[float] = []
        pbar = tqdm(train_loader,
                    desc=f'Epoch {epoch}/{n_epochs}',
                    dynamic_ncols=True)

        for batch in pbar:
            sx = batch['support_coords'].to(device)  # (B, n_support, 2)
            sy = batch['support_rgb'].to(device)      # (B, n_support, 3)
            B  = sx.size(0)

            # ── inner loop: タスクごとに適応 ──
            adapted_list: List[List[Tensor]] = []
            support_losses: List[float]      = []

            for i in range(B):
                adapted = reptile_inner_loop(
                    meta_model,
                    sx[i], sy[i],
                    inner_lr      = inner_lr,
                    n_inner_steps = n_inner_steps,
                    use_muon      = use_muon_inner,
                )
                adapted_list.append(adapted)

                # ログ用: 適応後 support loss を計算
                with torch.no_grad():
                    fast_params = adapted
                    # forward pass with adapted params（計測のみ）
                    fast_tmp = copy.deepcopy(meta_model)
                    for p, fp in zip(fast_tmp.parameters(), fast_params):
                        p.data.copy_(fp)
                    pred = fast_tmp(sx[i])
                    support_losses.append(F.mse_loss(pred, sy[i]).item())

            # ── outer update: Reptile 補間 ──
            reptile_meta_update(meta_model, adapted_list, meta_lr=meta_lr)

            avg_loss = sum(support_losses) / len(support_losses)
            epoch_losses.append(avg_loss)
            all_losses.append(avg_loss)
            global_step += 1

            pbar.set_postfix({'support_loss': f'{avg_loss:.5f}'})

            if global_step % log_every == 0:
                recent = all_losses[-log_every:]
                tqdm.write(
                    f'[step {global_step:6d}] '
                    f'avg_support_loss = {sum(recent)/len(recent):.6f}'
                )
                torch.save({
                    'epoch':       epoch,
                    'global_step': global_step,
                    'model_state': meta_model.state_dict(),
                    'all_losses':  all_losses,
                    'cfg': dict(inner_lr=inner_lr, n_inner_steps=n_inner_steps,
                                meta_lr=meta_lr),
                }, os.path.join(save_dir, f'ckpt_epoch{epoch:03d}.pt'))

        avg_ep = sum(epoch_losses) / max(len(epoch_losses), 1)
        print(f'\nEpoch {epoch} done.  avg_support_loss = {avg_ep:.6f}')

    return all_losses


# ══════════════════════════════════════════════════════════════
#  7.  Evaluation & Visualization
# ══════════════════════════════════════════════════════════════

def psnr(pred: Tensor, target: Tensor) -> float:
    mse = F.mse_loss(pred.clamp(-1, 1), target).item()
    return 10 * math.log10(4.0 / (mse + 1e-8))


def evaluate(
    meta_model:    SIREN,
    val_loader:    DataLoader,
    inner_lr:      float = 5e-3,
    n_inner_steps: int   = 20,
    use_muon:      bool  = True,
    device:        str   = 'cpu',
    n_vis:         int   = 4,
    save_dir:      str   = 'eval_outputs',
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    meta_model.eval()

    psnr_before_list: List[float] = []
    psnr_after_list:  List[float] = []
    vis_count = 0

    for batch in tqdm(val_loader, desc='Evaluating', dynamic_ncols=True):
        sx = batch['support_coords'].to(device)
        sy = batch['support_rgb'].to(device)
        B  = sx.size(0)

        for i in range(B):
            fc = batch['full_coords'][i].to(device)
            fr = batch['full_rgb'][i].to(device)

            # 適応前
            with torch.no_grad():
                pb = psnr(meta_model(fc), fr)
            psnr_before_list.append(pb)

            # Reptile 適応（eval 用: より多くのステップ）
            adapted = reptile_inner_loop(
                meta_model, sx[i], sy[i],
                inner_lr=inner_lr, n_inner_steps=n_inner_steps,
                use_muon=use_muon,
            )
            fast = copy.deepcopy(meta_model)
            with torch.no_grad():
                for p, fp in zip(fast.parameters(), adapted):
                    p.data.copy_(fp)
                pred_after = fast(fc)
            pa = psnr(pred_after, fr)
            psnr_after_list.append(pa)

            # 可視化
            if vis_count < n_vis:
                H, W = val_loader.dataset.img_size

                def to_img(t: Tensor):
                    import numpy as np
                    arr = t.reshape(H, W, 3).cpu().numpy()
                    return (arr * 0.5 + 0.5).clip(0, 1)

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(to_img(fr))
                axes[0].set_title('Ground Truth')
                axes[1].imshow(to_img(meta_model(fc).clamp(-1,1).detach()))
                axes[1].set_title(f'Before adaptation\nPSNR={pb:.2f} dB')
                axes[2].imshow(to_img(pred_after.clamp(-1,1)))
                axes[2].set_title(
                    f'After {n_inner_steps} steps\nPSNR={pa:.2f} dB'
                )
                for ax in axes:
                    ax.axis('off')
                plt.suptitle('Reptile + SIREN + Muon', fontsize=13)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f'recon_{vis_count:03d}.png'),
                    dpi=150
                )
                plt.close()
                vis_count += 1

    result = {
        'psnr_before': sum(psnr_before_list) / len(psnr_before_list),
        'psnr_after':  sum(psnr_after_list)  / len(psnr_after_list),
        'n_samples':   len(psnr_before_list),
    }
    print(
        f"\n[Eval] n={result['n_samples']} | "
        f"PSNR before={result['psnr_before']:.2f} dB | "
        f"PSNR after ={result['psnr_after']:.2f} dB"
    )
    return result


def plot_losses(losses: List[float], save_path: str = 'training_curve.png'):
    w  = min(200, max(len(losses) // 10, 1))
    ma = [sum(losses[max(0,i-w):i+1]) / min(i+1, w) for i in range(len(losses))]
    plt.figure(figsize=(9, 4))
    plt.plot(losses, lw=0.6, alpha=0.45, color='steelblue', label='step loss')
    plt.plot(ma,     lw=2.0,             color='tomato',    label=f'moving avg ({w})')
    plt.xlabel('Meta-step')
    plt.ylabel('Support loss (MSE, after inner loop)')
    plt.title('Reptile + SIREN + Muon  —  COCO INR meta-training')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f'Saved → {save_path}')


# ══════════════════════════════════════════════════════════════
#  8.  CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Reptile + SIREN + Muon on COCO INR tasks'
    )
    # データ
    p.add_argument('--train_dir',   default='coco/images/val2017')
    p.add_argument('--val_dir',     default='coco/images/val2017')
    p.add_argument('--img_size',    type=int, nargs=2, default=[64, 64],
                   metavar=('H', 'W'))
    p.add_argument('--n_support',   type=int, default=1024,
                   help='Support pixels per task')
    p.add_argument('--max_train',   type=int, default=None)
    p.add_argument('--max_val',     type=int, default=200)

    # モデル
    p.add_argument('--hidden_features', type=int,   default=256)
    p.add_argument('--hidden_layers',   type=int,   default=4)
    p.add_argument('--first_omega',     type=float, default=30.0)
    p.add_argument('--hidden_omega',    type=float, default=30.0)

    # 学習
    p.add_argument('--n_epochs',      type=int,   default=10)
    p.add_argument('--meta_batch',    type=int,   default=4,
                   help='Tasks per Reptile update')
    p.add_argument('--meta_lr',       type=float, default=0.1,
                   help='Reptile interpolation coefficient ε')
    p.add_argument('--inner_lr',      type=float, default=5e-2)
    p.add_argument('--n_inner_steps', type=int,   default=8,
                   help='Inner steps k for Reptile')
    p.add_argument('--no_muon_inner', action='store_true',
                   help='Use SGD instead of Muon in inner loop')
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--seed',          type=int,   default=42)

    # 出力
    p.add_argument('--save_dir',         default='./')
    p.add_argument('--eval_dir',         default='eval_outputs')
    p.add_argument('--resume',           default=None)
    p.add_argument('--eval_only',        action='store_true')
    p.add_argument('--eval_inner_steps', type=int, default=20)

    return p.parse_args()


def main():
    args   = parse_args()
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print('=' * 64)
    print(f'  Reptile + SIREN + Muon  |  device = {device}')
    print(f'  img_size      = {args.img_size}')
    print(f'  hidden        = {args.hidden_features} × {args.hidden_layers}')
    print(f'  inner_lr      = {args.inner_lr}  k = {args.n_inner_steps}')
    print(f'  meta_lr (ε)   = {args.meta_lr}   batch = {args.meta_batch}')
    print(f'  muon_inner    = {not args.no_muon_inner}')
    print('=' * 64)

    img_size = tuple(args.img_size)

    train_ds = CocoINRDataset(args.train_dir, img_size,
                               args.n_support, args.max_train)
    val_ds   = CocoINRDataset(args.val_dir,   img_size,
                               args.n_support, args.max_val)
    print(f'Train: {len(train_ds)} images  |  Val: {len(val_ds)} images')

    train_loader = DataLoader(
        train_ds, batch_size=args.meta_batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device=='cuda'),
        collate_fn=coco_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device=='cuda'),
        collate_fn=coco_collate,
    )

    meta_model = SIREN(
        in_features     = 2,
        hidden_features = args.hidden_features,
        hidden_layers   = args.hidden_layers,
        out_features    = 3,
        first_omega_0   = args.first_omega,
        hidden_omega_0  = args.hidden_omega,
    ).to(device)
    print(f'Params: {sum(p.numel() for p in meta_model.parameters()):,}')

    start_losses: List[float] = []
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        meta_model.load_state_dict(ckpt['model_state'])
        start_losses = ckpt.get('all_losses', [])
        print(f'Resumed from {args.resume}')

    if args.eval_only:
        evaluate(meta_model, val_loader,
                 inner_lr=args.inner_lr,
                 n_inner_steps=args.eval_inner_steps,
                 use_muon=not args.no_muon_inner,
                 device=device, save_dir=args.eval_dir)
        return

    losses = meta_train(
        meta_model     = meta_model,
        train_loader   = train_loader,
        n_epochs       = args.n_epochs,
        inner_lr       = args.inner_lr,
        n_inner_steps  = args.n_inner_steps,
        meta_lr        = args.meta_lr,
        use_muon_inner = not args.no_muon_inner,
        device         = device,
        save_dir       = args.save_dir,
        log_every      = 250,
    )

    plot_losses(start_losses + losses)

    evaluate(meta_model, val_loader,
             inner_lr=args.inner_lr,
             n_inner_steps=args.eval_inner_steps,
             use_muon=not args.no_muon_inner,
             device=device, save_dir=args.eval_dir)


if __name__ == '__main__':
    main()
