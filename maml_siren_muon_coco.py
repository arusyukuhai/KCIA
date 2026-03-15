"""
MAML + SIREN + Muon  ―  COCO Implicit Neural Representation Meta-Learning
=========================================================================
タスク定義:
  COCO の各画像を 1 つの INR (Implicit Neural Representation) タスクとして扱う。
  SIREN は (x, y) ∈ [-1,1]² → (R, G, B) ∈ [-1,1]³ を学習する座標→画素写像。
  MAML により「任意の新画像に数ステップで適応できるメタ初期重み」を獲得する。

ディレクトリ構成 (事前準備):
  coco/
  ├── images/
  │   ├── train2017/   ← COCO train 画像
  │   └── val2017/     ← COCO val 画像
  └── annotations/
      ├── instances_train2017.json
      └── instances_val2017.json

  wget http://images.cocodataset.org/zips/train2017.zip
  wget http://images.cocodataset.org/zips/val2017.zip
  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

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
#  1.  SIREN  (2-D coord → RGB)
# ══════════════════════════════════════════════════════════════

class SineLayer(nn.Module):
    """sin(ω₀ · Wx + b) の 1 層。"""

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0  = omega_0
        self.is_first = is_first
        self.linear   = nn.Linear(in_features, out_features)
        """self._init_weights()
        
    def _init_weights(self) -> None:
        fan_in = self.linear.in_features
        if self.is_first:
            bound = 1.0 / fan_in
        else:
            bound = math.sqrt(6.0 / fan_in) / self.omega_0
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)"""

    def forward(self, x: Tensor) -> Tensor:
        return torch.concatenate((torch.sin(self.omega_0 * x[..., ::2]) * torch.nn.functional.softplus(self.omega_0 * y[..., 1::2]) / self.omega_0, torch.cos(self.omega_0 * x[..., ::2]) * torch.nn.functional.softplus(self.omega_0 * y[..., 1::2]) / self.omega_0), dim=-1)


class SIREN(nn.Module):
    """
    SIREN for INR:
      in_features=2  (normalized x, y)
      out_features=3 (R, G, B  in [-1, 1])
    """

    def __init__(self,
                 in_features:     int   = 2,
                 hidden_features: int   = 256,
                 hidden_layers:   int   = 4,
                 out_features:    int   = 3,
                 first_omega_0:   float = 30.0,
                 hidden_omega_0:  float = 30.0):
        super().__init__()
        layers: List[nn.Module] = []

        layers.append(SineLayer(in_features, hidden_features,
                                omega_0=first_omega_0, is_first=True))
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
        """coords: (..., 2) → rgb: (..., 3)"""
        return self.net(coords)


# ══════════════════════════════════════════════════════════════
#  2.  Muon Optimizer
# ══════════════════════════════════════════════════════════════

def _zeropower_via_newtonschulz5(G: Tensor,
                                  steps: int = 6,
                                  eps: float  = 1e-7) -> Tensor:
    """Newton-Schulz 反復で行列 G の直交極因子を計算。"""
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.bfloat16() if G.is_cuda else G.float()
    X = X / (X.norm() + eps)

    transposed = G.size(-2) > G.size(-1)
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
    Muon: MomentUm Orthogonalized by Newton-schulz
    2-D weight → 直交化勾配, 1-D bias → Nesterov SGD
    """

    def __init__(self, params,
                 lr:           float = 0.02,
                 momentum:     float = 0.95,
                 nesterov:     bool  = True,
                 ns_steps:     int   = 6,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, mu  = group['lr'], group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            wd       = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.clone()
                if wd != 0.0:
                    g.add_(p, alpha=wd)

                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(p)
                buf: Tensor = state['buf']
                buf.mul_(mu).add_(g)

                g = g.add(buf, alpha=mu) if nesterov else buf.clone()

                if g.ndim >= 2:
                    g = _zeropower_via_newtonschulz5(g, steps=ns_steps)
                    g = g * max(g.size(-2), g.size(-1)) ** 0.5

                p.add_(g, alpha=-lr)
        return loss


# ══════════════════════════════════════════════════════════════
#  3.  COCO INR Dataset
# ══════════════════════════════════════════════════════════════

class CocoINRDataset(Dataset):
    """
    COCO 画像を INR タスクとして返すデータセット。
    1 サンプル = 1 画像全ピクセルの (coord, rgb) ペア。
    image_dir のみ指定すれば annotation 不要で動作する。

    Args:
        image_dir   : COCO 画像フォルダ (train2017 等)
        img_size    : リサイズ後の解像度 (H, W)
        n_support   : support set のピクセル数
        n_query     : query set のピクセル数 (None → 全ピクセル)
        max_images  : データセットサイズの上限 (None → 全画像)
    """

    def __init__(self,
                 image_dir:  str,
                 img_size:   Tuple[int, int] = (64, 64),
                 n_support:  int  = 512,
                 n_query:    Optional[int] = 2048,
                 max_images: Optional[int] = None):
        self.image_dir = Path(image_dir)
        self.img_size  = img_size
        self.n_support = n_support
        self.n_query   = n_query

        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        paths = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in exts
        )
        if max_images is not None:
            paths = paths[:max_images]
        self.paths = paths

        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No images found in {image_dir}. "
                "Please check the path or download COCO images first."
            )

        H, W = img_size
        # 正規化座標グリッド [-1, 1] × [-1, 1]
        ys = torch.linspace(-1, 1, H)
        xs = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        # (H*W, 2)
        self.coords: Tensor = torch.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1
        )  # (N_pixels, 2)

        self.to_tensor = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),                   # [0,1]
            transforms.Normalize([0.5]*3, [0.5]*3),  # → [-1,1]
        ])

        total_pixels = H * W
        self.n_query_eff = min(n_query, total_pixels) if n_query else total_pixels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert('RGB')
        rgb = self.to_tensor(img)           # (3, H, W)
        rgb = rgb.permute(1, 2, 0).reshape(-1, 3)  # (N_pixels, 3)

        total = rgb.size(0)
        perm  = torch.randperm(total)

        s_idx = perm[:self.n_support]
        q_idx = perm[self.n_support: self.n_support + self.n_query_eff]

        return {
            'support_coords': self.coords[s_idx],   # (n_support, 2)
            'support_rgb':    rgb[s_idx],            # (n_support, 3)
            'query_coords':   self.coords[q_idx],    # (n_query, 2)
            'query_rgb':      rgb[q_idx],            # (n_query, 3)
            'full_coords':    self.coords,           # (N_pixels, 2)  ← 可視化用
            'full_rgb':       rgb,                   # (N_pixels, 3)
            'path':           str(self.paths[idx]),
        }


def coco_collate(batch):
    """可変長フィールド (path) を除いてスタック。"""
    keys = ['support_coords', 'support_rgb', 'query_coords', 'query_rgb']
    out  = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    out['paths'] = [b['path'] for b in batch]
    # full_coords / full_rgb は可視化用なのでリストで渡す
    out['full_coords'] = [b['full_coords'] for b in batch]
    out['full_rgb']    = [b['full_rgb']    for b in batch]
    return out


# ══════════════════════════════════════════════════════════════
#  4.  Inner Loop (task adaptation)
# ══════════════════════════════════════════════════════════════

_muon_first_order_warned = False


def inner_loop(
    model:         SIREN,
    support_coords: Tensor,   # (n_support, 2)
    support_rgb:    Tensor,   # (n_support, 3)
    inner_lr:      float = 1e-3,
    n_inner_steps: int   = 3,
    use_muon:      bool  = True,
    first_order:   bool  = False,
) -> SIREN:
    """
    Support set で fast_model を few-step 適応させる。
    first_order=False → 2次微分あり（完全 MAML）
    first_order=True  → FOMAML（メモリ節約）

    NOTE: Muon の step() は @torch.no_grad() のため、
    create_graph=True のグラフが活用されない。Muon 使用時は
    自動的に first_order=True として動作する。
    """
    global _muon_first_order_warned

    # Muon.step() は @torch.no_grad() なので create_graph=True の
    # 計算グラフは step() で切断される。メモリの無駄を避けるため
    # Muon 使用時は first_order を強制する。
    effective_first_order = first_order
    if use_muon and not first_order:
        effective_first_order = True
        if not _muon_first_order_warned:
            print('[INFO] Muon inner loop: create_graph=True は '
                  '@torch.no_grad() で無効化されるため '
                  'first_order=True として動作します')
            _muon_first_order_warned = True

    fast = copy.deepcopy(model)
    fast.train()

    if use_muon:
        optimizer = Muon(fast.parameters(), lr=inner_lr,
                         momentum=0.5, nesterov=True, ns_steps=5)
    else:
        optimizer = torch.optim.SGD(fast.parameters(), lr=inner_lr,
                                    momentum=0.9, nesterov=True)

    for _ in range(n_inner_steps):
        pred = fast(support_coords)
        loss = F.mse_loss(pred, support_rgb)

        optimizer.zero_grad()
        if effective_first_order:
            loss.backward()
        else:
            grads = torch.autograd.grad(
                loss, fast.parameters(), create_graph=True
            )
            for p, g in zip(fast.parameters(), grads):
                p.grad = g
        optimizer.step()

    return fast


# ══════════════════════════════════════════════════════════════
#  5.  Meta-Training Loop
# ══════════════════════════════════════════════════════════════

def meta_train(
    meta_model:    SIREN,
    meta_optimizer: torch.optim.Optimizer,
    train_loader:  DataLoader,
    n_epochs:      int   = 10,
    inner_lr:      float = 1e-3,
    n_inner_steps: int   = 3,
    use_muon_inner: bool = True,
    first_order:   bool  = False,
    grad_clip:     float = 1.0,
    device:        str   = 'cpu',
    save_dir:      str   = 'checkpoints',
    log_every:     int   = 50,
) -> List[float]:

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
            # batch: dict with (B, n_support, *) tensors
            sx = batch['support_coords'].to(device)   # (B, n_support, 2)
            sy = batch['support_rgb'].to(device)       # (B, n_support, 3)
            qx = batch['query_coords'].to(device)      # (B, n_query,   2)
            qy = batch['query_rgb'].to(device)          # (B, n_query,   3)
            B  = sx.size(0)

            # ── タスクごとに即 backward してグラフを解放 ──
            # (L1+L2+...+LB)/B の勾配 = 各 Li/B の勾配の和（勾配の線形性）
            meta_optimizer.zero_grad()
            total_loss = 0.0

            for i in range(B):
                fast = inner_loop(
                    meta_model,
                    sx[i], sy[i],
                    inner_lr      = inner_lr,
                    n_inner_steps = n_inner_steps,
                    use_muon      = use_muon_inner,
                    first_order   = first_order,
                )
                q_pred = fast(qx[i])
                task_loss = F.mse_loss(q_pred, qy[i]) / B
                task_loss.backward()          # 即 backward → グラフ解放
                total_loss += task_loss.item()
                del fast, q_pred, task_loss   # 参照を明示的に解放

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    meta_model.parameters(), max_norm=grad_clip
                )
            meta_optimizer.step()

            # GPU メモリの断片化を軽減
            if device == 'cuda':
                torch.cuda.empty_cache()

            loss_val = total_loss
            epoch_losses.append(loss_val)
            all_losses.append(loss_val)
            global_step += 1

            pbar.set_postfix({'meta_loss': f'{loss_val:.5f}'})

            if global_step % log_every == 0:
                avg = sum(all_losses[-log_every:]) / log_every
                tqdm.write(
                    f'[step {global_step:6d}] avg_meta_loss = {avg:.6f}'
                )

        # ── epoch 末にチェックポイント保存 ──
        avg_epoch = sum(epoch_losses) / max(len(epoch_losses), 1)
        print(f'\nEpoch {epoch} done. avg_loss = {avg_epoch:.6f}')
        torch.save({
            'epoch':        epoch,
            'global_step':  global_step,
            'model_state':  meta_model.state_dict(),
            'optim_state':  meta_optimizer.state_dict(),
            'all_losses':   all_losses,
        }, os.path.join(save_dir, f'ckpt_epoch{epoch:03d}.pt'))

    return all_losses


# ══════════════════════════════════════════════════════════════
#  6.  Evaluation & Visualization
# ══════════════════════════════════════════════════════════════

def psnr(pred: Tensor, target: Tensor) -> float:
    """PSNR in dB (both in [-1,1])。"""
    mse = F.mse_loss(pred, target).item()
    return 10 * math.log10(4.0 / (mse + 1e-8))  # max range = 2 ([-1,1])


@torch.no_grad()
def evaluate(
    meta_model:    SIREN,
    val_loader:    DataLoader,
    inner_lr:      float = 1e-3,
    n_inner_steps: int   = 10,
    use_muon:      bool  = True,
    device:        str   = 'cpu',
    n_vis:         int   = 4,
    save_dir:      str   = 'eval_outputs',
) -> dict:
    """
    val set で各タスクに適応後の PSNR を計測し、
    最初の n_vis 枚の再構成画像を保存。
    """
    os.makedirs(save_dir, exist_ok=True)
    meta_model.eval()

    psnr_before_list: List[float] = []
    psnr_after_list:  List[float] = []
    vis_count = 0

    for batch in tqdm(val_loader, desc='Evaluating', dynamic_ncols=True):
        sx   = batch['support_coords'].to(device)
        sy   = batch['support_rgb'].to(device)
        full_coords_list = batch['full_coords']
        full_rgb_list    = batch['full_rgb']
        B = sx.size(0)

        for i in range(B):
            fc = full_coords_list[i].to(device)   # (N_pix, 2)
            fr = full_rgb_list[i].to(device)       # (N_pix, 3)

            # ── 適応前 PSNR ──
            pred_before = meta_model(fc)
            pb = psnr(pred_before.clamp(-1, 1), fr)
            psnr_before_list.append(pb)

            # ── 適応後 PSNR ──
            fast = inner_loop(
                meta_model, sx[i], sy[i],
                inner_lr=inner_lr, n_inner_steps=n_inner_steps,
                use_muon=use_muon, first_order=True,
            )
            fast.eval()
            with torch.no_grad():
                pred_after = fast(fc)
            pa = psnr(pred_after.clamp(-1, 1), fr)
            psnr_after_list.append(pa)

            # ── 可視化 ──
            if vis_count < n_vis:
                H, W = val_loader.dataset.img_size

                def to_img(t: Tensor) -> 'np.ndarray':
                    import numpy as np
                    arr = t.reshape(H, W, 3).cpu().numpy()
                    arr = (arr * 0.5 + 0.5).clip(0, 1)
                    return arr

                gt    = to_img(fr)
                recon = to_img(pred_after.clamp(-1, 1))

                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(gt);    axes[0].set_title('Ground Truth')
                axes[1].imshow(recon); axes[1].set_title(
                    f'After {n_inner_steps} steps\nPSNR={pa:.2f} dB'
                )
                for ax in axes:
                    ax.axis('off')
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
        f"PSNR after={result['psnr_after']:.2f} dB"
    )
    return result


def plot_losses(losses: List[float], save_path: str = 'training_curve.png'):
    w = min(200, len(losses) // 5 + 1)
    ma = [sum(losses[max(0, i-w):i+1]) / min(i+1, w) for i in range(len(losses))]
    plt.figure(figsize=(9, 4))
    plt.plot(losses, linewidth=0.6, alpha=0.5, color='steelblue', label='step loss')
    plt.plot(ma, linewidth=2.0,    color='tomato',    label=f'moving avg ({w})')
    plt.xlabel('Meta-step'); plt.ylabel('Meta-loss (MSE)')
    plt.title('MAML + SIREN + Muon  —  COCO INR meta-training')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f'Saved → {save_path}')


# ══════════════════════════════════════════════════════════════
#  7.  CLI Entry Point
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='MAML + SIREN + Muon on COCO INR tasks'
    )
    # ── データ ──
    p.add_argument('--train_dir',   default='coco/images/train2017',
                   help='COCO train images directory')
    p.add_argument('--val_dir',     default='coco/images/val2017',
                   help='COCO val images directory')
    p.add_argument('--img_size',    type=int, nargs=2, default=[64, 64],
                   metavar=('H', 'W'),
                   help='Resize images to this resolution')
    p.add_argument('--n_support',   type=int, default=512,
                   help='Support set pixels per task')
    p.add_argument('--n_query',     type=int, default=2048,
                   help='Query set pixels per task')
    p.add_argument('--max_train',   type=int, default=None,
                   help='Max training images (None = all)')
    p.add_argument('--max_val',     type=int, default=200,
                   help='Max val images')

    # ── モデル ──
    p.add_argument('--hidden_features', type=int,   default=256)
    p.add_argument('--hidden_layers',   type=int,   default=4)
    p.add_argument('--first_omega',     type=float, default=30.0)
    p.add_argument('--hidden_omega',    type=float, default=30.0)

    # ── 学習 ──
    p.add_argument('--n_epochs',      type=int,   default=10)
    p.add_argument('--meta_batch',    type=int,   default=4,
                   help='Tasks per meta-update')
    p.add_argument('--meta_lr',       type=float, default=0.0001)
    p.add_argument('--inner_lr',      type=float, default=0.001)
    p.add_argument('--n_inner_steps', type=int,   default=3)
    p.add_argument('--first_order',   action='store_true',
                   help='Use FOMAML (faster, less memory)')
    p.add_argument('--no_muon_inner', action='store_true',
                   help='Use SGD instead of Muon for inner loop')
    p.add_argument('--grad_clip',     type=float, default=1.0)
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--seed',          type=int,   default=42)

    # ── 出力 ──
    p.add_argument('--save_dir',  default='checkpoints')
    p.add_argument('--eval_dir',  default='eval_outputs')
    p.add_argument('--resume',    default=None,
                   help='Path to checkpoint to resume from')

    # ── eval only ──
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--eval_inner_steps', type=int, default=10,
                   help='Inner steps at eval time')

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 64)
    print(f'  MAML + SIREN + Muon  |  device = {device}')
    print(f'  img_size      = {args.img_size}')
    print(f'  hidden        = {args.hidden_features} × {args.hidden_layers}')
    print(f'  inner_lr      = {args.inner_lr}  steps = {args.n_inner_steps}')
    print(f'  meta_lr       = {args.meta_lr}   batch = {args.meta_batch}')
    print(f'  first_order   = {args.first_order}')
    print(f'  muon_inner    = {not args.no_muon_inner}')
    print('=' * 64)

    # ── データセット ──
    img_size = tuple(args.img_size)

    train_ds = CocoINRDataset(
        image_dir  = args.train_dir,
        img_size   = img_size,
        n_support  = args.n_support,
        n_query    = args.n_query,
        max_images = args.max_train,
    )
    val_ds = CocoINRDataset(
        image_dir  = args.val_dir,
        img_size   = img_size,
        n_support  = args.n_support,
        n_query    = args.n_query,
        max_images = args.max_val,
    )

    print(f'Train images: {len(train_ds)}  |  Val images: {len(val_ds)}')

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.meta_batch,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = (device == 'cuda'),
        collate_fn  = coco_collate,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = 1,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = (device == 'cuda'),
        collate_fn  = coco_collate,
    )

    # ── モデル ──
    meta_model = SIREN(
        in_features     = 2,
        hidden_features = args.hidden_features,
        hidden_layers   = args.hidden_layers,
        out_features    = 3,
        first_omega_0   = args.first_omega,
        hidden_omega_0  = args.hidden_omega,
    ).to(device)

    n_params = sum(p.numel() for p in meta_model.parameters())
    print(f'Model params: {n_params:,}')

    # ── メタオプティマイザ ──
    meta_opt = optimizer = Muon(meta_model.parameters(), lr=args.meta_lr,
                         momentum=0.9, nesterov=True, ns_steps=5)

    # ── Resume ──
    start_losses: List[float] = []
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        meta_model.load_state_dict(ckpt['model_state'])
        meta_opt.load_state_dict(ckpt['optim_state'])
        start_losses = ckpt.get('all_losses', [])
        print(f'Resumed from {args.resume}  '
              f'(epoch {ckpt.get("epoch", "?")})')

    if args.eval_only:
        evaluate(
            meta_model,
            val_loader,
            inner_lr      = args.inner_lr,
            n_inner_steps = args.eval_inner_steps,
            use_muon      = not args.no_muon_inner,
            device        = device,
            save_dir      = args.eval_dir,
        )
        return

    # ── メタ訓練 ──
    losses = meta_train(
        meta_model      = meta_model,
        meta_optimizer  = meta_opt,
        train_loader    = train_loader,
        n_epochs        = args.n_epochs,
        inner_lr        = args.inner_lr,
        n_inner_steps   = args.n_inner_steps,
        use_muon_inner  = not args.no_muon_inner,
        first_order     = args.first_order,
        grad_clip       = args.grad_clip,
        device          = device,
        save_dir        = args.save_dir,
        log_every       = 50,
    )

    all_losses = start_losses + losses
    plot_losses(all_losses, save_path='training_curve.png')

    # ── 最終評価 ──
    evaluate(
        meta_model,
        val_loader,
        inner_lr      = args.inner_lr,
        n_inner_steps = args.eval_inner_steps,
        use_muon      = not args.no_muon_inner,
        device        = device,
        save_dir      = args.eval_dir,
    )


if __name__ == '__main__':
    main()
