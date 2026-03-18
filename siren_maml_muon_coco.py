"""
SIREN + LoRA風内側ループ (W*B@A + C@D) + Hyperball外側最適化によるMAML on COCO

アーキテクチャ概要:
  - SIREN: 各線形層にsin活性化とω₀スケーリング
  - LoRA風内側ループ:
      通常LoRA: W_eff = W + B@A
      本実装:   W_eff = W * (B@A) + C@D
        W : ベース重み (外側ループで学習, Hyperball)
        B, A: 乗算LoRA分解 (rank r, 内側ループで適応)
        C, D: 加算LoRA分解 (rank r, 内側ループで適応)
  - 外側ループ: Hyperball Optimization (Adam + 重み/更新ノルム正規化)
  - タスク: COCO画像を座標→RGB INR (Implicit Neural Representation) としてfew-shot適応
"""

import math
import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# ─────────────────────────────────────────────
# 1. Hyperball Optimizer (Adam base + Norm Projection)
# ─────────────────────────────────────────────

def newton_schulz(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Newton-Schulz反復によって G を直交行列に近似する。
    X_{k+1} = a*X_k + b*X_k @ X_k^T @ X_k  (a=1.5, b=-0.5 の近似)
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)   # 5次多項式係数（NSP5）
    X = G / (G.norm() + eps)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X


def _normalize_to_sphere(x: Tensor, radius: float, eps: float = 1e-8) -> Tensor:
    """テンソルをフラット化してL2ノルムでradius球面上に射影する。"""
    norm = x.norm() + eps
    return x * (radius / norm)


class Hyperball(torch.optim.Optimizer):
    """
    Hyperball Optimization (Wen et al.)
    W_{t+1} = Normalize_R( W_t - η · Normalize_R(u_t) )

    - Adamで更新量 u_t を計算
    - u_t を初期重みノルム R の球面上に射影して正規化
    - 更新後の重みも R の球面上に再射影
    - 2D以上のパラメータにはNewton-Schulz直交化も適用可能
    - 1Dパラメータ (bias等): 通常のAdamにfallback
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        ns_steps: int = 5,
        use_ns: bool = True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            ns_steps=ns_steps,
            use_ns=use_ns,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group["lr"]
            beta1, beta2 = group["betas"]
            eps      = group["eps"]
            ns_steps = group["ns_steps"]
            use_ns   = group["use_ns"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # ── 状態初期化 ──
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"]     = torch.zeros_like(p)
                    state["exp_avg_sq"]  = torch.zeros_like(p)
                    # 初期重みノルム R を記録
                    state["init_norm"] = p.norm().item()
                    # scratch buffer (一時テンソル再利用でメモリ節約)
                    state["scratch"] = torch.empty_like(p)

                state["step"] += 1
                t = state["step"]
                R = state["init_norm"]

                exp_avg    = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                scratch    = state["scratch"]

                # ── Adam更新量の計算 (インプレース) ──
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # gradの参照を早期解放
                p.grad = None

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                # scratch = exp_avg_sq / bc2  (インプレース再利用)
                torch.div(exp_avg_sq, bias_correction2, out=scratch)
                scratch.sqrt_().add_(eps)
                # scratch = (exp_avg / bc1) / scratch = Adam update
                scratch.reciprocal_().mul_(exp_avg).div_(bias_correction1)
                # scratch が update を保持

                if p.ndim >= 2:
                    # 2D以上 → Newton-Schulz直交化 (オプション)
                    if use_ns:
                        orig_shape = scratch.shape
                        g2d = scratch.view(orig_shape[0], -1)
                        g2d = newton_schulz(g2d, steps=ns_steps)
                        scale = orig_shape[0] ** 0.5
                        scratch = g2d.view(orig_shape).mul_(scale)

                    # Hyperball: 更新ノルム正規化 + 重みノルム正規化
                    if R > 0:
                        norm = scratch.norm().clamp(min=1e-8)
                        scratch.mul_(R / norm)
                        p.add_(scratch, alpha=-lr)
                        norm = p.data.norm().clamp(min=1e-8)
                        p.data.mul_(R / norm)
                    else:
                        p.add_(scratch, alpha=-lr)
                else:
                    # 1Dパラメータ → 通常のAdam更新
                    p.add_(scratch, alpha=-lr)

        return loss


# ─────────────────────────────────────────────
# 2. LoRA風アダプタ定義 (W*B@A + C@D)
# ─────────────────────────────────────────────

class LoRAStyleAdapter(nn.Module):
    """
    LoRA風アダプタ。内側ループで適応するパラメータ。

    有効重み計算:
        W_eff = W_base * (I + B @ A) + C @ D
        ただし W_base は外から渡す (Hyperball管理下)

    注: ここでは「W * (B@A)」をハダマード積ではなく
        「W_base @ (B @ A)」（行列積の合成）として実装する。
        つまり W_eff = W_base @ (I + B @ A) + C @ D
        - 乗算項: W_base に右から (I + B @ A) を掛ける
          → 特徴空間のロータリー的変換
        - 加算項: C @ D は通常LoRA
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank = rank

        # 乗算LoRA (W @ (I + B@A))
        # B: (in, r), A: (r, in)  → B@A: (in, in)
        self.B_mul = nn.Parameter(torch.zeros(in_features, rank))
        self.A_mul = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.B_mul, a=math.sqrt(5))
        nn.init.zeros_(self.A_mul)   # 初期状態: I + 0 = I (恒等変換)

        # 加算LoRA (C@D)
        # C: (out, r), D: (r, in)
        self.C_add = nn.Parameter(torch.zeros(out_features, rank))
        self.D_add = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.C_add, a=math.sqrt(5))
        nn.init.zeros_(self.D_add)   # 初期状態: 0 (何も加算しない)

    def forward(self, x: Tensor, W_base: Tensor, bias: Optional[Tensor]) -> Tensor:
        """
        x      : (batch, in_features)
        W_base : (out_features, in_features)  ← SIRENのベース重み
        bias   : (out_features,) or None
        """
        # 乗算項: x → x @ (I + B@A)^T = x @ (I + A^T @ B^T)
        # 効率化: まず x @ B_mul で (batch, rank), 次に @ A_mul で (batch, in)
        mul_delta = x @ self.B_mul @ self.A_mul  # (batch, in_features)
        x_mul = x + mul_delta                    # x @ (I + B@A)

        # 加算項: x → (C @ D)^T @ x = D^T @ C^T
        # linear: out = x_mul @ W_base^T + x @ D^T @ C^T + bias
        out = F.linear(x_mul, W_base, bias)      # (batch, out_features)
        add_delta = x @ self.D_add.T @ self.C_add.T  # (batch, out_features)
        out = out + add_delta

        return out


# ─────────────────────────────────────────────
# 3. SIREN + LoRAアダプタ付き層
# ─────────────────────────────────────────────

class SIRENLoRALayer(nn.Module):
    """
    SIREN層 + LoRA風アダプタ。
    ベース重みWはHyperballで管理、アダプタは内側ループで適応。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        lora_rank: int = 4,
    ):
        super().__init__()
        self.omega_0    = omega_0
        self.is_first   = is_first
        self.in_features = in_features

        # ベース重み (Hyperballが更新)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # SIREN初期化
        if is_first:
            nn.init.uniform_(self.weight, -1 / in_features, 1 / in_features)
        else:
            nn.init.uniform_(
                self.weight,
                -math.sqrt(6 / in_features) / omega_0,
                 math.sqrt(6 / in_features) / omega_0,
            )

        # LoRAアダプタ (内側ループが更新)
        self.adapter = LoRAStyleAdapter(in_features, out_features, rank=lora_rank)

    def forward(self, x: Tensor) -> Tensor:
        out = self.adapter(x, self.weight, self.bias)
        return torch.sin(self.omega_0 * out)

    def base_params(self):
        """Hyperball管理対象: ベース重みとbias"""
        return [self.weight, self.bias]

    def adapter_params(self):
        """内側ループ管理対象: LoRAアダプタ"""
        return list(self.adapter.parameters())


# ─────────────────────────────────────────────
# 4. SIRENネットワーク (INR用)
# ─────────────────────────────────────────────

class SIRENLoRANet(nn.Module):
    """
    座標 (x, y) → RGB を学習するImplicit Neural Representation。
    入力: 2次元座標 [batch, 2]
    出力: RGB値 [batch, 3]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        omega_0: float = 30.0,
        lora_rank: int = 4,
    ):
        super().__init__()

        layers = []
        # 入力層
        layers.append(SIRENLoRALayer(2, hidden_dim, omega_0=omega_0, is_first=True, lora_rank=lora_rank))
        # 中間層
        for _ in range(num_layers - 2):
            layers.append(SIRENLoRALayer(hidden_dim, hidden_dim, omega_0=omega_0, lora_rank=lora_rank))
        # 出力層 (sin不要 → 通常linear)
        self.output_layer = nn.Linear(hidden_dim, 3)

        self.siren_layers = nn.ModuleList(layers)
        nn.init.uniform_(self.output_layer.weight, -math.sqrt(6 / hidden_dim) / omega_0,
                                                     math.sqrt(6 / hidden_dim) / omega_0)

    def forward(self, coords: Tensor) -> Tensor:
        """coords: (batch, 2) in [-1, 1]"""
        x = coords
        for layer in self.siren_layers:
            x = layer(x)
        return torch.sigmoid(self.output_layer(x))  # RGB in [0,1]

    def base_params(self):
        """外側ループ (Hyperball) が更新するパラメータ"""
        params = []
        for layer in self.siren_layers:
            params.extend(layer.base_params())
        params.extend(self.output_layer.parameters())
        return params

    def adapter_params(self):
        """内側ループが更新するLoRAアダプタパラメータ"""
        params = []
        for layer in self.siren_layers:
            params.extend(layer.adapter_params())
        return params


# ─────────────────────────────────────────────
# 5. MAMLトレーナー
# ─────────────────────────────────────────────

class MAMLTrainer:
    """
    MAML (Model-Agnostic Meta-Learning) トレーナー。
    - 内側ループ: LoRAアダプタパラメータのみをSGDで適応
    - 外側ループ: ベース重みをHyperball (Adam + ノルム射影) で更新
    """

    def __init__(
        self,
        model: SIRENLoRANet,
        inner_lr: float = 1e-2,
        inner_steps: int = 5,
        outer_lr: float = 1e-3,
        first_order: bool = False,
        device: str = "cuda",
    ):
        self.model       = model.to(device)
        self.inner_lr    = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.device      = device

        # 外側ループ: Hyperball (ベース重みのみ, Adam + ノルム射影)
        self.outer_optimizer = Hyperball(
            [{"params": model.base_params()}],
            lr=outer_lr,
            betas=(0.9, 0.999),
            use_ns=True,
        )

    # ── 内側ループ ──────────────────────────────

    def inner_loop(
        self,
        support_coords: Tensor,
        support_rgb: Tensor,
        fast_weights: Optional[dict] = None,
    ) -> dict:
        """
        アダプタパラメータをsupport setで適応する。
        fast_weights: 前のステップからのアダプタパラメータ (Noneなら初期化)
        Returns: 更新後のアダプタパラメータ dict {name: Tensor}
        """
        # アダプタパラメータのコピーを作成
        adapter_named = dict(
            (name, param)
            for layer in self.model.siren_layers
            for name, param in layer.adapter.named_parameters()
        )

        if fast_weights is None:
            fast_weights = {k: v.clone() for k, v in adapter_named.items()}

        for step in range(self.inner_steps):
            pred = self._forward_with_fast(support_coords, fast_weights)
            loss = F.mse_loss(pred, support_rgb)

            grads = torch.autograd.grad(
                loss,
                list(fast_weights.values()),
                create_graph=not self.first_order,
                allow_unused=True,
            )

            fast_weights = {
                k: v - self.inner_lr * (g if g is not None else torch.zeros_like(v))
                for (k, v), g in zip(fast_weights.items(), grads)
            }

        return fast_weights

    def _forward_with_fast(self, coords: Tensor, fast_weights: dict) -> Tensor:
        """
        fast_weightsを使ってモデルをforward。
        functional_callを使ってアダプタパラメータを差し替える。
        """
        # アダプタのパラメータ名→層インデックスのマッピングを構築してpatch
        # torch.nn.utils.parametrize や functional API を使う

        # より単純なアプローチ: 一時的にアダプタパラメータを書き換え
        adapter_named = {}
        layer_idx = 0
        for layer in self.model.siren_layers:
            for name, param in layer.adapter.named_parameters():
                full_key = f"layer{layer_idx}_{name}"
                adapter_named[full_key] = (layer.adapter, name)
            layer_idx += 1

        # fast_weightsのキーはflat (adapter内のname)
        # 全アダプタのパラメータをflat listとして取り出してfast_weightsと対応
        all_adapter_params = []
        for layer in self.model.siren_layers:
            for name, param in layer.adapter.named_parameters():
                all_adapter_params.append((layer.adapter, name, param))

        # 一時書き換え (コンテキスト外でもOKなようにtry-finally)
        original_data = {}
        key_list = list(fast_weights.keys())

        try:
            for i, (adapter_module, pname, _) in enumerate(all_adapter_params):
                if i < len(key_list):
                    key = key_list[i]
                    original_data[(id(adapter_module), pname)] = \
                        getattr(adapter_module, pname).data.clone()
                    getattr(adapter_module, pname).data.copy_(fast_weights[key])

            out = self.model(coords)
        finally:
            for (mod_id, pname), orig in original_data.items():
                # 元に戻す (adapter_moduleを再取得)
                pass  # 下で別途処理

        # よりクリーンな実装: torch.func.functional_call を使用
        return out

    def _forward_functional(self, coords: Tensor, fast_weights: dict) -> Tensor:
        """
        torch.func.functional_call を使ったクリーンなforward。
        """
        from torch.func import functional_call

        # アダプタパラメータのdict (モジュール名.パラメータ名 形式)
        params_dict = {}
        for i, layer in enumerate(self.model.siren_layers):
            for name, _ in layer.adapter.named_parameters():
                fk = list(fast_weights.keys())
                # インデックスベースで対応
                params_dict[f"siren_layers.{i}.adapter.{name}"] = fast_weights.get(
                    f"siren_layers.{i}.adapter.{name}",
                    layer.adapter.state_dict()[name]
                )

        return functional_call(self.model, params_dict, (coords,))

    # ── 外側ループ (メタ更新) ───────────────────

    def meta_step(self, tasks: list[dict]) -> float:
        """
        tasks: [{"support_coords", "support_rgb", "query_coords", "query_rgb"}, ...]
        Returns: meta loss (float)
        """
        self.outer_optimizer.zero_grad()

        meta_loss = torch.tensor(0.0, device=self.device)

        for task in tasks:
            s_coords = task["support_coords"].to(self.device)
            s_rgb    = task["support_rgb"].to(self.device)
            q_coords = task["query_coords"].to(self.device)
            q_rgb    = task["query_rgb"].to(self.device)

            # ─ 内側ループ: アダプタを適応 ─
            fast_weights = self._get_flat_adapter_weights()
            fast_weights = self._inner_loop_functional(s_coords, s_rgb, fast_weights)

            # ─ クエリセットで損失計算 ─
            q_pred = self._forward_with_fast_functional(q_coords, fast_weights)
            task_loss = F.mse_loss(q_pred, q_rgb)
            meta_loss = meta_loss + task_loss

        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        self.outer_optimizer.step()

        return meta_loss.item()

    def _get_flat_adapter_weights(self) -> dict:
        """全アダプタパラメータをflat dictとして返す (functional_call用のキー形式)"""
        weights = {}
        for i, layer in enumerate(self.model.siren_layers):
            for name, param in layer.adapter.named_parameters():
                weights[f"siren_layers.{i}.adapter.{name}"] = param.clone()
        return weights

    def _inner_loop_functional(
        self,
        support_coords: Tensor,
        support_rgb: Tensor,
        fast_weights: dict,
    ) -> dict:
        """functional_callを使った内側ループ"""
        from torch.func import functional_call

        for _ in range(self.inner_steps):
            pred = functional_call(self.model, fast_weights, (support_coords,))
            loss = F.mse_loss(pred, support_rgb)

            grads = torch.autograd.grad(
                loss,
                list(fast_weights.values()),
                create_graph=not self.first_order,
                allow_unused=True,
            )

            fast_weights = {
                k: v - self.inner_lr * (g if g is not None else torch.zeros_like(v))
                for (k, v), g in zip(fast_weights.items(), grads)
            }

        return fast_weights

    def _forward_with_fast_functional(self, coords: Tensor, fast_weights: dict) -> Tensor:
        from torch.func import functional_call
        return functional_call(self.model, fast_weights, (coords,))


# ─────────────────────────────────────────────
# 6. COCOタスクデータセット
# ─────────────────────────────────────────────

class COCOINRTaskDataset(Dataset):
    """
    アノテーション不要の画像ディレクトリからINRタスクを提供するDataset。
    val2017 のような「画像ファイルだけ入ったフォルダ」をそのまま指定できる。

    各タスク = 1枚の画像のサポート/クエリ座標サンプリング。
    """

    # PIL が読める代表的な拡張子
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(
        self,
        img_dir: str,
        img_size: int = 64,
        support_samples: int = 256,
        query_samples: int = 256,
    ):
        self.img_size        = img_size
        self.support_samples = support_samples
        self.query_samples   = query_samples

        img_dir = Path(img_dir)
        if not img_dir.exists():
            raise FileNotFoundError(f"画像ディレクトリが見つかりません: {img_dir}")

        self.paths = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTS
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"{img_dir} に対応画像ファイルがありません")

        print(f"[Dataset] {len(self.paths)} 枚の画像を読み込みました: {img_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),   # → (C, H, W) float [0, 1]
        ])

        # 座標グリッド生成 [-1, 1]  (H*W, 2)  ← 全タスク共通なのでキャッシュ
        xs = torch.linspace(-1, 1, img_size)
        ys = torch.linspace(-1, 1, img_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.all_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.paths[idx]).convert("RGB")
        img_tensor = self.transform(img)          # (3, H, W)

        # RGBフラット化: (H*W, 3)
        rgb = img_tensor.permute(1, 2, 0).reshape(-1, 3)

        # サポート/クエリをランダムサンプリング (重複なし)
        perm  = torch.randperm(self.all_coords.shape[0])
        s_idx = perm[:self.support_samples]
        q_idx = perm[self.support_samples:self.support_samples + self.query_samples]

        return {
            "support_coords": self.all_coords[s_idx],   # (S, 2)
            "support_rgb":    rgb[s_idx],                # (S, 3)
            "query_coords":   self.all_coords[q_idx],    # (Q, 2)
            "query_rgb":      rgb[q_idx],                # (Q, 3)
        }


def collate_tasks(batch: list[dict]) -> list[dict]:
    """DataLoaderのcollate: バッチをタスクリストとして返す"""
    return batch


# ─────────────────────────────────────────────
# 7. 学習ループ
# ─────────────────────────────────────────────

def train(
    img_dir: str          = "./data/coco/val2017",
    # モデル設定
    hidden_dim: int   = 256,
    num_layers: int   = 5,
    omega_0: float    = 30.0,
    lora_rank: int    = 8,
    # MAMLハイパーパラメータ
    inner_lr: float   = 1e-2,
    inner_steps: int  = 5,
    outer_lr: float   = 3e-4,
    first_order: bool = False,
    # 学習設定
    meta_batch_size: int = 4,
    num_epochs: int      = 50,
    img_size: int        = 64,
    support_samples: int = 256,
    query_samples: int   = 256,
    save_every: int      = 10,
    checkpoint_dir: str  = "./checkpoints",
    device: str          = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Device: {device}")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── データセット ──
    dataset = COCOINRTaskDataset(
        img_dir=img_dir,
        img_size=img_size,
        support_samples=support_samples,
        query_samples=query_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=meta_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
        collate_fn=collate_tasks,
    )

    # ── モデル・トレーナー ──
    model = SIRENLoRANet(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        omega_0=omega_0,
        lora_rank=lora_rank,
    )

    trainer = MAMLTrainer(
        model=model,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        outer_lr=outer_lr,
        first_order=first_order,
        device=device,
    )

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Base params (Hyperball): {sum(p.numel() for p in model.base_params()):,}")
    print(f"  Adapter params (inner): {sum(p.numel() for p in model.adapter_params()):,}")

    # ── 学習ループ ──
    for epoch in range(1, num_epochs + 1):
        epoch_losses = []

        for step, tasks in enumerate(loader):
            loss = trainer.meta_step(tasks)
            epoch_losses.append(loss)

            if step % 50 == 0:
                print(f"Epoch {epoch:3d} | Step {step:4d} | Loss: {np.mean(epoch_losses[-50:])} | PSNR: {-np.mean(np.log10(epoch_losses[-50:]) * 10)}")

            if step % save_every == 0:
                ckpt_path = Path(checkpoint_dir) / f"siren_maml_muon_epoch{epoch:04d}.pt"
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optimizer":   trainer.outer_optimizer.state_dict(),
                    "config": {
                        "hidden_dim": hidden_dim, "num_layers": num_layers,
                        "omega_0": omega_0, "lora_rank": lora_rank,
                        "inner_lr": inner_lr, "inner_steps": inner_steps,
                        "outer_lr": outer_lr,
                    },
                }, ckpt_path)

        avg_loss = np.mean(epoch_losses)
        print(f"── Epoch {epoch:3d} avg loss: {avg_loss} | PSNR: {-np.mean(np.log10(epoch_losses)) * 10} ──")

    return model


# ─────────────────────────────────────────────
# 8. 適応デモ (学習済みモデルで新画像に適応)
# ─────────────────────────────────────────────

@torch.no_grad()
def adapt_and_reconstruct(
    model: SIRENLoRANet,
    image_tensor: Tensor,     # (3, H, W) in [0,1]
    inner_lr: float = 1e-2,
    inner_steps: int = 100,
    device: str = "cuda",
) -> Tensor:
    """
    学習済みモデルを新画像に適応させてINR再構成を行う。
    Returns: 再構成画像 (3, H, W)
    """
    from torch.func import functional_call

    model = model.to(device)
    H, W  = image_tensor.shape[1], image_tensor.shape[2]

    xs = torch.linspace(-1, 1, W, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    all_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (H*W, 2)
    all_rgb    = image_tensor.permute(1, 2, 0).reshape(-1, 3).to(device)

    # 内側ループ (勾配あり)
    fast_weights = {}
    for i, layer in enumerate(model.siren_layers):
        for name, param in layer.adapter.named_parameters():
            fast_weights[f"siren_layers.{i}.adapter.{name}"] = param.clone().requires_grad_(True)

    for step in range(inner_steps):
        with torch.enable_grad():
            pred = functional_call(model, fast_weights, (all_coords,))
            loss = F.mse_loss(pred, all_rgb)
            grads = torch.autograd.grad(loss, list(fast_weights.values()))
            fast_weights = {
                k: (v - inner_lr * g).detach().requires_grad_(True)
                for (k, v), g in zip(fast_weights.items(), grads)
            }
        if step % 20 == 0:
            print(f"  Adapt step {step:3d} | MSE: {loss.item():.6f} | "
                  f"PSNR: {-10*math.log10(loss.item()):.2f} dB")

    # 再構成
    with torch.no_grad():
        recon_flat = functional_call(model, fast_weights, (all_coords,))
    return recon_flat.reshape(H, W, 3).permute(2, 0, 1).clamp(0, 1)


# ─────────────────────────────────────────────
# 9. エントリポイント
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SIREN+LoRA MAML with Hyperball on COCO val2017")
    parser.add_argument("--img_dir",       default="./data/coco/val2017",
                        help="画像ファイルが入ったディレクトリ (アノテーション不要)")
    parser.add_argument("--hidden_dim",    type=int,   default=96)
    parser.add_argument("--num_layers",    type=int,   default=5)
    parser.add_argument("--omega_0",       type=float, default=30.0)
    parser.add_argument("--lora_rank",     type=int,   default=6)
    parser.add_argument("--inner_lr",      type=float, default=1e-2)
    parser.add_argument("--inner_steps",   type=int,   default=12)
    parser.add_argument("--outer_lr",      type=float, default=3e-3)
    parser.add_argument("--first_order",   action="store_true", help="FOMAML")
    parser.add_argument("--meta_batch",    type=int,   default=4)
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--img_size",      type=int,   default=512)
    parser.add_argument("--support",       type=int,   default=16384)
    parser.add_argument("--query",         type=int,   default=16384)
    parser.add_argument("--save_every",    type=int,   default=50)
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    trained_model = train(
        img_dir=args.img_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        omega_0=args.omega_0,
        lora_rank=args.lora_rank,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        outer_lr=args.outer_lr,
        first_order=args.first_order,
        meta_batch_size=args.meta_batch,
        num_epochs=args.epochs,
        img_size=args.img_size,
        support_samples=args.support,
        query_samples=args.query,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )