"""
SIREN + LoRA風内側ループ (W*B@A + C@D) + Muon外側最適化によるMAML on COCO

アーキテクチャ概要:
  - SIREN: 各線形層にsin活性化とω₀スケーリング
  - LoRA風内側ループ:
      通常LoRA: W_eff = W + B@A
      本実装:   W_eff = W * (B@A) + C@D
        W : ベース重み (外側ループで学習, Muon)
        B, A: 乗算LoRA分解 (rank r, 内側ループで適応)
        C, D: 加算LoRA分解 (rank r, 内側ループで適応)
  - 外側ループ: Muon (Newton-Schulz直交化 + Nesterov momentum)
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
# 1. Muon Optimizer
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


class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-Schulz
    - 2D以上のパラメータ: Newton-Schulz直交化した勾配で更新
    - 1Dパラメータ (bias等): 通常のSGD+momentumにfallback
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr         = group["lr"]
            momentum   = group["momentum"]
            nesterov   = group["nesterov"]
            ns_steps   = group["ns_steps"]
            wd         = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                update = buf if not nesterov else grad.add(buf, alpha=momentum)

                if update.ndim >= 2:
                    # 2D以上 → Newton-Schulz直交化
                    orig_shape = update.shape
                    g2d = update.view(orig_shape[0], -1)
                    g2d = newton_schulz(g2d, steps=ns_steps)
                    # スケール保存: 元のFrobenius normを掛ける
                    scale = orig_shape[0] ** 0.5
                    update = g2d.view(orig_shape).mul_(scale)
                # 1D → そのまま使用

                p.add_(update, alpha=-lr)

        return loss


# ─────────────────────────────────────────────
# 2. LoRA風アダプタ定義 (W*B@A + C@D)
# ─────────────────────────────────────────────

class LoRAStyleAdapter(nn.Module):
    """
    LoRA風アダプタ。内側ループで適応するパラメータ。

    有効重み計算:
        W_eff = W_base * (I + B @ A) + C @ D
        ただし W_base は外から渡す (Muon管理下)

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
    ベース重みWはMuonで管理、アダプタは内側ループで適応。
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

        # ベース重み (Muonが更新)
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
        """Muon管理対象: ベース重みとbias"""
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
        """外側ループ (Muon) が更新するパラメータ"""
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
    - 外側ループ: ベース重みをMuonで更新
    - ロス安定化: 勾配クリッピング + スパイク検知 + ベストモデル復元
    """

    def __init__(
        self,
        model: SIRENLoRANet,
        inner_lr: float = 1e-2,
        inner_steps: int = 5,
        outer_lr: float = 1e-3,
        first_order: bool = False,
        device: str = "cuda",
        max_grad_norm: float = 1.0,
        spike_factor: float = 4.0,
        ema_decay: float = 0.99,
    ):
        self.model       = model.to(device)
        self.inner_lr    = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.device      = device

        # ── ロス安定化パラメータ ──
        self.max_grad_norm = max_grad_norm
        self.spike_factor  = spike_factor
        self.ema_decay     = ema_decay
        self._loss_ema: Optional[float]  = None   # ロスの指数移動平均
        self._best_loss: Optional[float] = None   # ベストロス
        self._best_state: Optional[dict] = None   # ベストモデル状態
        self._spike_count = 0

        # 外側ループ: Muon (ベース重みのみ)
        self.outer_optimizer = Muon(
            [{"params": model.base_params()}],
            lr=outer_lr,
            momentum=0.95,
            nesterov=True,
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

    def _is_spike(self, loss_val: float) -> bool:
        """ロスが EMA の spike_factor 倍を超えたら True"""
        if self._loss_ema is None:
            return False
        return loss_val > self._loss_ema * self.spike_factor

    def _update_ema(self, loss_val: float):
        """ロスの指数移動平均を更新"""
        if self._loss_ema is None:
            self._loss_ema = loss_val
        else:
            self._loss_ema = self.ema_decay * self._loss_ema + (1 - self.ema_decay) * loss_val

    def _save_best(self, loss_val: float):
        """ベストロスを更新し、モデル状態を保存"""
        if self._best_loss is None or self._loss_ema < self._best_loss:
            self._best_loss  = self._loss_ema
            self._best_state = copy.deepcopy(self.model.state_dict())

    def _rollback_to_best(self):
        """ベストモデル状態に復元"""
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            print(f"  [Rollback] ベストモデル (loss={self._best_loss:.6f}) に復元")

    def meta_step(self, tasks: list[dict]) -> float:
        """
        tasks: [{"support_coords", "support_rgb", "query_coords", "query_rgb"}, ...]
        Returns: meta loss (float)

        安定化機能:
          1. 勾配クリッピング (max_grad_norm)
          2. スパイク検知: EMA の spike_factor 倍を超えたら更新スキップ
          3. EMA がベストの 2 倍以上に悪化 → ベストモデルに復元
        """
        # ── ステップ前のスナップショット (スパイク時復元用) ──
        pre_step_state = copy.deepcopy(self.model.state_dict())

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
        loss_val  = meta_loss.item()

        # ── スパイク検知 ──
        if self._is_spike(loss_val):
            self._spike_count += 1
            print(f"  [Spike #{self._spike_count}] loss={loss_val:.6f} >> "
                  f"EMA={self._loss_ema:.6f} — 更新スキップ")
            # 重みをステップ前に戻す (逆伝播で汚れた in-place 変更があればリセット)
            self.model.load_state_dict(pre_step_state)
            self.outer_optimizer.zero_grad()
            return self._loss_ema  # EMA を返して学習カーブを滑らかに保つ

        # ── 通常更新 ──
        meta_loss.backward()

        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(
            self.model.base_params(), self.max_grad_norm
        )

        self.outer_optimizer.step()

        # ── EMA & ベスト更新 ──
        self._update_ema(loss_val)
        self._save_best(loss_val)

        # ── ドリフト検知: EMA がベストの 2 倍以上 → ベストに復元 ──
        if (self._best_loss is not None and self._loss_ema is not None
                and self._loss_ema > self._best_loss * 2.0):
            print(f"  [Drift] EMA={self._loss_ema:.6f} >> "
                  f"best={self._best_loss:.6f} — ベストモデルに復元")
            self._rollback_to_best()
            self._loss_ema = self._best_loss  # EMA もリセット

        return loss_val

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
    print(f"  Base params (Muon): {sum(p.numel() for p in model.base_params()):,}")
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

    parser = argparse.ArgumentParser(description="SIREN+LoRA MAML with Muon on COCO val2017")
    parser.add_argument("--img_dir",       default="./data/coco/val2017",
                        help="画像ファイルが入ったディレクトリ (アノテーション不要)")
    parser.add_argument("--hidden_dim",    type=int,   default=96)
    parser.add_argument("--num_layers",    type=int,   default=5)
    parser.add_argument("--omega_0",       type=float, default=30.0)
    parser.add_argument("--lora_rank",     type=int,   default=6)
    parser.add_argument("--inner_lr",      type=float, default=1e-2)
    parser.add_argument("--inner_steps",   type=int,   default=12)
    parser.add_argument("--outer_lr",      type=float, default=2e-5)
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