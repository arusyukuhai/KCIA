"""
MAML テストスクリプト: 学習済みメタ初期化から LoRA を実画像に適応

使い方:
  # 単一画像で適応テスト
  python test_maml_adapt.py --checkpoint ckpt.pt --image test.jpg

  # ディレクトリの画像全体でバッチ評価
  python test_maml_adapt.py --checkpoint ckpt.pt --img_dir ./data/coco/val2017 --max_images 50

  # ランダム初期化との比較 (--random_baseline)
  python test_maml_adapt.py --checkpoint ckpt.pt --image test.jpg --random_baseline

出力:
  - 適応過程の PSNR/SSIM を CSV に記録
  - 再構成画像を保存
  - ランダム初期化 vs MAML 初期化の比較 (オプション)
"""

import math
import csv
import copy
import time
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
from PIL import Image
import numpy as np

# 学習コードからモデル定義をインポート
from siren_maml_muon_coco import SIRENLoRANet


# ─────────────────────────────────────────────
# SSIM (構造的類似度)
# ─────────────────────────────────────────────

def _ssim_single_channel(x: Tensor, y: Tensor, window_size: int = 11) -> float:
    """単一チャネルの SSIM を計算 (x, y: (1,1,H,W))"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # ガウシアンカーネル
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)  # (ws, ws)
    window = window.unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)

    pad = window_size // 2
    mu_x  = F.conv2d(x, window, padding=pad)
    mu_y  = F.conv2d(y, window, padding=pad)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x * x, window, padding=pad) - mu_xx
    sigma_yy = F.conv2d(y * y, window, padding=pad) - mu_yy
    sigma_xy = F.conv2d(x * y, window, padding=pad) - mu_xy

    num   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denom = (mu_xx + mu_yy + C1) * (sigma_xx + sigma_yy + C2)

    return (num / denom).mean().item()


def compute_ssim(img1: Tensor, img2: Tensor) -> float:
    """RGB 画像の SSIM (img1, img2: (3, H, W) in [0,1])"""
    ssim_vals = []
    for c in range(3):
        x = img1[c:c+1].unsqueeze(0)
        y = img2[c:c+1].unsqueeze(0)
        ssim_vals.append(_ssim_single_channel(x, y))
    return float(np.mean(ssim_vals))


# ─────────────────────────────────────────────
# LoRA 適応 (内側ループのみ)
# ─────────────────────────────────────────────

def adapt_lora(
    model: SIRENLoRANet,
    image_tensor: Tensor,     # (3, H, W) in [0, 1]
    inner_lr: float = 1e-2,
    inner_steps: int = 200,
    log_interval: int = 10,
    device: str = "cuda",
    csv_writer=None,
    image_name: str = "",
    batch_size: int = 0,
) -> tuple[Tensor, list[dict]]:
    """
    学習済み MAML モデルの LoRA アダプタのみを実画像に適応。

    Returns:
        recon: 再構成画像 (3, H, W)
        log:   各ステップの {step, mse, psnr, ssim, time} リスト
    """
    from torch.func import functional_call

    model = model.to(device).eval()
    H, W = image_tensor.shape[1], image_tensor.shape[2]

    # 座標グリッド
    xs = torch.linspace(-1, 1, W, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    all_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    all_rgb    = image_tensor.permute(1, 2, 0).reshape(-1, 3).to(device)

    # LoRA アダプタパラメータを抽出 (ベース重みは凍結)
    fast_weights = {}
    for i, layer in enumerate(model.siren_layers):
        for name, param in layer.adapter.named_parameters():
            fast_weights[f"siren_layers.{i}.adapter.{name}"] = param.clone().requires_grad_(True)

    log = []
    t0 = time.perf_counter()

    N = all_coords.shape[0]
    bs = batch_size if batch_size > 0 else N

    for step in range(inner_steps):
        with torch.enable_grad():
            if bs >= N:
                # 全座標を一括処理
                pred = functional_call(model, fast_weights, (all_coords,))
                loss = F.mse_loss(pred, all_rgb)
            else:
                # ミニバッチで分割処理
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)
                for start in range(0, N, bs):
                    end = min(start + bs, N)
                    batch_coords = all_coords[start:end]
                    batch_rgb = all_rgb[start:end]
                    batch_pred = functional_call(model, fast_weights, (batch_coords,))
                    batch_loss = F.mse_loss(batch_pred, batch_rgb) * (end - start)
                    total_loss = total_loss + batch_loss
                loss = total_loss / N

            grads = torch.autograd.grad(loss, list(fast_weights.values()))
            fast_weights = {
                k: (v - inner_lr * g).detach().requires_grad_(True)
                for (k, v), g in zip(fast_weights.items(), grads)
            }

        if step % log_interval == 0 or step == inner_steps - 1:
            mse_val  = loss.item()
            psnr_val = -10.0 * math.log10(mse_val) if mse_val > 0 else float("inf")
            elapsed  = time.perf_counter() - t0

            # SSIM は重いので粗めに計算
            with torch.no_grad():
                if bs >= N:
                    recon_tmp = functional_call(model, fast_weights, (all_coords,))
                else:
                    parts = []
                    for start in range(0, N, bs):
                        end = min(start + bs, N)
                        parts.append(functional_call(model, fast_weights, (all_coords[start:end],)))
                    recon_tmp = torch.cat(parts, dim=0)
                recon_img = recon_tmp.reshape(H, W, 3).permute(2, 0, 1).clamp(0, 1)
                ssim_val  = compute_ssim(recon_img, image_tensor.to(device))

            entry = {
                "image": image_name,
                "step":  step,
                "mse":   mse_val,
                "psnr":  psnr_val,
                "ssim":  ssim_val,
                "time":  elapsed,
            }
            log.append(entry)

            if csv_writer:
                csv_writer.writerow([
                    image_name, step,
                    f"{mse_val:.8f}", f"{psnr_val:.4f}",
                    f"{ssim_val:.6f}", f"{elapsed:.3f}",
                ])

            print(f"  [{image_name}] Step {step:4d} | "
                  f"MSE: {mse_val:.6f} | PSNR: {psnr_val:.2f} dB | "
                  f"SSIM: {ssim_val:.4f} | {elapsed:.1f}s")

    # 最終再構成
    with torch.no_grad():
        if bs >= N:
            recon_flat = functional_call(model, fast_weights, (all_coords,))
        else:
            parts = []
            for start in range(0, N, bs):
                end = min(start + bs, N)
                parts.append(functional_call(model, fast_weights, (all_coords[start:end],)))
            recon_flat = torch.cat(parts, dim=0)
    recon = recon_flat.reshape(H, W, 3).permute(2, 0, 1).clamp(0, 1)

    return recon, log


# ─────────────────────────────────────────────
# チェックポイントからモデル構築
# ─────────────────────────────────────────────

def load_model_from_checkpoint(
    ckpt_path: str,
    device: str = "cuda",
    override_config: Optional[dict] = None,
) -> SIRENLoRANet:
    """チェックポイントからモデルをロード"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    if override_config:
        cfg.update(override_config)

    model = SIRENLoRANet(
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        omega_0=cfg["omega_0"],
        lora_rank=cfg["lora_rank"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"[Loaded] {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    print(f"  config: {cfg}")
    return model


# ─────────────────────────────────────────────
# メイン: 単一 / バッチ評価
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MAML テスト: LoRA 適応による画像再構成評価"
    )
    # 入力
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="学習済みチェックポイント (.pt)")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str,
                             help="テスト画像パス (単一)")
    input_group.add_argument("--img_dir", type=str,
                             help="テスト画像ディレクトリ (バッチ)")

    # 適応パラメータ
    parser.add_argument("--inner_lr",     type=float, default=1e-2)
    parser.add_argument("--inner_steps",  type=int,   default=200)
    parser.add_argument("--img_size",     type=int,   default=512)
    parser.add_argument("--batch_size",   type=int,   default=8192,
                        help="座標のバッチサイズ (0=全座標一括, 例: 65536)")
    parser.add_argument("--log_interval", type=int,   default=10)

    # バッチ評価用
    parser.add_argument("--max_images",   type=int,   default=0,
                        help="評価する最大画像数 (0=全部)")

    # 比較
    parser.add_argument("--random_baseline", action="store_true",
                        help="ランダム初期化モデルとの比較も実行")

    # 出力
    parser.add_argument("--output_dir",  type=str, default="./test_results")
    parser.add_argument("--device",      type=str,
                        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 画像リスト構築 ──
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    if args.image:
        image_paths = [Path(args.image)]
    else:
        img_dir = Path(args.img_dir)
        image_paths = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS
        )
        if args.max_images > 0:
            image_paths = image_paths[:args.max_images]

    print(f"評価画像数: {len(image_paths)}")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    # ── モデルロード ──
    model = load_model_from_checkpoint(args.checkpoint, args.device)

    # ── ランダムベースライン (オプション) ──
    random_model = None
    if args.random_baseline:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        cfg = ckpt["config"]
        random_model = SIRENLoRANet(
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            omega_0=cfg["omega_0"],
            lora_rank=cfg["lora_rank"],
        ).to(args.device)
        print("[Baseline] ランダム初期化モデルを作成")

    # ── CSV 初期化 ──
    csv_path = output_dir / "adapt_log.csv"
    csv_fp = open(csv_path, "w", newline="")
    csv_wr = csv.writer(csv_fp)
    csv_wr.writerow(["image", "step", "mse", "psnr", "ssim", "time_s"])

    csv_baseline_wr = None
    csv_baseline_fp = None
    if random_model:
        csv_bl_path = output_dir / "adapt_log_baseline.csv"
        csv_baseline_fp = open(csv_bl_path, "w", newline="")
        csv_baseline_wr = csv.writer(csv_baseline_fp)
        csv_baseline_wr.writerow(["image", "step", "mse", "psnr", "ssim", "time_s"])

    # ── 評価サマリ CSV ──
    summary_path = output_dir / "summary.csv"
    summary_fp = open(summary_path, "w", newline="")
    summary_wr = csv.writer(summary_fp)
    header = ["image", "final_psnr", "final_ssim", "final_mse", "adapt_time_s"]
    if random_model:
        header += ["baseline_psnr", "baseline_ssim", "baseline_mse", "psnr_gain"]
    summary_wr.writerow(header)

    all_results = []
    all_baseline_results = []

    # ── 画像ごとに適応 ──
    for img_idx, img_path in enumerate(image_paths):
        print(f"\n{'='*60}")
        print(f"[{img_idx+1}/{len(image_paths)}] {img_path.name}")
        print(f"{'='*60}")

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)  # (3, H, W) in [0,1]

        # ── MAML 初期化から適応 ──
        print("── MAML 初期化 ──")
        model_copy = copy.deepcopy(model)
        recon, log = adapt_lora(
            model_copy, img_tensor,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            log_interval=args.log_interval,
            device=args.device,
            csv_writer=csv_wr,
            image_name=img_path.stem,
            batch_size=args.batch_size,
        )
        csv_fp.flush()

        # 再構成画像を保存
        recon_pil = transforms.ToPILImage()(recon.cpu())
        recon_pil.save(output_dir / f"{img_path.stem}_recon.png")

        final = log[-1]
        all_results.append(final)

        # ── ランダムベースライン ──
        baseline_final = None
        if random_model:
            print("── ランダム初期化 ──")
            random_copy = copy.deepcopy(random_model)
            recon_bl, log_bl = adapt_lora(
                random_copy, img_tensor,
                inner_lr=args.inner_lr,
                inner_steps=args.inner_steps,
                log_interval=args.log_interval,
                device=args.device,
                csv_writer=csv_baseline_wr,
                image_name=img_path.stem,
                batch_size=args.batch_size,
            )
            csv_baseline_fp.flush()

            recon_bl_pil = transforms.ToPILImage()(recon_bl.cpu())
            recon_bl_pil.save(output_dir / f"{img_path.stem}_recon_baseline.png")

            baseline_final = log_bl[-1]
            all_baseline_results.append(baseline_final)

        # ── サマリ行 ──
        row = [
            img_path.stem,
            f"{final['psnr']:.4f}",
            f"{final['ssim']:.6f}",
            f"{final['mse']:.8f}",
            f"{final['time']:.3f}",
        ]
        if baseline_final:
            gain = final["psnr"] - baseline_final["psnr"]
            row += [
                f"{baseline_final['psnr']:.4f}",
                f"{baseline_final['ssim']:.6f}",
                f"{baseline_final['mse']:.8f}",
                f"{gain:+.4f}",
            ]
        summary_wr.writerow(row)
        summary_fp.flush()

    # ── 全体統計 ──
    csv_fp.close()
    if csv_baseline_fp:
        csv_baseline_fp.close()

    print(f"\n{'='*60}")
    print("全体統計")
    print(f"{'='*60}")

    psnrs = [r["psnr"] for r in all_results]
    ssims = [r["ssim"] for r in all_results]
    times = [r["time"] for r in all_results]

    avg_row = [
        "AVERAGE",
        f"{np.mean(psnrs):.4f}",
        f"{np.mean(ssims):.6f}",
        "",
        f"{np.mean(times):.3f}",
    ]
    print(f"  MAML  — PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB | "
          f"SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")

    if all_baseline_results:
        bl_psnrs = [r["psnr"] for r in all_baseline_results]
        bl_ssims = [r["ssim"] for r in all_baseline_results]
        gains    = [m["psnr"] - b["psnr"] for m, b in zip(all_results, all_baseline_results)]
        avg_row += [
            f"{np.mean(bl_psnrs):.4f}",
            f"{np.mean(bl_ssims):.6f}",
            "",
            f"{np.mean(gains):+.4f}",
        ]
        print(f"  Random — PSNR: {np.mean(bl_psnrs):.2f} ± {np.std(bl_psnrs):.2f} dB | "
              f"SSIM: {np.mean(bl_ssims):.4f} ± {np.std(bl_ssims):.4f}")
        print(f"  MAML gain: {np.mean(gains):+.2f} dB")

    summary_wr.writerow(avg_row)
    summary_fp.close()

    print(f"\n結果保存先: {output_dir}")
    print(f"  adapt_log.csv   — 各ステップの詳細ログ")
    print(f"  summary.csv     — 画像ごとの最終結果 + 全体平均")
    if random_model:
        print(f"  adapt_log_baseline.csv — ランダム初期化のログ")


if __name__ == "__main__":
    main()
