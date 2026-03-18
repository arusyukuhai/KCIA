"""
COCO Dataset ダウンローダー
- train2017 / val2017 画像 + アノテーション
- 途中再開対応 (既存ファイルはスキップ)
- 解凍後に zip を削除するオプション付き
"""
 
import argparse
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
 
 
# ── ファイル定義 ──────────────────────────────────────────
COCO_FILES = [
    {
        "name":    "train2017.zip",
        "url":     "http://images.cocodataset.org/zips/train2017.zip",
        "size_gb": 18.0,
        "extract": "images",   # 解凍先サブディレクトリ
    },
    {
        "name":    "val2017.zip",
        "url":     "http://images.cocodataset.org/zips/val2017.zip",
        "size_gb": 0.8,
        "extract": "images",
    },
    {
        "name":    "annotations_trainval2017.zip",
        "url":     "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "size_gb": 0.25,
        "extract": ".",        # ルート直下に annotations/ が展開される
    },
]
 
 
# ── プログレスバー ────────────────────────────────────────
class ProgressBar:
    def __init__(self, total: int = 0):
        self.total = total
        self.width = 40
 
    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size > 0:
            self.total = total_size
 
        downloaded = block_num * block_size
        if self.total > 0:
            ratio   = min(downloaded / self.total, 1.0)
            filled  = int(self.width * ratio)
            bar     = "█" * filled + "░" * (self.width - filled)
            mb_done = downloaded / 1e6
            mb_tot  = self.total / 1e6
            print(
                f"\r  [{bar}] {ratio*100:5.1f}%  {mb_done:7.1f}/{mb_tot:.1f} MB",
                end="", flush=True,
            )
        else:
            print(f"\r  {block_num * block_size / 1e6:.1f} MB ...", end="", flush=True)
 
    def done(self):
        print()
 
 
# ── ダウンロード ──────────────────────────────────────────
def download_file(url: str, dest: Path, expected_gb: float) -> bool:
    """
    dest にファイルをダウンロード。既存なら (サイズ確認後) スキップ。
    Returns True if downloaded (or already present), False on error.
    """
    if dest.exists():
        size_gb = dest.stat().st_size / 1e9
        if size_gb > expected_gb * 0.95:
            print(f"  ✓ スキップ (既存): {dest.name}  ({size_gb:.2f} GB)")
            return True
        else:
            print(f"  ⚠ 不完全なファイルを検出、再ダウンロードします: {dest.name}")
            dest.unlink()
 
    print(f"  ダウンロード中: {dest.name}  (目安 {expected_gb:.1f} GB)")
    bar = ProgressBar()
    try:
        urlretrieve(url, dest, reporthook=bar)
        bar.done()
    except Exception as e:
        bar.done()
        print(f"  ✗ 失敗: {e}")
        if dest.exists():
            dest.unlink()
        return False
 
    actual_gb = dest.stat().st_size / 1e9
    print(f"  ✓ 完了: {dest.name}  ({actual_gb:.2f} GB)")
    return True
 
 
# ── 解凍 ─────────────────────────────────────────────────
def extract_zip(zip_path: Path, extract_dir: Path, remove_zip: bool = False):
    print(f"  解凍中: {zip_path.name} → {extract_dir}/")
    extract_dir.mkdir(parents=True, exist_ok=True)
 
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total   = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, extract_dir)
            if i % 5000 == 0 or i == total:
                print(f"\r    {i:,}/{total:,} ファイル  ({i/total*100:.1f}%)", end="", flush=True)
    print()
 
    if remove_zip:
        zip_path.unlink()
        print(f"  🗑  zip 削除: {zip_path.name}")
 
    print(f"  ✓ 解凍完了: {zip_path.name}")
 
 
# ── 解凍済みチェック ─────────────────────────────────────
def is_already_extracted(zip_name: str, data_dir: Path) -> bool:
    if "annotations" in zip_name:
        p = data_dir / "annotations" / "instances_train2017.json"
        return p.exists()
    if "train2017" in zip_name:
        d = data_dir / "images" / "train2017"
        return d.exists() and len(list(d.glob("*.jpg"))) > 100
    if "val2017" in zip_name:
        d = data_dir / "images" / "val2017"
        return d.exists() and len(list(d.glob("*.jpg"))) > 100
    return False
 
 
# ── ツリー表示 ───────────────────────────────────────────
def print_tree(path: Path, max_depth: int = 3, depth: int = 0, prefix: str = ""):
    if depth > max_depth or not path.exists():
        return
    items = sorted(path.iterdir()) if path.is_dir() else []
    shown, skipped = (items[:3], len(items) - 3) if len(items) > 6 else (items, 0)
 
    for i, item in enumerate(shown):
        is_last    = i == len(shown) - 1 and skipped == 0
        connector  = "└── " if is_last else "├── "
        size_str   = (
            f"  ({item.stat().st_size/1e6:.0f} MB)"
            if item.is_file() and item.stat().st_size > 1e6
            else ""
        )
        print(f"  {prefix}{connector}{item.name}{size_str}")
        if item.is_dir():
            print_tree(item, max_depth, depth + 1, prefix + ("    " if is_last else "│   "))
 
    if skipped > 0:
        print(f"  {prefix}└── ... 他 {skipped} 件")
 
 
# ── メイン ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="COCO 2017 ダウンローダー")
    parser.add_argument("--data_dir",         default="./data/coco",
                        help="保存先ルートディレクトリ (default: ./data/coco)")
    parser.add_argument("--split",            choices=["train", "val", "both"], default="both",
                        help="ダウンロードするsplit (default: both)")
    parser.add_argument("--no_extract",       action="store_true",
                        help="zip を解凍しない (zipのみ保存)")
    parser.add_argument("--remove_zip",       action="store_true",
                        help="解凍後に zip ファイルを削除してディスクを節約")
    parser.add_argument("--annotations_only", action="store_true",
                        help="アノテーションのみダウンロード (画像スキップ)")
    args = parser.parse_args()
 
    data_dir = Path(args.data_dir)
    zip_dir  = data_dir / "zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
 
    # ダウンロード対象フィルタリング
    targets = []
    for f in COCO_FILES:
        name = f["name"]
        if args.annotations_only and "annotations" not in name:
            continue
        if args.split == "train" and name == "val2017.zip":
            continue
        if args.split == "val" and name == "train2017.zip":
            continue
        targets.append(f)
 
    total_gb = sum(t["size_gb"] for t in targets)
    print("=" * 60)
    print("COCO 2017 ダウンローダー")
    print("=" * 60)
    print(f"保存先      : {data_dir.resolve()}")
    print(f"対象        : {len(targets)} ファイル  (~{total_gb:.1f} GB)")
    print(f"解凍        : {'しない' if args.no_extract else 'する'}")
    print(f"zip削除     : {'する' if args.remove_zip else 'しない'}")
    print()
 
    for entry in targets:
        name        = entry["name"]
        url         = entry["url"]
        size_gb     = entry["size_gb"]
        extract_sub = entry["extract"]
        zip_path    = zip_dir / name
        extract_dir = data_dir / extract_sub if extract_sub != "." else data_dir
 
        print(f"─── {name} ───")
 
        ok = download_file(url, zip_path, size_gb)
        if not ok:
            print(f"  ✗ ダウンロード失敗のためスキップ: {name}")
            print()
            continue
 
        if not args.no_extract:
            if is_already_extracted(name, data_dir):
                print(f"  ✓ 解凍済みのためスキップ")
                if args.remove_zip and zip_path.exists():
                    zip_path.unlink()
                    print(f"  🗑  zip 削除: {name}")
            else:
                extract_zip(zip_path, extract_dir, remove_zip=args.remove_zip)
 
        print()
 
    # ── 完了サマリ ──
    print("=" * 60)
    print("完了！")
    print()
    print("ディレクトリ構成:")
    print_tree(data_dir, max_depth=3)
    print()
    print("学習スクリプトへの引数例:")
    print(f"  python siren_maml_muon_coco.py \\")
    print(f"    --coco_root {data_dir}/images/train2017 \\")
    print(f"    --ann_file  {data_dir}/annotations/instances_train2017.json")
 
 
if __name__ == "__main__":
    main()
 