#!/usr/bin/env python3
"""
CMA-ES によるADF-CGP ハイパーパラメータ探索
============================================

探索対象:
  - N_LAYERS       : 2〜5 (整数)
  - LAYER_LEN      : 各層共通のノード数 4〜64 (整数)
  - N_ADF_PER_LAYER: 各ADF層共通のADF数 1〜8 (整数)
  - ELITE          : 8〜128 (整数)
  - POP_SIZE       : 128〜8192 (整数)
  - VEC_LEN        : 64〜2048 (整数)
  - PROB_EML       : 0.0〜1.0 (実数)
  - MUT_STOP_PROB  : 0.05〜0.6 (実数) ← 追加: Chromosome突然変異の幾何分布停止確率
  - MUT_MAX_TARGETS: 1〜8 (整数)      ← 追加: Genome突然変異の最大対象Chromosome数

評価方法:
  各候補パラメータでRustソースを生成 → cargo build --release →
  30秒間実行 → stdout から最終行の acc を取得
  目的関数 = -acc (最小化)

使い方:
  python3 cmaes_search.py [--src SRC_RS] [--out OUT_DIR] [--budget N] [--time SEC]

依存:
  pip install cma
  cargo / rustc が PATH 上にあること
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Optional

import cma

# ─── デフォルト設定 ───────────────────────────────────────────────────────────

DEFAULT_SRC     = "src/main.rs"   # Rustソースの相対パス (プロジェクトルートから)
DEFAULT_OUT     = "cmaes_results" # 結果ディレクトリ
DEFAULT_BUDGET  = 3000             # CMA-ES の最大評価回数
DEFAULT_TIME    = 40              # 各評価の実行秒数

# ─── ハイパーパラメータ定義 ──────────────────────────────────────────────────

# CMA-ES は実数ベクトルを操作する。
# 各パラメータを以下のスケーリングで扱う:
#   整数パラメータ → round(clip(x, lo, hi))
#   実数パラメータ → clip(x, lo, hi)

PARAM_DEFS = [
    # (name,                      lo,        hi,      init,    is_int, log_scale)
    ("N_LAYERS",                   2,         9,         4,    True,   True),
    ("LAYER_LEN",                  4,       128,        32,    True,   True),
    ("LAYER_LEN_LAST",             4,      1024,        64,    True,   True),
    ("N_ADF_PER_LAYER",            1,        32,        12,    True,   True),
    ("ELITE",                      8,       256,        24,    True,   True),
    ("POP_SIZE",                  48,     16384,      1024,    True,   True),
    ("VEC_LEN",                   64,     16384,      1024,    True,   True),
    ("PROB_EML",                 0.0,       1.0,       0.3,   False,  False),
    ("A",                       -1.5,       2.0,      0.01,   False,  False),
    ("B",                        0.0,       1.0,       0.5,   False,  False),
    ("C",                       -1.5,       2.0,      0.01,   False,  False),
    ("D",                        0.0,       1.0,       0.5,   False,  False),
    ("E",                       -1.5,       2.0,      0.01,   False,  False),
    ("HILO",                      7.5,      50.0,      10.0,   False,   True),
    ("P",                         0.5,       3.0,       1.5,   False,   True),
    ("MUT_STOP_PROB",           0.001,      12.0,       6.0,   False,  False),
    ("MUT_MAX_TARGETS",           1.0,       8.0,       3.0,    True,   True),

    # learned mutator / mixer
    ("LEARNED_MUTATION_PROB",     0.0,      0.75,      0.15,   False,  False),
    ("NEAREST_BETTER_CACHE_SIZE", 64.0, 1048576.0,   16384.0,    True,   True),
    ("MIXER_D1",                  8.0,     256.0,      64.0,    True,   True),
    ("MIXER_D2",                  8.0,     256.0,      64.0,    True,   True),
    ("MIXER_BLOCKS",              1.0,       6.0,       2.0,    True,   True),
    ("MIXER_TOKEN_HIDDEN",       16.0,     512.0,     128.0,    True,   True),
    ("MIXER_CHANNEL_HIDDEN",     16.0,     512.0,     128.0,    True,   True),
    ("MIXER_COND_PROJ_HIDDEN",    4.0,     256.0,      32.0,    True,   True),
    ("MIXER_COND_VEC_DIM",        2.0,     128.0,      32.0,    True,   True),
    ("MIXER_TRAIN_EVERY",         1.0,       8.0,       1.0,    True,   True),
    ("MIXER_TRAIN_EPOCHS",        1.0,       8.0,       1.0,    True,   True),
    ("MIXER_BATCH_SIZE",          1.0,      64.0,       8.0,    True,   True),
    ("MIXER_TRAIN_POP_SUBSET",    8.0,    2048.0,     256.0,    True,   True),
    ("MIXER_MAX_TEACHER_PAIRS",   8.0,    4096.0,     256.0,    True,   True),
    ("MIXER_MAX_MINIBATCHES",     1.0,     256.0,      16.0,    True,   True),
    ("MIXER_LR_MATRIX",       1.0e-4,     1.0e-1,     2.0e-2,   False,   True),
    ("MIXER_LR_VECTOR",       1.0e-5,     1.0e-2,     1.0e-3,   False,   True),
    ("MUON_WEIGHT_DECAY",     1.0e-6,     1.0e-1,     1.0e-2,   False,   True),
    ("MUON_MOMENTUM",             0.5,     0.999,      0.95,    False,  False),
    ("MUON_NS_STEPS",             1.0,      10.0,       5.0,    True,   True),
    ("MUON_NESTEROV",             0.0,       1.0,       1.0,    True,  False),
    ("ADAMW_BETA1",               0.5,     0.999,       0.9,    False,  False),
    ("ADAMW_BETA2",               0.8,    0.9999,      0.95,    False,  False),
    ("ADAMW_EPS",             1.0e-10,    1.0e-6,     1.0e-8,   False,   True),
    ("ADAMW_WEIGHT_DECAY",    1.0e-6,     1.0e-1,     1.0e-2,   False,   True),
]



# ─── ユーティリティ ──────────────────────────────────────────────────────────

def encode(params: dict) -> list[float]:
    """パラメータ辞書 → CMA-ES ベクトル"""
    vec = []
    for name, lo, hi, init, is_int, log_scale in PARAM_DEFS:
        v = params[name]
        if log_scale:
            vec.append(math.log(max(v, 1e-6)))
        else:
            vec.append(float(v))
    return vec


def decode(x: list[float]) -> dict:
    """CMA-ES ベクトル → パラメータ辞書"""
    result = {}
    for i, (name, lo, hi, init, is_int, log_scale) in enumerate(PARAM_DEFS):
        v = x[i]
        if log_scale:
            v = math.exp(v)
        v = max(lo, min(hi, v))
        if is_int:
            v = int(round(v))
            v = max(int(lo), min(int(hi), v))
        result[name] = v
    return result


def initial_x() -> list[float]:
    params = {name: init for name, lo, hi, init, is_int, log_scale in PARAM_DEFS}
    return encode(params)


def initial_sigma() -> float:
    return 6.0


# ─── Rustソース生成 ──────────────────────────────────────────────────────────

RUST_TEMPLATE = """\
// AUTO-GENERATED by cmaes_search.py — DO NOT EDIT
use ahash::AHasher;
use dashmap::DashMap;
use ndarray::{{s, Array1, Array2, Axis}};
use num_complex::Complex;
use rand::rngs::{{SmallRng, StdRng}};
use rand::seq::SliceRandom;
use rand::{{Rng, SeedableRng}};
use rand_distr::{{Distribution, StandardNormal}};
use rayon::prelude::*;
use std::hash::{{Hash, Hasher}};
use std::sync::Arc;

// ─── アーキテクチャ ハイパーパラメータ ───────────────────────────────────────
const N_LAYERS: usize = {N_LAYERS};
const LAYER_LEN: [usize; N_LAYERS] = [{LAYER_LEN_ARR}];
const N_INPUTS_MAIN: usize = 2;
/// ADF の外部入力: [in0, in1, one, x_global]
/// x_global はトップ層の ext[0]（入力 x）を全 ADF 層に伝播させたもの。
const N_INPUTS_ADF: usize = 4;
const N_ADF_PER_LAYER: [usize; N_LAYERS - 1] = [{N_ADF_ARR}];

const ONE_SIG: Sig = 0xFFFF_FFFF_FFFF_FFFFu64;
/// x_global の固定 Sig。バッチ変化時は exec_adf に渡す x_sig 引数で上書きされる。
const X_GLOBAL_SIG_SLOT: usize = 3;
const VEC_LEN: usize = {VEC_LEN};
const POP_SIZE: usize = {POP_SIZE};
const ELITE: usize = {ELITE};
const N_GEN: usize = 99999999;  // 時間制限で止める
const PROB_EML: f64 = {PROB_EML};
const A: f64 = {A};
const B: f64 = {B};
const C: f64 = {C};
const D: f64 = {D};
const E: f64 = {E};
const HILO: f64 = {HILO};
const P: f64 = {P};

// ─── 突然変異率パラメータ ────────────────────────────────────────────────────
const MUT_STOP_PROB: f64 = {MUT_STOP_PROB};
const MUT_MAX_TARGETS: usize = {MUT_MAX_TARGETS};

// ─── learned mutator / mixer hyperparams ─────────────────────────────────────
const LEARNED_MUTATION_PROB: f64 = {LEARNED_MUTATION_PROB};
const MIXER_D1: usize = {MIXER_D1};
const MIXER_D2: usize = {MIXER_D2};
const MIXER_BLOCKS: usize = {MIXER_BLOCKS};
const MIXER_TOKEN_HIDDEN: usize = {MIXER_TOKEN_HIDDEN};
const MIXER_CHANNEL_HIDDEN: usize = {MIXER_CHANNEL_HIDDEN};
const MIXER_COND_PROJ_HIDDEN: usize = {MIXER_COND_PROJ_HIDDEN};
const MIXER_COND_VEC_DIM: usize = {MIXER_COND_VEC_DIM};
const MIXER_TRAIN_EVERY: usize = {MIXER_TRAIN_EVERY};
const MIXER_TRAIN_EPOCHS: usize = {MIXER_TRAIN_EPOCHS};
const MIXER_BATCH_SIZE: usize = {MIXER_BATCH_SIZE};
const MIXER_TRAIN_POP_SUBSET: usize = {MIXER_TRAIN_POP_SUBSET};
const MIXER_MAX_TEACHER_PAIRS: usize = {MIXER_MAX_TEACHER_PAIRS};
const MIXER_MAX_MINIBATCHES: usize = {MIXER_MAX_MINIBATCHES};
const MIXER_LR_MATRIX: f32 = {MIXER_LR_MATRIX};
const MIXER_LR_VECTOR: f32 = {MIXER_LR_VECTOR};
const NEAREST_BETTER_CACHE_SIZE: usize = {NEAREST_BETTER_CACHE_SIZE};
const MUON_WEIGHT_DECAY: f32 = {MUON_WEIGHT_DECAY};
const MUON_MOMENTUM: f32 = {MUON_MOMENTUM};
const MUON_NS_STEPS: usize = {MUON_NS_STEPS};
const MUON_NESTEROV: bool = {MUON_NESTEROV};
const ADAMW_BETA1: f32 = {ADAMW_BETA1};
const ADAMW_BETA2: f32 = {ADAMW_BETA2};
const ADAMW_EPS: f32 = {ADAMW_EPS};
const ADAMW_WEIGHT_DECAY: f32 = {ADAMW_WEIGHT_DECAY};
const EVAL_PROGRESS_CHUNK: usize = 32;
const TRAIN_PROGRESS_EVERY: usize = 4;

// ─── カリキュラム学習 ────────────────────────────────────────────────────────
const CURRICULUM_RAMP_GENS: usize = 1;
const INTER_WEIGHT_MAX: f64 = 1.0;

// ─── adf_cache 上限 ──────────────────────────────────────────────────────────
const ADF_CACHE_MAX: usize = 1 << 18;

// ─── バッチ設定 ──────────────────────────────────────────────────────────────
const N_BATCHES: usize = 1;

{BODY}
"""

# ─── ここに元の関数群を埋め込むため、テンプレート末尾 {BODY} に挿入する ─────

RUST_BODY = r"""
// ─── レイヤー設定ヘルパー ─────────────────────────────────────────────────────

#[inline]
fn is_top(layer_idx: usize) -> bool {
    layer_idx == N_LAYERS - 1
}
#[inline]
fn layer_len(layer_idx: usize) -> usize {
    LAYER_LEN[layer_idx]
}
#[inline]
fn layer_n_ext(layer_idx: usize) -> usize {
    if is_top(layer_idx) { N_INPUTS_MAIN } else { N_INPUTS_ADF }
}
#[inline]
fn layer_n_adf(layer_idx: usize) -> usize {
    N_ADF_PER_LAYER[layer_idx]
}
#[inline]
fn layer_n_funcs(layer_idx: usize) -> usize {
    if layer_idx == 0 { 1 } else { 1 + layer_n_adf(layer_idx - 1) }
}

fn selectfunc(max: u8, rng: &mut SmallRng) -> u8 {
    if rng.gen::<f64>() < PROB_EML { return 0; }
    rng.gen_range(0..max)
}

// ─── 複素数ユーティリティ ─────────────────────────────────────────────────────

#[inline(always)]
fn clamp(z: Complex<f64>) -> Complex<f64> {
    const M: f64 = 15.0;
    Complex::new(z.re.clamp(-M, M), z.im.clamp(-M, M))
}
#[inline(always)]
fn safe_log(z: Complex<f64>) -> Complex<f64> {
    const EPS: f64 = 1e-6;
    if z.norm_sqr() < EPS * EPS { Complex::new(EPS, 0.0).ln() } else { z.ln() }
}
#[inline(always)]
fn eml(a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
    clamp(a).exp() - safe_log(b)
}
fn eml_vec(x: &[Complex<f64>], y: &[Complex<f64>]) -> Vec<Complex<f64>> {
    x.iter().zip(y).map(|(&a, &b)| eml(a, b)).collect()
}

// ─── Chromosome ───────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Chromosome {
    layer_idx: usize,
    conn: Box<[[u16; 2]]>,
    func: Box<[u8]>,
}

impl Chromosome {
    fn random(layer_idx: usize, rng: &mut SmallRng) -> Self {
        let n = layer_len(layer_idx);
        let n_ext = layer_n_ext(layer_idx);
        let n_f = layer_n_funcs(layer_idx) as u8;
        let conn = (0..n)
            .map(|i| { let m = (n_ext + i) as u16; [((1.0 - rng.gen::<f64>().powf(P)) * m as f64).floor() as u16, ((1.0 - rng.gen::<f64>().powf(P)) * m as f64).floor() as u16] })
            .collect::<Vec<_>>().into_boxed_slice();
        let func = (0..n).map(|_| selectfunc(n_f, rng)).collect::<Vec<_>>().into_boxed_slice();
        Self { layer_idx, conn, func }
    }

    fn active_and_sig(&self) -> (Vec<usize>, u64) {
        let n = layer_len(self.layer_idx);
        let n_ext = layer_n_ext(self.layer_idx);
        let total = n_ext + n;
        let mut active = vec![false; total];
        active[total - 1] = true;
        for i in (0..n).rev() {
            if !active[n_ext + i] { continue; }
            active[self.conn[i][0] as usize] = true;
            active[self.conn[i][1] as usize] = true;
        }
        let mut h = AHasher::default();
        (self.layer_idx as u8).hash(&mut h);
        let list: Vec<usize> = (0..total).filter(|&i| active[i])
            .inspect(|&abs| {
                if abs >= n_ext {
                    let i = abs - n_ext;
                    self.conn[i][0].hash(&mut h); self.conn[i][1].hash(&mut h);
                    self.func[i].hash(&mut h); abs.hash(&mut h);
                }
            }).collect();
        (list, h.finish())
    }

    fn mutate(&self, rng: &mut SmallRng) -> Self {
        let mut conn = self.conn.to_vec();
        let mut func = self.func.to_vec();
        let n = layer_len(self.layer_idx);
        let n_ext = layer_n_ext(self.layer_idx);
        let n_f = layer_n_funcs(self.layer_idx) as u8;
        // MUT_STOP_PROB: 対数一様分布の最大値。大きいほど多くのノードを変異させる。
        let mut n_mut = (rng.gen::<f64>() * MUT_STOP_PROB).exp() as usize + 1;
        for _ in 0..n_mut {
            let i = rng.gen_range(0..n);
            let max = (n_ext + i) as u16;
            match rng.gen_range(0..3u8) {
                0 => conn[i][0] = ((1.0 - rng.gen::<f64>().powf(P)) * max as f64).floor() as u16,
                1 => conn[i][1] = ((1.0 - rng.gen::<f64>().powf(P)) * max as f64).floor() as u16,
                _ => func[i] = selectfunc(n_f, rng),
            }
            if rng.gen::<f64>() < 0.05 {
                conn[i][rng.gen_range(0..2)] = rng.gen_range(0..n_ext as u16);
            }
        }
        Self { layer_idx: self.layer_idx, conn: conn.into_boxed_slice(), func: func.into_boxed_slice() }
    }

    fn mix(&self, rng: &mut SmallRng, other: &Chromosome) -> Self {
        let (active2, _) = other.active_and_sig();
        let n_ext = layer_n_ext(self.layer_idx);
        let mut conn = self.conn.to_vec();
        let mut func = self.func.to_vec();
        let mut cands: Vec<usize> = active2.into_iter().filter(|&i| i >= n_ext).collect();
        if !cands.is_empty() {
            let n_copy = rng.gen_range(1..=cands.len());
            for i in 0..n_copy.min(cands.len()) {
                let j = rng.gen_range(i..cands.len());
                cands.swap(i, j);
                let local = cands[i] - n_ext;
                conn[local] = other.conn[local];
                func[local] = other.func[local];
            }
        }
        Self { layer_idx: self.layer_idx, conn: conn.into_boxed_slice(), func: func.into_boxed_slice() }
    }
}

// ─── Genome ───────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Genome { layers: Vec<Vec<Chromosome>> }

impl Genome {
    fn random(rng: &mut SmallRng) -> Self {
        let layers = (0..N_LAYERS).map(|li| {
            let count = if is_top(li) { 1 } else { layer_n_adf(li) };
            (0..count).map(|_| Chromosome::random(li, rng)).collect()
        }).collect();
        Self { layers }
    }

    fn mutate(&self, rng: &mut SmallRng) -> Self {
        let mut g = self.clone();
        let totals: Vec<usize> = (0..N_LAYERS)
            .map(|li| if is_top(li) { 1 } else { layer_n_adf(li) }).collect();
        let grand_total: usize = totals.iter().sum();
        // MUT_MAX_TARGETS: 一度に変異させる Chromosome の最大数
        let n_targets = rng.gen_range(1..=MUT_MAX_TARGETS.min(grand_total));
        for _ in 0..n_targets {
            let mut t = rng.gen_range(0..grand_total);
            for (li, &cnt) in totals.iter().enumerate() {
                if t < cnt { g.layers[li][t] = self.layers[li][t].mutate(rng); break; }
                t -= cnt;
            }
        }
        g
    }

    fn mix(&self, rng: &mut SmallRng, other: &Genome) -> Self {
        let mut g = self.clone();
        for li in 0..N_LAYERS {
            let cnt = if is_top(li) { 1 } else { layer_n_adf(li) };
            for k in 0..cnt {
                if rng.gen::<f64>() < 0.5 {
                    g.layers[li][k] = self.layers[li][k].mix(rng, &other.layers[li][k]);
                }
            }
        }
        g
    }
}

// ─── Signature ────────────────────────────────────────────────────────────────

type Sig = u64;

#[inline]
fn make_sig(a: Sig, b: Sig, func_id: u8) -> Sig {
    let mut x = a ^ ((func_id as u64).wrapping_mul(0x9e3779b97f4a7c15));
    x ^= b.wrapping_mul(0x6c62272e07bb0142);
    x ^= x >> 30; x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn genome_key(all_sigs: &[Vec<Sig>]) -> u64 {
    let mut h = AHasher::default();
    for layer_sigs in all_sigs { for &s in layer_sigs { s.hash(&mut h); } }
    h.finish()
}

// ─── Cache ────────────────────────────────────────────────────────────────────

type ArcVec = Arc<[Complex<f64>]>;
/// ADF キャッシュのキー。
/// (batch_sig, c_sig, in0_sig, in1_sig, x_sig)
/// x_sig を加えることで、同じ ADF 構造でもバッチの x 値が異なれば別エントリになる。
type AdfKey = (Sig, Sig, Sig, Sig, Sig);
type Score = (f64, f64);

struct NodeBuf { sigs: Vec<Sig>, vals: Vec<ArcVec> }

impl NodeBuf {
    fn new(total: usize) -> Self {
        let dummy: ArcVec = Arc::from(vec![Complex::new(0.0, 0.0); VEC_LEN].into_boxed_slice());
        Self { sigs: vec![0u64; total], vals: vec![dummy; total] }
    }
    #[inline] fn set(&mut self, idx: usize, sig: Sig, val: ArcVec) { self.sigs[idx] = sig; self.vals[idx] = val; }
    #[inline] fn sig(&self, idx: usize) -> Sig { self.sigs[idx] }
    #[inline] fn val(&self, idx: usize) -> &ArcVec { &self.vals[idx] }
}

#[inline]
fn node_get_or_compute(sig: Sig, v0: &[Complex<f64>], v1: &[Complex<f64>], ev: &Evaluator) -> ArcVec {
    if let Some(v) = ev.node_cache.get(&sig) { return Arc::clone(&*v); }
    let out: ArcVec = Arc::from(eml_vec(v0, v1).into_boxed_slice());
    ev.node_cache.entry(sig).or_insert_with(|| Arc::clone(&out));
    out
}

// ─── Dataset ──────────────────────────────────────────────────────────────────

fn target_fn(x: Complex<f64>) -> Complex<f64> { (x * x * x - x).sin() * (x * x).sin()  }

fn make_batch(rng: &mut SmallRng, x_range: (f64, f64))
    -> ([Vec<Complex<f64>>; N_INPUTS_MAIN], Vec<Complex<f64>>)
{
    let (lo, hi) = x_range;
    let xs: Vec<Complex<f64>> = (0..VEC_LEN)
        .map(|_| Complex::new(lo + rng.gen::<f64>() * (hi - lo), 0.0)).collect();
    let one = vec![Complex::new(1.0, 0.0); VEC_LEN];
    let ys = xs.iter().map(|&x| target_fn(x)).collect();
    ([xs, one], ys)
}

struct Dataset {
    batches: Vec<([Vec<Complex<f64>>; N_INPUTS_MAIN], Vec<Complex<f64>>)>,
    batch_sig: Sig,
}

impl Dataset {
    /// 固定データセット: 起動時に一度だけ生成し、以降は変更しない。
    /// batch_sig も固定値になるため、世代をまたいだキャッシュが正しく機能する。
    fn new_fixed(rng: &mut SmallRng) -> Self {
        let batches: Vec<_> = (0..N_BATCHES).map(|_| make_batch(rng, (-HILO, HILO))).collect();
        let batch_sig = Self::compute_sig(&batches);
        Self { batches, batch_sig }
    }
    fn compute_sig(batches: &[([Vec<Complex<f64>>; N_INPUTS_MAIN], Vec<Complex<f64>>)]) -> Sig {
        let mut h = AHasher::default();
        for (inputs, _) in batches { for z in inputs[0].iter().take(8) { z.re.to_bits().hash(&mut h); } }
        h.finish()
    }
}

// ─── Evaluator ────────────────────────────────────────────────────────────────

struct Evaluator {
    node_cache: DashMap<Sig, ArcVec>,
    adf_cache: DashMap<AdfKey, (Sig, ArcVec)>,
    /// 固定データセット運用では fitness_cache はゲノムが同一なら永続的に有効。
    /// 世代をまたいで保持し、新規ゲノムのみ計算する。
    fitness_cache: DashMap<u64, Score>,
    score_cache: DashMap<(Sig, Sig), Score>,
}

impl Evaluator {
    fn new() -> Self {
        Self {
            node_cache: DashMap::with_capacity(1 << 17),
            adf_cache: DashMap::with_capacity(1 << 16),
            fitness_cache: DashMap::with_capacity(POP_SIZE * 8),
            score_cache: DashMap::with_capacity(1 << 17),
        }
    }

    /// 固定データセット運用向けリセット。
    /// - node_cache: メモリ節約のため世代ごとにクリア（再計算コストは小さい）。
    /// - fitness_cache / adf_cache / score_cache:
    ///   データセットが変わらないため保持。同一ゲノムは再評価不要。
    ///   ただし adf_cache がメモリ上限を超えたら解放する。
    fn reset_generation(&self) {
        self.node_cache.clear();
        if self.adf_cache.len() > ADF_CACHE_MAX {
            self.adf_cache.clear();
            self.score_cache.clear();
            // fitness_cache は adf_cache に依存しないので保持してよい
        }
    }

    fn get_node_score(&self, batch_sig: Sig, node_sig: Sig, val: &ArcVec, target: &[Complex<f64>],
        p_buf: &mut [f64], t_buf: &mut [f64], order_buf: &mut [usize], rank_buf: &mut [i64]) -> Score
    {
        let key = (batch_sig, node_sig);
        if let Some(s) = self.score_cache.get(&key) { return *s; }
        let s = score_and_acc_into(val, target, p_buf, t_buf, order_buf, rank_buf);
        self.score_cache.insert(key, s); s
    }
}

// ─── 評価指標 ─────────────────────────────────────────────────────────────────

fn pearson_raw(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut num, mut dx, mut dy) = (0.0, 0.0, 0.0f64);
    for (&xi, &yi) in x.iter().zip(y) {
        let a = xi - mx; let b = yi - my;
        num += a * b; dx += a * a; dy += b * b;
    }
    if dx < 1e-12 || dy < 1e-12 { 0.0 } else { num / (dx.sqrt() * dy.sqrt()) }
}

#[inline]
fn chatterjee_from_ranks(sorted_idx: &[usize], ranks: &[i64]) -> f64 {
    let s: i64 = sorted_idx.windows(2).map(|w| (ranks[w[0]] - ranks[w[1]]).abs()).sum();
    let n = sorted_idx.len() as f64;
    1.0 - 3.0 * s as f64 / (n * n - 1.0)
}

fn score_and_acc_into(pred: &[Complex<f64>], target: &[Complex<f64>],
    p_buf: &mut [f64], t_buf: &mut [f64], order_buf: &mut [usize], rank_buf: &mut [i64]) -> (f64, f64)
{
    let n = pred.len();
    let extractors: [fn(Complex<f64>) -> f64; 4] = [|z| z.re, |z| z.im, |z| z.norm(), |z| z.arg()];
    let mut total_score = 0.0f64;
    let mut max_pearson = 0.0f64;
    let mut p_rank_buf = vec![0i64; n];
    let mut t_order_buf = vec![0usize; n];
    for ext in extractors {
        for k in 0..n { p_buf[k] = ext(pred[k]); t_buf[k] = ext(target[k]); }
        for k in 0..n { order_buf[k] = k; }
        order_buf[..n].sort_unstable_by(|&a, &b| p_buf[a].partial_cmp(&p_buf[b]).unwrap_or(std::cmp::Ordering::Equal));
        for (r, &i) in order_buf[..n].iter().enumerate() { p_rank_buf[i] = r as i64; }
        for k in 0..n { t_order_buf[k] = k; }
        t_order_buf[..n].sort_unstable_by(|&a, &b| t_buf[a].partial_cmp(&t_buf[b]).unwrap_or(std::cmp::Ordering::Equal));
        for (r, &i) in t_order_buf[..n].iter().enumerate() { rank_buf[i] = r as i64; }
        let c_fwd = chatterjee_from_ranks(&order_buf[..n], rank_buf);
        let c_bwd = chatterjee_from_ranks(&t_order_buf[..n], &p_rank_buf);
        let pe = pearson_raw(&p_buf[..n], &t_buf[..n]).abs();
        total_score += (1.0 - ((((c_fwd.max(0.0).powf(C) + c_bwd.max(0.0).powf(C)) / 2.0) * (1.0 - D) + pe.powf(C) * D)).powf(1.0 / C)).powf(E);
        if pe > max_pearson { max_pearson = pe; }
    }
    (total_score / 4.0, max_pearson)
}

// ─── 汎用 ADF 実行 ────────────────────────────────────────────────────────────

/// ADF を実行する。
///
/// `x_sig` / `x_arc` はトップ層の入力 x（ext[0]）をそのまま伝播させたもの。
/// ADF 内部のスロット 3 (`X_GLOBAL_SIG_SLOT`) に配置され、
/// ノードの conn がスロット 3 を指せば x_global を直接参照できる（カンニング）。
/// 再帰呼び出しでも同じ x_sig/x_arc を渡すことで全層で共有される。
fn exec_adf(
    layer_idx: usize, c: &Chromosome, c_sig: Sig, active: &[usize],
    in0_sig: Sig, in0: &ArcVec, in1_sig: Sig, in1: &ArcVec, one: &ArcVec,
    x_sig: Sig, x_arc: &ArcVec,
    genome: &Genome, all_sigs: &[Vec<Sig>], all_acts: &[Vec<Vec<usize>>],
    batch_sig: Sig, ev: &Evaluator,
) -> (Sig, ArcVec) {
    // x_sig をキーに含めることで、同一構造でも x が変われば別エントリになる
    let key: AdfKey = (batch_sig, c_sig, in0_sig, in1_sig, x_sig);
    if let Some(v) = ev.adf_cache.get(&key) { return (v.0, Arc::clone(&v.1)); }
    let n_ext = N_INPUTS_ADF; // = 4: [in0, in1, one, x_global]
    let total = n_ext + layer_len(layer_idx);
    let mut buf = NodeBuf::new(total);
    buf.set(0, in0_sig,  Arc::clone(in0));
    buf.set(1, in1_sig,  Arc::clone(in1));
    buf.set(2, ONE_SIG,  Arc::clone(one));
    buf.set(3, x_sig,    Arc::clone(x_arc)); // ← x_global スロット
    for &abs in active {
        if abs < n_ext { continue; }
        let i = abs - n_ext;
        let c0 = c.conn[i][0] as usize;
        let c1 = c.conn[i][1] as usize;
        let s0 = buf.sig(c0); let s1 = buf.sig(c1);
        let f = c.func[i] as usize;
        let (sig, val) = if f == 0 || layer_idx == 0 {
            let sig = make_sig(s0, s1, 0);
            let val = node_get_or_compute(sig, buf.val(c0), buf.val(c1), ev);
            (sig, val)
        } else {
            let sub_li = layer_idx - 1;
            let sub_idx = (f - 1).min(layer_n_adf(sub_li) - 1);
            exec_adf(
                sub_li, &genome.layers[sub_li][sub_idx], all_sigs[sub_li][sub_idx],
                &all_acts[sub_li][sub_idx], s0, buf.val(c0), s1, buf.val(c1), one,
                x_sig, x_arc, // ← 再帰先にも x を伝播
                genome, all_sigs, all_acts, batch_sig, ev,
            )
        };
        buf.set(abs, sig, val);
    }
    let result = (buf.sig(total - 1), Arc::clone(buf.val(total - 1)));
    ev.adf_cache.entry(key).or_insert_with(|| result.clone());
    result
}

// ─── 最終出力層 実行 ──────────────────────────────────────────────────────────

fn exec_top(
    genome: &Genome, top_active: &[usize], all_sigs: &[Vec<Sig>], all_acts: &[Vec<Vec<usize>>],
    ext: &[Vec<Complex<f64>>; N_INPUTS_MAIN], target: &[Complex<f64>],
    batch_sig: Sig, ev: &Evaluator,
    p_buf: &mut [f64], t_buf: &mut [f64], order_buf: &mut [usize], rank_buf: &mut [i64],
) -> (ArcVec, Sig, f64, usize) {
    let top_li = N_LAYERS - 1;
    let c = &genome.layers[top_li][0];
    let n_ext = N_INPUTS_MAIN;
    let total = n_ext + layer_len(top_li);
    let mut buf = NodeBuf::new(total);
    for i in 0..n_ext { buf.set(i, i as Sig + 1, Arc::from(ext[i].as_slice())); }
    let one_arc: ArcVec = Arc::from(ext[1].as_slice());
    // x_global: トップ層の ext[0]（入力 x）を ADF に渡すための Arc と Sig
    // トップ層では buf.set(0, 1, ...) としているので x_sig = 1
    let x_arc: ArcVec = Arc::from(ext[0].as_slice());
    let x_sig: Sig = 1; // ext[0] に割り当てた固定 Sig
    let mut sum_score = 0.0; let mut count = 0;
    for &abs in top_active {
        if abs < n_ext { continue; }
        let i = abs - n_ext;
        let c0 = c.conn[i][0] as usize; let c1 = c.conn[i][1] as usize;
        let s0 = buf.sig(c0); let s1 = buf.sig(c1);
        let f = c.func[i] as usize;
        let (sig, val) = if f == 0 || top_li == 0 {
            let sig = make_sig(s0, s1, 0);
            let val = node_get_or_compute(sig, buf.val(c0), buf.val(c1), ev);
            (sig, val)
        } else {
            let sub_li = top_li - 1;
            let sub_idx = (f - 1).min(layer_n_adf(sub_li) - 1);
            exec_adf(
                sub_li, &genome.layers[sub_li][sub_idx], all_sigs[sub_li][sub_idx],
                &all_acts[sub_li][sub_idx], s0, buf.val(c0), s1, buf.val(c1), &one_arc,
                x_sig, &x_arc, // ← x_global を ADF に注入
                genome, all_sigs, all_acts, batch_sig, ev,
            )
        };
        let (s, _) = ev.get_node_score(batch_sig, sig, &val, target, p_buf, t_buf, order_buf, rank_buf);
        sum_score += s; count += 1;
        buf.set(abs, sig, val);
    }
    let out_sig = buf.sig(total - 1);
    (Arc::clone(buf.val(total - 1)), out_sig, sum_score, count)
}

// ─── Fitness 評価 ─────────────────────────────────────────────────────────────

fn eval(g: &Genome, ds: &Dataset, ev: &Evaluator, inter_weight: f64) -> (f64, f64) {
    let top_li = N_LAYERS - 1;
    let all_data: Vec<Vec<(Vec<usize>, Sig)>> = (0..N_LAYERS)
        .map(|li| g.layers[li].iter().map(|c| c.active_and_sig()).collect()).collect();
    let all_sigs: Vec<Vec<Sig>> = all_data.iter()
        .map(|layer| layer.iter().map(|d| d.1).collect()).collect();
    let all_acts: Vec<Vec<Vec<usize>>> = all_data.into_iter()
        .map(|layer| layer.into_iter().map(|d| d.0).collect()).collect();
    let top_active = &all_acts[top_li][0];
    let genome_base = genome_key(&all_sigs);
    let top_sig = all_sigs[top_li][0];
    // 固定データセット: batch_sig は不変なのでキャッシュキーとして安全に使える。
    // inter_weight もキーに含める（世代が異なると値が変わるため）。
    let gkey = make_sig(make_sig(genome_base, ds.batch_sig, 0),
                        inter_weight.to_bits(), 1);
    if let Some(v) = ev.fitness_cache.get(&gkey) { return *v; }
    let mut p_buf = vec![0.0f64; VEC_LEN];
    let mut t_buf = vec![0.0f64; VEC_LEN];
    let mut order_buf = vec![0usize; VEC_LEN];
    let mut rank_buf = vec![0i64; VEC_LEN];
    let mut total_loss = 0.0; let mut total_acc = 0.0;
    for (inputs, target) in &ds.batches {
        let (pred, out_sig, sum_score, count) = exec_top(
            g, top_active, &all_sigs, &all_acts, inputs, target, ds.batch_sig, ev,
            &mut p_buf, &mut t_buf, &mut order_buf, &mut rank_buf);
        let (final_s, final_a) = ev.get_node_score(ds.batch_sig, out_sig, &pred, target,
            &mut p_buf, &mut t_buf, &mut order_buf, &mut rank_buf);
        let avg_inter = if count > 0 { (sum_score / count as f64).powf(1.0 / E) } else { 0.0 };
        let combined = avg_inter;
        total_loss += ((combined).powf(A) * B + (1.0 - final_a).powf(A) * (1.0 - B)).powf(1.0 / A);
        total_acc += final_a;
    }
    let _ = top_sig;
    let n = ds.batches.len() as f64;
    let res = (total_loss / n, total_acc / n);
    ev.fitness_cache.insert(gkey, res); res
}

// ============================================================
// learned mutation model (from main(3).rs)
// ============================================================

// ============================================================
// math helpers
// ============================================================

fn gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
}

fn gelu_derivative_scalar(x: f32) -> f32 {
    let u = 0.7978845608 * (x + 0.044715 * x.powi(3));
    let t = u.tanh();
    let sech2 = 1.0 - t * t;
    0.5 * (1.0 + t) + 0.5 * x * sech2 * 0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x)
}

fn sigmoid_scalar(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn softmax_rows(x: &Array2<f32>) -> Array2<f32> {
    let mut out = x.clone();
    for mut row in out.axis_iter_mut(Axis(0)) {
        let maxv = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - maxv).exp());
        let sum = row.sum();
        row.mapv_inplace(|v| v / sum);
    }
    out
}

// ============================================================
// optimizer helpers: Muon for 2D, AdamW for 1D
// ============================================================

#[derive(Clone, Copy)]
struct OptimHyper {
    // Muon for 2D matrix parameters
    lr_matrix: f32,
    muon_weight_decay: f32,
    muon_momentum: f32,
    muon_ns_steps: usize,
    muon_nesterov: bool,

    // AdamW for 1D vector parameters
    lr_vector: f32,
    adamw_beta1: f32,
    adamw_beta2: f32,
    adamw_eps: f32,
    adamw_weight_decay: f32,
}

fn fro_norm(x: &Array2<f32>) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn zeropower_via_newtonschulz5(g: &Array2<f32>, steps: usize) -> Array2<f32> {
    let (orig_r, orig_c) = g.dim();
    let mut x = if orig_r > orig_c {
        g.t().to_owned()
    } else {
        g.clone()
    };

    let norm = fro_norm(&x);
    if norm > 0.0 {
        x.mapv_inplace(|v| v / (norm + 1e-7));
    }

    let a = 3.4445_f32;
    let b = -4.7750_f32;
    let c = 2.0315_f32;

    for _ in 0..steps {
        let xx_t = x.dot(&x.t());
        let xx_t2 = xx_t.dot(&xx_t);
        let bmat = xx_t.mapv(|v| b * v) + xx_t2.mapv(|v| c * v);
        x = x.mapv(|v| a * v) + bmat.dot(&x);
    }

    let mut out = if orig_r > orig_c { x.t().to_owned() } else { x };
    let scale = (1.0_f32).max(orig_r as f32 / orig_c as f32).sqrt();
    out.mapv_inplace(|v| v * scale);
    out
}

fn muon_update_2d(
    grad: &Array2<f32>,
    momentum_buf: &mut Array2<f32>,
    beta: f32,
    ns_steps: usize,
    nesterov: bool,
) -> Array2<f32> {
    *momentum_buf = momentum_buf.mapv(|v| beta * v) + grad.mapv(|v| (1.0 - beta) * v);

    let raw = if nesterov {
        grad.mapv(|v| (1.0 - beta) * v) + momentum_buf.mapv(|v| beta * v)
    } else {
        momentum_buf.clone()
    };

    zeropower_via_newtonschulz5(&raw, ns_steps)
}

fn adamw_update_1d(
    param: &mut Array1<f32>,
    grad: &Array1<f32>,
    m: &mut Array1<f32>,
    v: &mut Array1<f32>,
    step: &mut usize,
    hp: OptimHyper,
) {
    *step += 1;
    let t = *step as i32;

    param.mapv_inplace(|x| x * (1.0 - hp.lr_vector * hp.adamw_weight_decay));

    *m = m.mapv(|x| x * hp.adamw_beta1) + grad.mapv(|g| (1.0 - hp.adamw_beta1) * g);
    *v = v.mapv(|x| x * hp.adamw_beta2) + grad.mapv(|g| (1.0 - hp.adamw_beta2) * g * g);

    let m_hat = m.mapv(|x| x / (1.0 - hp.adamw_beta1.powi(t)));
    let v_hat = v.mapv(|x| x / (1.0 - hp.adamw_beta2.powi(t)));

    for i in 0..param.len() {
        param[i] -= hp.lr_vector * m_hat[i] / (v_hat[i].sqrt() + hp.adamw_eps);
    }
}

// ============================================================
// trainable parameter wrappers
// ============================================================

#[derive(Clone)]
struct MatrixParam {
    w: Array2<f32>,
    gw: Array2<f32>,
    mom: Array2<f32>,
}

impl MatrixParam {
    fn new(rng: &mut StdRng, rows: usize, cols: usize, scale: f32) -> Self {
        let mut w = Array2::<f32>::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let z: f32 = StandardNormal.sample(rng);
                w[[i, j]] = z * scale;
            }
        }
        Self {
            w,
            gw: Array2::zeros((rows, cols)),
            mom: Array2::zeros((rows, cols)),
        }
    }

    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            w: Array2::zeros((rows, cols)),
            gw: Array2::zeros((rows, cols)),
            mom: Array2::zeros((rows, cols)),
        }
    }

    fn zero_grad(&mut self) {
        self.gw.fill(0.0);
    }

    fn scale_grad(&mut self, s: f32) {
        self.gw.mapv_inplace(|v| v * s);
    }

    fn step_muon(&mut self, hp: OptimHyper) {
        let upd = muon_update_2d(
            &self.gw,
            &mut self.mom,
            hp.muon_momentum,
            hp.muon_ns_steps,
            hp.muon_nesterov,
        );
        self.w
            .mapv_inplace(|v| v * (1.0 - hp.lr_matrix * hp.muon_weight_decay));
        self.w -= &upd.mapv(|v| hp.lr_matrix * v);
    }
}

#[derive(Clone)]
struct VectorParam {
    w: Array1<f32>,
    gw: Array1<f32>,
    m: Array1<f32>,
    v: Array1<f32>,
    step: usize,
}

impl VectorParam {
    fn new(size: usize) -> Self {
        Self {
            w: Array1::zeros(size),
            gw: Array1::zeros(size),
            m: Array1::zeros(size),
            v: Array1::zeros(size),
            step: 0,
        }
    }

    fn zero_grad(&mut self) {
        self.gw.fill(0.0);
    }

    fn scale_grad(&mut self, s: f32) {
        self.gw.mapv_inplace(|v| v * s);
    }

    fn step_adamw(&mut self, hp: OptimHyper) {
        adamw_update_1d(
            &mut self.w,
            &self.gw,
            &mut self.m,
            &mut self.v,
            &mut self.step,
            hp,
        );
    }
}

// ============================================================
// Linear layer
// ============================================================

#[derive(Clone)]
struct Linear {
    w: MatrixParam, // [in_dim, out_dim]
    b: VectorParam, // [out_dim]
}

#[derive(Clone)]
struct LinearCache {
    x: Array2<f32>, // [batch, in_dim]
}

impl Linear {
    fn new(rng: &mut StdRng, in_dim: usize, out_dim: usize, scale: f32) -> Self {
        Self {
            w: MatrixParam::new(rng, in_dim, out_dim, scale),
            b: VectorParam::new(out_dim),
        }
    }

    fn zero_grad(&mut self) {
        self.w.zero_grad();
        self.b.zero_grad();
    }

    fn scale_grad(&mut self, s: f32) {
        self.w.scale_grad(s);
        self.b.scale_grad(s);
    }

    fn step(&mut self, hp: OptimHyper) {
        self.w.step_muon(hp);
        self.b.step_adamw(hp);
    }

    fn forward(&self, x: &Array2<f32>) -> (Array2<f32>, LinearCache) {
        let y = x.dot(&self.w.w) + &self.b.w;
        (y, LinearCache { x: x.clone() })
    }

    fn backward(&mut self, cache: &LinearCache, dy: &Array2<f32>) -> Array2<f32> {
        self.w.gw += &cache.x.t().dot(dy);
        self.b.gw += &dy.sum_axis(Axis(0));
        dy.dot(&self.w.w.t())
    }
}

// ============================================================
// Bilinear map: Y = L X R^T
// ============================================================

#[derive(Clone)]
struct BilinearMap {
    l: MatrixParam, // [out_r, in_r]
    r: MatrixParam, // [out_c, in_c]
}

#[derive(Clone)]
struct BilinearCache {
    x: Array2<f32>,  // [in_r, in_c]
    xr: Array2<f32>, // [in_r, out_c]
}

impl BilinearMap {
    fn new(
        rng: &mut StdRng,
        in_r: usize,
        in_c: usize,
        out_r: usize,
        out_c: usize,
        scale: f32,
    ) -> Self {
        Self {
            l: MatrixParam::new(rng, out_r, in_r, scale),
            r: MatrixParam::new(rng, out_c, in_c, scale),
        }
    }

    fn zero_grad(&mut self) {
        self.l.zero_grad();
        self.r.zero_grad();
    }

    fn scale_grad(&mut self, s: f32) {
        self.l.scale_grad(s);
        self.r.scale_grad(s);
    }

    fn step(&mut self, hp: OptimHyper) {
        self.l.step_muon(hp);
        self.r.step_muon(hp);
    }

    fn forward(&self, x: &Array2<f32>) -> (Array2<f32>, BilinearCache) {
        let xr = x.dot(&self.r.w.t());
        let y = self.l.w.dot(&xr);
        (y, BilinearCache { x: x.clone(), xr })
    }

    fn backward(&mut self, cache: &BilinearCache, dy: &Array2<f32>) -> Array2<f32> {
        self.l.gw += &dy.dot(&cache.xr.t());
        let dxr = self.l.w.t().dot(dy);
        self.r.gw += &dxr.t().dot(&cache.x);
        dxr.dot(&self.r.w)
    }
}

// ============================================================
// condition embedding
// ============================================================

#[derive(Clone)]
enum CondEncodingKind {
    RawVector,
    SinCosScalar,
}

#[derive(Clone)]
struct CondEmbedding {
    kind: CondEncodingKind,
    dim: usize,
    max_period: f32,
}

#[derive(Clone)]
struct CondEmbeddingCache {
    cond: Array1<f32>,
    emb: Array1<f32>,
}

impl CondEmbedding {
    fn new_raw_vec(dim: usize) -> Self {
        assert!(dim >= 1);
        Self {
            kind: CondEncodingKind::RawVector,
            dim,
            max_period: 10000.0,
        }
    }

    fn new_sincos(dim: usize, max_period: f32) -> Self {
        assert!(dim >= 2);
        Self {
            kind: CondEncodingKind::SinCosScalar,
            dim,
            max_period,
        }
    }

    fn output_dim(&self) -> usize {
        self.dim
    }

    fn forward(&self, cond: &Array1<f32>) -> (Array1<f32>, CondEmbeddingCache) {
        let emb = match self.kind {
            CondEncodingKind::RawVector => {
                assert_eq!(cond.len(), self.dim);
                cond.clone()
            }
            CondEncodingKind::SinCosScalar => {
                assert!(
                    !cond.is_empty(),
                    "SinCosScalar condition embedding expects at least one scalar input"
                );
                let scalar = cond[0];
                let half = self.dim / 2;
                let mut out = Array1::<f32>::zeros(self.dim);

                for i in 0..half {
                    let frac = i as f32 / half as f32;
                    let freq = self.max_period.powf(-frac);
                    let x = scalar * freq;
                    out[i] = x.sin();
                    out[half + i] = x.cos();
                }

                if self.dim % 2 == 1 {
                    out[self.dim - 1] = 0.0;
                }
                out
            }
        };

        (emb.clone(), CondEmbeddingCache { cond: cond.clone(), emb })
    }

    fn backward(&self, cache: &CondEmbeddingCache, demb: &Array1<f32>) -> Array1<f32> {
        match self.kind {
            CondEncodingKind::RawVector => {
                assert_eq!(demb.len(), cache.cond.len());
                demb.clone()
            }
            CondEncodingKind::SinCosScalar => {
                let mut dcond = Array1::<f32>::zeros(cache.cond.len());
                let scalar = cache.cond[0];
                let half = self.dim / 2;
                let mut grad0 = 0.0;
                for i in 0..half {
                    let frac = i as f32 / half as f32;
                    let freq = self.max_period.powf(-frac);
                    let x = scalar * freq;
                    grad0 += demb[i] * x.cos() * freq;
                    grad0 += demb[half + i] * (-x.sin()) * freq;
                }
                dcond[0] = grad0;
                dcond
            }
        }
    }
}

// ============================================================
// per-block condition projection
// cond_emb -> hidden -> cond_vec
// ============================================================

#[derive(Clone)]
struct CondProjector {
    fc1: Linear,
    fc2: Linear,
}

#[derive(Clone)]
struct CondProjectorCache {
    z1: Array2<f32>, // [1, hidden]
    fc1_cache: LinearCache,
    fc2_cache: LinearCache,
}

impl CondProjector {
    fn new(rng: &mut StdRng, in_dim: usize, hidden_dim: usize, out_dim: usize, scale: f32) -> Self {
        Self {
            fc1: Linear::new(rng, in_dim, hidden_dim, scale),
            fc2: Linear::new(rng, hidden_dim, out_dim, scale),
        }
    }

    fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
    }

    fn scale_grad(&mut self, s: f32) {
        self.fc1.scale_grad(s);
        self.fc2.scale_grad(s);
    }

    fn step(&mut self, hp: OptimHyper) {
        self.fc1.step(hp);
        self.fc2.step(hp);
    }

    fn forward(&self, emb: &Array1<f32>) -> (Array1<f32>, CondProjectorCache) {
        let x = emb.clone().insert_axis(Axis(0)); // [1, in_dim]
        let (z1, fc1_cache) = self.fc1.forward(&x);
        let a1 = z1.mapv(gelu_scalar);
        let (y, fc2_cache) = self.fc2.forward(&a1);
        (
            y.row(0).to_owned(),
            CondProjectorCache {
                z1,
                fc1_cache,
                fc2_cache,
            },
        )
    }

    fn backward(&mut self, cache: &CondProjectorCache, dy: &Array1<f32>) -> Array1<f32> {
        let dy2 = dy.clone().insert_axis(Axis(0)); // [1, out_dim]
        let da1 = self.fc2.backward(&cache.fc2_cache, &dy2);

        let mut dz1 = da1.clone();
        for ((i, j), v) in dz1.indexed_iter_mut() {
            *v *= gelu_derivative_scalar(cache.z1[[i, j]]);
        }

        let dx = self.fc1.backward(&cache.fc1_cache, &dz1);
        dx.row(0).to_owned()
    }
}

// ============================================================
// AdaLN with vector condition
// y = LN(x) * (1 + gamma(cond_vec)) + beta(cond_vec)
// ============================================================

#[derive(Clone)]
struct AdaLn {
    eps: f32,
    cond_dim: usize,

    gamma_w: MatrixParam, // [channels, cond_dim]
    gamma_b: VectorParam, // [channels]
    beta_w: MatrixParam,  // [channels, cond_dim]
    beta_b: VectorParam,  // [channels]
}

#[derive(Clone)]
struct AdaLnCache {
    xhat: Array2<f32>,
    inv_std: Array1<f32>,
    gamma: Array1<f32>,
    cond_vec: Array1<f32>,
}

impl AdaLn {
    fn new(channels: usize, cond_dim: usize) -> Self {
        Self {
            eps: 1e-5,
            cond_dim,
            gamma_w: MatrixParam::zeros(channels, cond_dim),
            gamma_b: VectorParam::new(channels),
            beta_w: MatrixParam::zeros(channels, cond_dim),
            beta_b: VectorParam::new(channels),
        }
    }

    fn zero_grad(&mut self) {
        self.gamma_w.zero_grad();
        self.gamma_b.zero_grad();
        self.beta_w.zero_grad();
        self.beta_b.zero_grad();
    }

    fn scale_grad(&mut self, s: f32) {
        self.gamma_w.scale_grad(s);
        self.gamma_b.scale_grad(s);
        self.beta_w.scale_grad(s);
        self.beta_b.scale_grad(s);
    }

    fn step(&mut self, hp: OptimHyper) {
        self.gamma_w.step_muon(hp);
        self.gamma_b.step_adamw(hp);
        self.beta_w.step_muon(hp);
        self.beta_b.step_adamw(hp);
    }

    fn forward(&self, x: &Array2<f32>, cond_vec: &Array1<f32>) -> (Array2<f32>, AdaLnCache) {
        assert_eq!(cond_vec.len(), self.cond_dim);

        let t = x.nrows();
        let c = x.ncols();

        let mut xhat = Array2::<f32>::zeros((t, c));
        let mut inv_std = Array1::<f32>::zeros(t);

        for i in 0..t {
            let row = x.slice(s![i, ..]);
            let mu = row.sum() / c as f32;
            let var = row
                .iter()
                .map(|&v| {
                    let d = v - mu;
                    d * d
                })
                .sum::<f32>()
                / c as f32;
            let istd = 1.0 / (var + self.eps).sqrt();
            inv_std[i] = istd;

            for j in 0..c {
                xhat[[i, j]] = (x[[i, j]] - mu) * istd;
            }
        }

        let gamma = self.gamma_w.w.dot(cond_vec) + &self.gamma_b.w;
        let beta = self.beta_w.w.dot(cond_vec) + &self.beta_b.w;

        let mut y = Array2::<f32>::zeros((t, c));
        for i in 0..t {
            for j in 0..c {
                y[[i, j]] = xhat[[i, j]] * (1.0 + gamma[j]) + beta[j];
            }
        }

        (
            y,
            AdaLnCache {
                xhat,
                inv_std,
                gamma,
                cond_vec: cond_vec.clone(),
            },
        )
    }

    fn backward(&mut self, cache: &AdaLnCache, dy: &Array2<f32>) -> (Array2<f32>, Array1<f32>) {
        let t = dy.nrows();
        let c = dy.ncols();

        let mut dgamma = Array1::<f32>::zeros(c);
        let mut dbeta = Array1::<f32>::zeros(c);
        let mut dxhat = Array2::<f32>::zeros((t, c));

        for i in 0..t {
            for j in 0..c {
                dgamma[j] += dy[[i, j]] * cache.xhat[[i, j]];
                dbeta[j] += dy[[i, j]];
                dxhat[[i, j]] = dy[[i, j]] * (1.0 + cache.gamma[j]);
            }
        }

        self.gamma_w.gw += &dgamma
            .view()
            .insert_axis(Axis(1))
            .dot(&cache.cond_vec.view().insert_axis(Axis(0)));
        self.gamma_b.gw += &dgamma;

        self.beta_w.gw += &dbeta
            .view()
            .insert_axis(Axis(1))
            .dot(&cache.cond_vec.view().insert_axis(Axis(0)));
        self.beta_b.gw += &dbeta;

        let dcond_vec = self.gamma_w.w.t().dot(&dgamma) + self.beta_w.w.t().dot(&dbeta);

        let mut dx = Array2::<f32>::zeros((t, c));
        for i in 0..t {
            let xhat_row = cache.xhat.slice(s![i, ..]);
            let dxhat_row = dxhat.slice(s![i, ..]);

            let sum1 = dxhat_row.sum();
            let sum2 = dxhat_row
                .iter()
                .zip(xhat_row.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>();

            for j in 0..c {
                dx[[i, j]] = (1.0 / c as f32)
                    * cache.inv_std[i]
                    * ((c as f32) * dxhat[[i, j]] - sum1 - xhat_row[j] * sum2);
            }
        }

        (dx, dcond_vec)
    }
}

// ============================================================
// MLP
// ============================================================

#[derive(Clone)]
struct Mlp2D {
    fc1: Linear,
    fc2: Linear,
}

#[derive(Clone)]
struct Mlp2DCache {
    z1: Array2<f32>,
    fc1_cache: LinearCache,
    fc2_cache: LinearCache,
}

impl Mlp2D {
    fn new(rng: &mut StdRng, in_dim: usize, hidden_dim: usize, scale: f32) -> Self {
        Self {
            fc1: Linear::new(rng, in_dim, hidden_dim, scale),
            fc2: Linear::new(rng, hidden_dim, in_dim, scale),
        }
    }

    fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
    }

    fn scale_grad(&mut self, s: f32) {
        self.fc1.scale_grad(s);
        self.fc2.scale_grad(s);
    }

    fn step(&mut self, hp: OptimHyper) {
        self.fc1.step(hp);
        self.fc2.step(hp);
    }

    fn forward(&self, x: &Array2<f32>) -> (Array2<f32>, Mlp2DCache) {
        let (z1, fc1_cache) = self.fc1.forward(x);
        let a1 = z1.mapv(gelu_scalar);
        let (y, fc2_cache) = self.fc2.forward(&a1);
        (
            y,
            Mlp2DCache {
                z1,
                fc1_cache,
                fc2_cache,
            },
        )
    }

    fn backward(&mut self, cache: &Mlp2DCache, dy: &Array2<f32>) -> Array2<f32> {
        let da1 = self.fc2.backward(&cache.fc2_cache, dy);
        let mut dz1 = da1.clone();
        for ((i, j), v) in dz1.indexed_iter_mut() {
            *v *= gelu_derivative_scalar(cache.z1[[i, j]]);
        }
        self.fc1.backward(&cache.fc1_cache, &dz1)
    }
}

// ============================================================
// single-head attention
// ============================================================

#[derive(Clone)]
struct SelfAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    scale: f32,
}

#[derive(Clone)]
struct SelfAttentionCache {
    q: Array2<f32>,
    k: Array2<f32>,
    v: Array2<f32>,
    attn: Array2<f32>,
    q_cache: LinearCache,
    k_cache: LinearCache,
    v_cache: LinearCache,
    o_cache: LinearCache,
}

impl SelfAttention {
    fn new(rng: &mut StdRng, channels: usize, scale: f32) -> Self {
        Self {
            wq: Linear::new(rng, channels, channels, scale),
            wk: Linear::new(rng, channels, channels, scale),
            wv: Linear::new(rng, channels, channels, scale),
            wo: Linear::new(rng, channels, channels, scale),
            scale: (channels as f32).sqrt(),
        }
    }

    fn zero_grad(&mut self) {
        self.wq.zero_grad();
        self.wk.zero_grad();
        self.wv.zero_grad();
        self.wo.zero_grad();
    }

    fn scale_grad(&mut self, s: f32) {
        self.wq.scale_grad(s);
        self.wk.scale_grad(s);
        self.wv.scale_grad(s);
        self.wo.scale_grad(s);
    }

    fn step(&mut self, hp: OptimHyper) {
        self.wq.step(hp);
        self.wk.step(hp);
        self.wv.step(hp);
        self.wo.step(hp);
    }

    fn forward(&self, x: &Array2<f32>) -> (Array2<f32>, SelfAttentionCache) {
        let (q, q_cache) = self.wq.forward(x);
        let (k, k_cache) = self.wk.forward(x);
        let (v, v_cache) = self.wv.forward(x);

        let scores = q.dot(&k.t()) / self.scale;
        let attn = softmax_rows(&scores);
        let ctx = attn.dot(&v);
        let (y, o_cache) = self.wo.forward(&ctx);

        (
            y,
            SelfAttentionCache {
                q,
                k,
                v,
                attn,
                q_cache,
                k_cache,
                v_cache,
                o_cache,
            },
        )
    }

    fn backward(&mut self, cache: &SelfAttentionCache, dy: &Array2<f32>) -> Array2<f32> {
        let dctx = self.wo.backward(&cache.o_cache, dy);

        let dattn = dctx.dot(&cache.v.t());
        let dv = cache.attn.t().dot(&dctx);

        let t = cache.attn.nrows();
        let mut dscores = Array2::<f32>::zeros((t, t));

        for i in 0..t {
            let a = cache.attn.slice(s![i, ..]);
            let da = dattn.slice(s![i, ..]);
            let dot = a
                .iter()
                .zip(da.iter())
                .map(|(&ai, &dai)| ai * dai)
                .sum::<f32>();

            for j in 0..t {
                dscores[[i, j]] = a[j] * (da[j] - dot);
            }
        }

        let dq = dscores.dot(&cache.k) / self.scale;
        let dk = dscores.t().dot(&cache.q) / self.scale;

        let dxq = self.wq.backward(&cache.q_cache, &dq);
        let dxk = self.wk.backward(&cache.k_cache, &dk);
        let dxv = self.wv.backward(&cache.v_cache, &dv);

        dxq + dxk + dxv
    }
}

// ============================================================
// Mixer block
// ============================================================

#[derive(Clone)]
struct MixerBlock {
    cond_proj: CondProjector,
    ln_attn: AdaLn,
    attn: SelfAttention,
    ln_tok: AdaLn,
    tok_mlp: Mlp2D,
    ln_ch: AdaLn,
    ch_mlp: Mlp2D,
}

#[derive(Clone)]
struct MixerBlockCache {
    cond_proj_cache: CondProjectorCache,
    ln_attn_cache: AdaLnCache,
    attn_cache: SelfAttentionCache,
    ln_tok_cache: AdaLnCache,
    tok_mlp_cache: Mlp2DCache,
    ln_ch_cache: AdaLnCache,
    ch_mlp_cache: Mlp2DCache,
}

impl MixerBlock {
    fn new(
        rng: &mut StdRng,
        tokens: usize,
        channels: usize,
        token_hidden: usize,
        channel_hidden: usize,
        cond_emb_dim: usize,
        cond_proj_hidden: usize,
        cond_vec_dim: usize,
        scale: f32,
    ) -> Self {
        Self {
            cond_proj: CondProjector::new(rng, cond_emb_dim, cond_proj_hidden, cond_vec_dim, scale),
            ln_attn: AdaLn::new(channels, cond_vec_dim),
            attn: SelfAttention::new(rng, channels, scale),
            ln_tok: AdaLn::new(channels, cond_vec_dim),
            tok_mlp: Mlp2D::new(rng, tokens, token_hidden, scale),
            ln_ch: AdaLn::new(channels, cond_vec_dim),
            ch_mlp: Mlp2D::new(rng, channels, channel_hidden, scale),
        }
    }

    fn zero_grad(&mut self) {
        self.cond_proj.zero_grad();
        self.ln_attn.zero_grad();
        self.attn.zero_grad();
        self.ln_tok.zero_grad();
        self.tok_mlp.zero_grad();
        self.ln_ch.zero_grad();
        self.ch_mlp.zero_grad();
    }

    fn scale_grad(&mut self, s: f32) {
        self.cond_proj.scale_grad(s);
        self.ln_attn.scale_grad(s);
        self.attn.scale_grad(s);
        self.ln_tok.scale_grad(s);
        self.tok_mlp.scale_grad(s);
        self.ln_ch.scale_grad(s);
        self.ch_mlp.scale_grad(s);
    }

    fn step(&mut self, hp: OptimHyper) {
        self.cond_proj.step(hp);
        self.ln_attn.step(hp);
        self.attn.step(hp);
        self.ln_tok.step(hp);
        self.tok_mlp.step(hp);
        self.ln_ch.step(hp);
        self.ch_mlp.step(hp);
    }

    fn forward(&self, x: &Array2<f32>, cond_emb: &Array1<f32>) -> (Array2<f32>, MixerBlockCache) {
        let (cond_vec, cond_proj_cache) = self.cond_proj.forward(cond_emb);

        let (attn_norm, ln_attn_cache) = self.ln_attn.forward(x, &cond_vec);
        let (attn_out, attn_cache) = self.attn.forward(&attn_norm);
        let x1 = x + &attn_out;

        let (tok_norm, ln_tok_cache) = self.ln_tok.forward(&x1, &cond_vec);
        let tok_in = tok_norm.t().to_owned();
        let (tok_out_t, tok_mlp_cache) = self.tok_mlp.forward(&tok_in);
        let tok_out = tok_out_t.t().to_owned();
        let x2 = &x1 + &tok_out;

        let (ch_norm, ln_ch_cache) = self.ln_ch.forward(&x2, &cond_vec);
        let (ch_out, ch_mlp_cache) = self.ch_mlp.forward(&ch_norm);
        let y = &x2 + &ch_out;

        (
            y,
            MixerBlockCache {
                cond_proj_cache,
                ln_attn_cache,
                attn_cache,
                ln_tok_cache,
                tok_mlp_cache,
                ln_ch_cache,
                ch_mlp_cache,
            },
        )
    }

    fn backward(
        &mut self,
        cache: &MixerBlockCache,
        dy: &Array2<f32>,
    ) -> (Array2<f32>, Array1<f32>) {
        let cond_vec_dim = cache.ln_attn_cache.cond_vec.len();
        let mut dcond_vec = Array1::<f32>::zeros(cond_vec_dim);

        let dch_out = dy.clone();
        let mut dx2 = dy.clone();
        let dch_norm = self.ch_mlp.backward(&cache.ch_mlp_cache, &dch_out);
        let (dx2_from_ch, dc3) = self.ln_ch.backward(&cache.ln_ch_cache, &dch_norm);
        dcond_vec += &dc3;
        dx2 += &dx2_from_ch;

        let dtok_out = dx2.clone();
        let mut dx1 = dx2.clone();
        let dtok_out_t = dtok_out.t().to_owned();
        let dtok_in = self.tok_mlp.backward(&cache.tok_mlp_cache, &dtok_out_t);
        let dtok_norm = dtok_in.t().to_owned();
        let (dx1_from_tok, dc2) = self.ln_tok.backward(&cache.ln_tok_cache, &dtok_norm);
        dcond_vec += &dc2;
        dx1 += &dx1_from_tok;

        let dattn_out = dx1.clone();
        let mut dx0 = dx1.clone();
        let dattn_norm = self.attn.backward(&cache.attn_cache, &dattn_out);
        let (dx0_from_attn, dc1) = self.ln_attn.backward(&cache.ln_attn_cache, &dattn_norm);
        dcond_vec += &dc1;
        dx0 += &dx0_from_attn;

        let dcond_emb = self.cond_proj.backward(&cache.cond_proj_cache, &dcond_vec);
        (dx0, dcond_emb)
    }
}


// ============================================================
// sample structs
// ============================================================

#[derive(Clone)]
struct Sample {
    a_mats: Vec<Array2<f32>>, // each is [A, 2*(A+N_INPUTS_ADF)]
    b_mat: Array2<f32>,       // [B, 2*(B+N_INPUTS_MAIN)]
    cond: Array1<f32>,
}

#[derive(Clone)]
struct Targets {
    a_targets: Vec<Array2<f32>>, // each is [A, 2*(A+N_INPUTS_ADF)]
    b_target: Array2<f32>,       // [B, 2*(B+N_INPUTS_MAIN)]
}

struct ModelOutput {
    a_logits: Vec<Array2<f32>>,
    b_logits: Array2<f32>,
}

struct ForwardCache {
    enc_a_caches: Vec<BilinearCache>,
    enc_b_cache: BilinearCache,
    cond_cache: CondEmbeddingCache,
    block_caches: Vec<MixerBlockCache>,
    dec_a_caches: Vec<BilinearCache>,
    dec_b_cache: BilinearCache,
}

// ============================================================
// full model
// ============================================================

struct MixerModel {
    num_a: usize,
    a_rows: usize,
    a_cols: usize,
    b_rows: usize,
    b_cols: usize,
    d1: usize,
    d2: usize,

    enc_a: BilinearMap,
    dec_a: BilinearMap,
    enc_b: BilinearMap,
    dec_b: BilinearMap,

    cond_embed: CondEmbedding,
    blocks: Vec<MixerBlock>,
}

impl MixerModel {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rng: &mut StdRng,
        num_a: usize,
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
        d1: usize,
        d2: usize,
        num_blocks: usize,
        token_hidden: usize,
        channel_hidden: usize,
        cond_embed: CondEmbedding,
        cond_proj_hidden: usize,
        cond_vec_dim: usize,
    ) -> Self {
        let scale_a = (2.0 / (a_rows + d1 + a_cols + d2) as f32).sqrt();
        let scale_b = (2.0 / (b_rows + d1 + b_cols + d2) as f32).sqrt();
        let scale_inner = 0.05;

        let enc_a = BilinearMap::new(rng, a_rows, a_cols, d1, d2, scale_a);
        let dec_a = BilinearMap::new(rng, d1, d2, a_rows, a_cols, scale_a);
        let enc_b = BilinearMap::new(rng, b_rows, b_cols, d1, d2, scale_b);
        let dec_b = BilinearMap::new(rng, d1, d2, b_rows, b_cols, scale_b);

        let tokens = (num_a + 1) * d1;
        let cond_emb_dim = cond_embed.output_dim();

        let mut blocks = Vec::new();
        for _ in 0..num_blocks {
            blocks.push(MixerBlock::new(
                rng,
                tokens,
                d2,
                token_hidden,
                channel_hidden,
                cond_emb_dim,
                cond_proj_hidden,
                cond_vec_dim,
                scale_inner,
            ));
        }

        Self {
            num_a,
            a_rows,
            a_cols,
            b_rows,
            b_cols,
            d1,
            d2,
            enc_a,
            dec_a,
            enc_b,
            dec_b,
            cond_embed,
            blocks,
        }
    }

    fn zero_grad(&mut self) {
        self.enc_a.zero_grad();
        self.dec_a.zero_grad();
        self.enc_b.zero_grad();
        self.dec_b.zero_grad();
        for b in &mut self.blocks {
            b.zero_grad();
        }
    }

    fn scale_gradients(&mut self, s: f32) {
        self.enc_a.scale_grad(s);
        self.dec_a.scale_grad(s);
        self.enc_b.scale_grad(s);
        self.dec_b.scale_grad(s);
        for b in &mut self.blocks {
            b.scale_grad(s);
        }
    }

    fn step(&mut self, hp: OptimHyper) {
        self.enc_a.step(hp);
        self.dec_a.step(hp);
        self.enc_b.step(hp);
        self.dec_b.step(hp);
        for b in &mut self.blocks {
            b.step(hp);
        }
    }

    fn forward(&self, sample: &Sample) -> (ModelOutput, ForwardCache) {
        assert_eq!(sample.a_mats.len(), self.num_a);

        let mut parts: Vec<Array2<f32>> = Vec::new();
        let mut enc_a_caches = Vec::new();

        for x in &sample.a_mats {
            let (blk, cache) = self.enc_a.forward(x);
            parts.push(blk);
            enc_a_caches.push(cache);
        }

        let (b_blk, enc_b_cache) = self.enc_b.forward(&sample.b_mat);
        parts.push(b_blk);

        let total_tokens = (self.num_a + 1) * self.d1;
        let mut z = Array2::<f32>::zeros((total_tokens, self.d2));
        for (i, part) in parts.iter().enumerate() {
            let start = i * self.d1;
            let end = start + self.d1;
            z.slice_mut(s![start..end, ..]).assign(part);
        }

        let (cond_emb, cond_cache) = self.cond_embed.forward(&sample.cond);

        let mut block_caches = Vec::new();
        for block in &self.blocks {
            let (next, cache) = block.forward(&z, &cond_emb);
            z = next;
            block_caches.push(cache);
        }

        let mut a_logits = Vec::new();
        let mut dec_a_caches = Vec::new();

        for i in 0..self.num_a {
            let start = i * self.d1;
            let end = start + self.d1;
            let blk = z.slice(s![start..end, ..]).to_owned();
            let (mat, cache) = self.dec_a.forward(&blk);
            a_logits.push(mat);
            dec_a_caches.push(cache);
        }

        let start = self.num_a * self.d1;
        let end = start + self.d1;
        let blk_b = z.slice(s![start..end, ..]).to_owned();
        let (b_logits, dec_b_cache) = self.dec_b.forward(&blk_b);

        (
            ModelOutput { a_logits, b_logits },
            ForwardCache {
                enc_a_caches,
                enc_b_cache,
                cond_cache,
                block_caches,
                dec_a_caches,
                dec_b_cache,
            },
        )
    }

    fn bce_with_logits_loss(logits: &Array2<f32>, targets: &Array2<f32>) -> (f32, Array2<f32>) {
        let mut loss = 0.0;
        let mut grad = Array2::<f32>::zeros(logits.raw_dim());
        let n = logits.len() as f32;

        for ((i, j), &z) in logits.indexed_iter() {
            let t = targets[[i, j]];
            loss += z.max(0.0) - z * t + (1.0 + (-z.abs()).exp()).ln();
            grad[[i, j]] = (sigmoid_scalar(z) - t) / n;
        }

        (loss / n, grad)
    }

    fn loss_and_backward_accumulate(
        &mut self,
        sample: &Sample,
        target: &Targets,
    ) -> (f32, ModelOutput) {
        let (output, cache) = self.forward(sample);

        let mut total_loss = 0.0;
        let mut dz = Array2::<f32>::zeros(((self.num_a + 1) * self.d1, self.d2));

        for i in 0..self.num_a {
            let (loss_i, dlogits) =
                Self::bce_with_logits_loss(&output.a_logits[i], &target.a_targets[i]);
            total_loss += loss_i;

            let dblk = self.dec_a.backward(&cache.dec_a_caches[i], &dlogits);
            let start = i * self.d1;
            let end = start + self.d1;
            let mut sl = dz.slice_mut(s![start..end, ..]);
            sl += &dblk;
        }

        let (loss_b, dlogits_b) = Self::bce_with_logits_loss(&output.b_logits, &target.b_target);
        total_loss += loss_b;

        let dblk_b = self.dec_b.backward(&cache.dec_b_cache, &dlogits_b);
        let start_b = self.num_a * self.d1;
        let end_b = start_b + self.d1;
        {
            let mut sl = dz.slice_mut(s![start_b..end_b, ..]);
            sl += &dblk_b;
        }

        let mut dcond_emb = Array1::<f32>::zeros(self.cond_embed.output_dim());

        for i in (0..self.blocks.len()).rev() {
            let (next_dz, dc) = self.blocks[i].backward(&cache.block_caches[i], &dz);
            dz = next_dz;
            dcond_emb += &dc;
        }

        let _dcond_vec = self.cond_embed.backward(&cache.cond_cache, &dcond_emb);

        for i in 0..self.num_a {
            let start = i * self.d1;
            let end = start + self.d1;
            let dblk = dz.slice(s![start..end, ..]).to_owned();
            let _dx = self.enc_a.backward(&cache.enc_a_caches[i], &dblk);
        }

        let dblk_b2 = dz.slice(s![start_b..end_b, ..]).to_owned();
        let _dx_b = self.enc_b.backward(&cache.enc_b_cache, &dblk_b2);

        let denom = (self.num_a + 1) as f32;
        (total_loss / denom, output)
    }

    fn train_minibatch_step(&mut self, batch: &[(Sample, Targets)], hp: OptimHyper) -> f32 {
        self.zero_grad();

        let mut loss_sum = 0.0;
        for (sample, target) in batch {
            let (loss, _) = self.loss_and_backward_accumulate(sample, target);
            loss_sum += loss;
        }

        let inv_bs = 1.0 / batch.len() as f32;
        self.scale_gradients(inv_bs);
        self.step(hp);

        loss_sum * inv_bs
    }

    fn predict(&self, sample: &Sample) -> ModelOutput {
        self.forward(sample).0
    }

    fn predict_probs(&self, sample: &Sample) -> (Vec<Array2<f32>>, Array2<f32>) {
        let out = self.predict(sample);
        let as_ = out
            .a_logits
            .into_iter()
            .map(|x| x.mapv(sigmoid_scalar))
            .collect::<Vec<_>>();
        let b = out.b_logits.mapv(sigmoid_scalar);
        (as_, b)
    }
}

// ============================================================
// hybrid controller glue
// ============================================================

#[derive(Debug, Clone)]
pub struct NearestBetterResult {
    /// ans[i] = score が自分より高く、かつハミング距離最小の個体 index
    /// 候補が無ければ None
    pub ans: Vec<Option<usize>>,
}

#[inline]
fn freq_bucket(f: usize) -> u8 {
    if f == 0 {
        0
    } else {
        (usize::BITS - f.leading_zeros()) as u8
    }
}

/// main(2).rs の pruning / cached order を、そのまま
/// 「各座標ごとに語彙サイズが違う」一般形へ拡張した版。
///
/// arrs[k][i] は 0..alphabet_sizes[i]-1 にある必要がある。
pub fn nearest_better_hamming_pruned_generalized(
    arrs: &[Vec<usize>],
    scores: &[f64],
    alphabet_sizes: &[usize],
    max_cache_size: usize,
) -> NearestBetterResult {
    use std::collections::HashMap;

    let n = arrs.len();
    assert_eq!(n, scores.len(), "arrs.len() must equal scores.len()");
    if n == 0 {
        return NearestBetterResult { ans: vec![] };
    }

    let l = arrs[0].len();
    assert_eq!(l, alphabet_sizes.len(), "alphabet_sizes.len() must equal code length");
    for a in arrs.iter() {
        assert_eq!(a.len(), l, "all arrays must have the same length");
    }
    for x in arrs.iter() {
        for (i, &v) in x.iter().enumerate() {
            assert!(
                v < alphabet_sizes[i],
                "arr value out of range: x[{}]={}, expected 0..{}",
                i,
                v,
                alphabet_sizes[i]
            );
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        scores[j]
            .partial_cmp(&scores[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ans: Vec<Option<usize>> = vec![None; n];

    let mut postings: Vec<Vec<Vec<usize>>> = alphabet_sizes
        .iter()
        .map(|&k| vec![Vec::<usize>::new(); k])
        .collect();
    let mut freqs: Vec<Vec<usize>> = alphabet_sizes
        .iter()
        .map(|&k| vec![0usize; k])
        .collect();

    let mut count: Vec<usize> = vec![0; n];
    let mut seen_ver: Vec<u32> = vec![0; n];
    let mut active_mark: Vec<u32> = vec![0; n];
    let mut query_ver: u32 = 0;

    let mut order_cache: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();

    let get_coord_order = |x: &[usize],
                           freqs: &Vec<Vec<usize>>,
                           order_cache: &mut HashMap<Vec<u8>, Vec<usize>>|
     -> Vec<usize> {
        let mut signature = Vec::with_capacity(l);
        for i in 0..l {
            signature.push(freq_bucket(freqs[i][x[i]]));
        }

        if let Some(cached) = order_cache.get(&signature) {
            return cached.clone();
        }

        let mut coords: Vec<usize> = (0..l).collect();
        coords.sort_by_key(|&i| (signature[i], freqs[i][x[i]], i));

        if order_cache.len() >= max_cache_size {
            order_cache.clear();
        }
        order_cache.insert(signature, coords.clone());
        coords
    };

    let query_one = |x: &[usize],
                     postings: &Vec<Vec<Vec<usize>>>,
                     freqs: &Vec<Vec<usize>>,
                     order_cache: &mut HashMap<Vec<u8>, Vec<usize>>,
                     count: &mut Vec<usize>,
                     seen_ver: &mut Vec<u32>,
                     active_mark: &mut Vec<u32>,
                     query_ver: &mut u32|
     -> Option<usize> {
        *query_ver += 1;
        let ver = *query_ver;
        let coords = get_coord_order(x, freqs, order_cache);

        let mut best_id: Option<usize> = None;
        let mut best_match: usize = 0;
        let mut active: Vec<usize> = Vec::new();

        for (step, &i) in coords.iter().enumerate() {
            let plist = &postings[i][x[i]];
            let mut improved = false;

            for &j in plist.iter() {
                if seen_ver[j] != ver {
                    seen_ver[j] = ver;
                    count[j] = 0;
                    if active_mark[j] != ver {
                        active_mark[j] = ver;
                        active.push(j);
                    }
                }

                count[j] += 1;
                if count[j] > best_match {
                    best_match = count[j];
                    best_id = Some(j);
                    improved = true;
                }
            }

            if best_match == l {
                break;
            }

            let rem = l - (step + 1);
            if improved {
                let mut new_active = Vec::with_capacity(active.len());
                for &j in active.iter() {
                    if count[j] + rem >= best_match {
                        new_active.push(j);
                    }
                }
                active = new_active;
                if best_id.is_some() && active.is_empty() {
                    break;
                }
            }
        }

        best_id
    };

    let mut p = 0;
    while p < n {
        let s = scores[order[p]];
        let mut q = p + 1;
        while q < n && scores[order[q]] == s {
            q += 1;
        }

        let group = &order[p..q];
        for &idx in group.iter() {
            ans[idx] = query_one(
                &arrs[idx],
                &postings,
                &freqs,
                &mut order_cache,
                &mut count,
                &mut seen_ver,
                &mut active_mark,
                &mut query_ver,
            );
        }

        for &idx in group.iter() {
            let x = &arrs[idx];
            for i in 0..l {
                let v = x[i];
                postings[i][v].push(idx);
                freqs[i][v] += 1;
            }
        }

        p = q;
    }

    NearestBetterResult { ans }
}

#[derive(Clone, Debug)]
pub struct HybridMutationConfig {
    pub learned_mutation_prob: f64,
    pub train_every_generations: usize,
    pub train_epochs_per_generation: usize,
    pub batch_size: usize,
    pub nearest_better_cache_size: usize,
    pub train_population_subset: usize,
    pub max_teacher_pairs: usize,
    pub max_minibatches_per_generation: usize,
}

impl Default for HybridMutationConfig {
    fn default() -> Self {
        Self {
            learned_mutation_prob: 0.15,
            train_every_generations: 1,
            train_epochs_per_generation: 1,
            batch_size: 8,
            nearest_better_cache_size: 1 << 14,
            train_population_subset: 256,
            max_teacher_pairs: 256,
            max_minibatches_per_generation: 16,
        }
    }
}

#[derive(Clone)]
struct ChromosomeRef {
    layer_idx: usize,
    chrom_idx: usize,
}

/// learned mutator では connection topology のみを学習し、
/// func スロットは classical mutation に残す。
#[derive(Clone)]
pub struct GenomeCodec {
    adf_order: Vec<ChromosomeRef>,
    adf_len: usize,
    top_len: usize,
    adf_domain_cap: usize,
    top_domain_cap: usize,
    alphabet_sizes: Vec<usize>,
}

impl GenomeCodec {
    pub fn new() -> Self {
        let mut adf_order = Vec::new();
        let mut alphabet_sizes = Vec::new();

        let top_layer = N_LAYERS - 1;
        let top_len = layer_len(top_layer);
        let top_domain_cap = N_INPUTS_MAIN + top_len;

        let adf_len = if N_LAYERS >= 2 { layer_len(0) } else { 0 };
        let adf_domain_cap = if N_LAYERS >= 2 { N_INPUTS_ADF + adf_len } else { 0 };

        for li in 0..top_layer {
            let count = layer_n_adf(li);
            let len = layer_len(li);
            let n_ext = layer_n_ext(li);
            assert_eq!(len, adf_len, "all non-top chromosomes must share the same length");
            for ci in 0..count {
                adf_order.push(ChromosomeRef { layer_idx: li, chrom_idx: ci });
                for i in 0..len {
                    alphabet_sizes.push(n_ext + i);
                    alphabet_sizes.push(n_ext + i);
                }
            }
        }

        for i in 0..top_len {
            alphabet_sizes.push(N_INPUTS_MAIN + i);
            alphabet_sizes.push(N_INPUTS_MAIN + i);
        }

        Self {
            adf_order,
            adf_len,
            top_len,
            adf_domain_cap,
            top_domain_cap,
            alphabet_sizes,
        }
    }

    pub fn num_adf_mats(&self) -> usize {
        self.adf_order.len()
    }

    pub fn adf_shape(&self) -> (usize, usize) {
        (self.adf_len, 2 * self.adf_domain_cap)
    }

    pub fn top_shape(&self) -> (usize, usize) {
        (self.top_len, 2 * self.top_domain_cap)
    }

    pub fn alphabet_sizes(&self) -> &[usize] {
        &self.alphabet_sizes
    }

    pub fn flatten_discrete(&self, g: &Genome) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.alphabet_sizes.len());
        for r in &self.adf_order {
            let c = &g.layers[r.layer_idx][r.chrom_idx];
            let len = layer_len(r.layer_idx);
            for i in 0..len {
                out.push(c.conn[i][0] as usize);
                out.push(c.conn[i][1] as usize);
            }
        }

        let top = &g.layers[N_LAYERS - 1][0];
        for i in 0..self.top_len {
            out.push(top.conn[i][0] as usize);
            out.push(top.conn[i][1] as usize);
        }
        out
    }

    fn encode_chromosome(c: &Chromosome, len: usize, domain_cap: usize) -> Array2<f32> {
        let mut mat = Array2::<f32>::zeros((len, 2 * domain_cap));
        for i in 0..len {
            let c0 = c.conn[i][0] as usize;
            let c1 = c.conn[i][1] as usize;
            debug_assert!(c0 < domain_cap);
            debug_assert!(c1 < domain_cap);
            mat[[i, c0]] = 1.0;
            mat[[i, domain_cap + c1]] = 1.0;
        }
        mat
    }

    pub fn encode_sample(&self, g: &Genome, loss_cond: f32) -> Sample {
        let mut a_mats = Vec::with_capacity(self.adf_order.len());
        for r in &self.adf_order {
            let c = &g.layers[r.layer_idx][r.chrom_idx];
            a_mats.push(Self::encode_chromosome(c, self.adf_len, self.adf_domain_cap));
        }
        let top = &g.layers[N_LAYERS - 1][0];
        let b_mat = Self::encode_chromosome(top, self.top_len, self.top_domain_cap);
        Sample { a_mats, b_mat, cond: Array1::from_vec(vec![loss_cond]) }
    }

    pub fn encode_target(&self, g: &Genome) -> Targets {
        let sample = self.encode_sample(g, 0.0);
        Targets {
            a_targets: sample.a_mats,
            b_target: sample.b_mat,
        }
    }

    fn argmax_range(row: ndarray::ArrayView1<'_, f32>, start: usize, len: usize) -> usize {
        let mut best_idx = start;
        let mut best_val = f32::NEG_INFINITY;
        for j in start..(start + len) {
            let v = row[j];
            if v > best_val {
                best_val = v;
                best_idx = j;
            }
        }
        best_idx - start
    }

    fn decode_chromosome_into(
        c: &mut Chromosome,
        logits: &Array2<f32>,
        len: usize,
        domain_cap: usize,
        n_ext: usize,
    ) {
        for i in 0..len {
            let valid_conn = n_ext + i;
            let row = logits.index_axis(Axis(0), i);
            c.conn[i][0] = Self::argmax_range(row.view(), 0, valid_conn) as u16;
            c.conn[i][1] = Self::argmax_range(row.view(), domain_cap, valid_conn) as u16;
        }
    }

    pub fn decode_output(&self, template: &Genome, out: &ModelOutput) -> Genome {
        let mut g = template.clone();

        for (mat_idx, r) in self.adf_order.iter().enumerate() {
            let li = r.layer_idx;
            let c = &mut g.layers[li][r.chrom_idx];
            Self::decode_chromosome_into(
                c,
                &out.a_logits[mat_idx],
                self.adf_len,
                self.adf_domain_cap,
                layer_n_ext(li),
            );
        }

        let top = &mut g.layers[N_LAYERS - 1][0];
        Self::decode_chromosome_into(
            top,
            &out.b_logits,
            self.top_len,
            self.top_domain_cap,
            N_INPUTS_MAIN,
        );
        g
    }
}

pub struct HybridMutationController {
    pub codec: GenomeCodec,
    pub model: MixerModel,
    pub optim: OptimHyper,
    pub cfg: HybridMutationConfig,
    trained_pairs_last: usize,
}

impl HybridMutationController {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rng: &mut rand::rngs::StdRng,
        cfg: HybridMutationConfig,
        d1: usize,
        d2: usize,
        num_blocks: usize,
        token_hidden: usize,
        channel_hidden: usize,
        cond_embed: CondEmbedding,
        cond_proj_hidden: usize,
        cond_vec_dim: usize,
        optim: OptimHyper,
    ) -> Self {
        let codec = GenomeCodec::new();
        let (a_rows, a_cols) = codec.adf_shape();
        let (b_rows, b_cols) = codec.top_shape();
        let model = MixerModel::new(
            rng,
            codec.num_adf_mats(),
            a_rows,
            a_cols,
            b_rows,
            b_cols,
            d1,
            d2,
            num_blocks,
            token_hidden,
            channel_hidden,
            cond_embed,
            cond_proj_hidden,
            cond_vec_dim,
        );
        Self {
            codec,
            model,
            optim,
            cfg,
            trained_pairs_last: 0,
        }
    }

    pub fn trained_pairs_last(&self) -> usize {
        self.trained_pairs_last
    }

    fn loss_stats(losses: &[f64]) -> (f32, f32) {
        if losses.is_empty() {
            return (0.0, 1.0);
        }
        let mean = losses.iter().sum::<f64>() / losses.len() as f64;
        let var = losses
            .iter()
            .map(|&x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / losses.len() as f64;
        let std = var.sqrt().max(1e-8);
        (mean as f32, std as f32)
    }

    pub fn normalize_loss(loss: f64, mean: f32, std: f32) -> f32 {
        (((loss as f32) - mean) / std).tanh()
    }

    pub fn build_teacher_pairs(
        &self,
        genomes: &[Genome],
        losses: &[f64],
        rng: &mut rand::rngs::StdRng,
    ) -> Vec<(Sample, Targets)> {
        if genomes.is_empty() {
            return Vec::new();
        }

        let subset_cap = self.cfg.train_population_subset.max(2).min(genomes.len());
        let elite_take = (subset_cap / 2).max(1).min(genomes.len());
        let mut idxs: Vec<usize> = (0..elite_take).collect();
        if subset_cap > elite_take {
            let mut tail: Vec<usize> = (elite_take..genomes.len()).collect();
            tail.shuffle(rng);
            tail.truncate(subset_cap - elite_take);
            idxs.extend(tail);
            idxs.sort_unstable();
        }

        let sub_genomes: Vec<Genome> = idxs.iter().map(|&i| genomes[i].clone()).collect();
        let sub_losses: Vec<f64> = idxs.iter().map(|&i| losses[i]).collect();
        let codes: Vec<Vec<usize>> = sub_genomes
            .iter()
            .map(|g| self.codec.flatten_discrete(g))
            .collect();
        let scores: Vec<f64> = sub_losses.iter().map(|&x| -x).collect();

        let nb = nearest_better_hamming_pruned_generalized(
            &codes,
            &scores,
            self.codec.alphabet_sizes(),
            self.cfg.nearest_better_cache_size,
        );

        let (loss_mean, loss_std) = Self::loss_stats(&sub_losses);

        let best_loss = sub_losses
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        let mut out = Vec::new();
        for (i, maybe_j) in nb.ans.into_iter().enumerate() {
            if (sub_losses[i] - best_loss).abs() <= 1e-15 {
                continue;
            }
            if let Some(j) = maybe_j {
                if j == i {
                    continue;
                }
                if codes[i] == codes[j] {
                    continue;
                }
                let loss_cond = Self::normalize_loss(sub_losses[i], loss_mean, loss_std);
                out.push((
                    self.codec.encode_sample(&sub_genomes[i], loss_cond),
                    self.codec.encode_target(&sub_genomes[j]),
                ));
            }
        }
        if out.len() > self.cfg.max_teacher_pairs {
            out.shuffle(rng);
            out.truncate(self.cfg.max_teacher_pairs);
        }
        out
    }

    pub fn train_on_population(
        &mut self,
        genomes: &[Genome],
        losses: &[f64],
        rng: &mut rand::rngs::StdRng,
    ) {

        let mut pairs = self.build_teacher_pairs(genomes, losses, rng);
        self.trained_pairs_last = pairs.len();
        if pairs.is_empty() {
            return;
        }

        let mut steps = 0usize;
        let max_steps = self.cfg.max_minibatches_per_generation.max(1);
        for epoch in 0..self.cfg.train_epochs_per_generation {
            pairs.shuffle(rng);
            for batch in pairs.chunks(self.cfg.batch_size.max(1)) {
                self.model.train_minibatch_step(batch, self.optim);
                steps += 1;
                if steps % TRAIN_PROGRESS_EVERY == 0 || steps == max_steps {
                    println!(
                        "CMAES_STAGE stage=train_progress epoch={} step={} max_steps={} pairs={}",
                        epoch,
                        steps,
                        max_steps,
                        pairs.len(),
                    );
                }
                if steps >= max_steps {
                    return;
                }
            }
        }
    }

    pub fn propose_from_model(&self, parent: &Genome, loss_cond: f32) -> Genome {
        let sample = self.codec.encode_sample(parent, loss_cond);
        let out = self.model.predict(&sample);
        self.codec.decode_output(parent, &out)
    }

    pub fn maybe_mutate_with_model(
        &self,
        rng: &mut rand::rngs::SmallRng,
        parent: &Genome,
        fallback: Genome,
        loss_cond: f32,
    ) -> Genome {
        if rng.gen::<f64>() < self.cfg.learned_mutation_prob {
            self.propose_from_model(parent, loss_cond)
        } else {
            fallback
        }
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    assert_eq!(LAYER_LEN.len(), N_LAYERS, "LAYER_LEN の要素数は N_LAYERS に合わせてください");
    assert_eq!(N_ADF_PER_LAYER.len(), N_LAYERS - 1, "N_ADF_PER_LAYER の要素数は N_LAYERS-1 に合わせてください");
    assert!(N_LAYERS >= 1, "N_LAYERS は 1 以上にしてください");

    eprintln!(
        "ADF-CGP  N_LAYERS={N_LAYERS}  N_ADF_PER_LAYER={:?}  LAYER_LEN={:?}  POP={POP_SIZE}  BATCHES={N_BATCHES}  MUT_STOP_PROB={MUT_STOP_PROB}  MUT_MAX_TARGETS={MUT_MAX_TARGETS}  LEARNED_MUT_PROB={LEARNED_MUTATION_PROB}",
        N_ADF_PER_LAYER, LAYER_LEN,
    );

    let mut rng = SmallRng::seed_from_u64(42);
    let mut mlp_rng = StdRng::seed_from_u64(123456789);

    let ds = Dataset::new_fixed(&mut rng);
    let mut pop: Vec<Genome> = (0..POP_SIZE).map(|_| Genome::random(&mut rng)).collect();
    let evaluator = Evaluator::new();

    let hybrid_cfg = HybridMutationConfig {
        learned_mutation_prob: LEARNED_MUTATION_PROB,
        train_every_generations: MIXER_TRAIN_EVERY,
        train_epochs_per_generation: MIXER_TRAIN_EPOCHS,
        batch_size: MIXER_BATCH_SIZE,
        nearest_better_cache_size: NEAREST_BETTER_CACHE_SIZE,
        train_population_subset: MIXER_TRAIN_POP_SUBSET,
        max_teacher_pairs: MIXER_MAX_TEACHER_PAIRS,
        max_minibatches_per_generation: MIXER_MAX_MINIBATCHES,
    };

    let mixer_optim = OptimHyper {
        lr_matrix: MIXER_LR_MATRIX,
        muon_weight_decay: MUON_WEIGHT_DECAY,
        muon_momentum: MUON_MOMENTUM,
        muon_ns_steps: MUON_NS_STEPS,
        muon_nesterov: MUON_NESTEROV,
        lr_vector: MIXER_LR_VECTOR,
        adamw_beta1: ADAMW_BETA1,
        adamw_beta2: ADAMW_BETA2,
        adamw_eps: ADAMW_EPS,
        adamw_weight_decay: ADAMW_WEIGHT_DECAY,
    };

    let cond_embed = CondEmbedding::new_raw_vec(1);
    let mut learned_mutator = HybridMutationController::new(
        &mut mlp_rng,
        hybrid_cfg,
        MIXER_D1,
        MIXER_D2,
        MIXER_BLOCKS,
        MIXER_TOKEN_HIDDEN,
        MIXER_CHANNEL_HIDDEN,
        cond_embed,
        MIXER_COND_PROJ_HIDDEN,
        MIXER_COND_VEC_DIM,
        mixer_optim,
    );

    for gen in 0..N_GEN {
        println!("CMAES_STAGE gen={} stage=eval_begin", gen);
        evaluator.reset_generation();
        let inter_weight = INTER_WEIGHT_MAX * (gen as f64 / CURRICULUM_RAMP_GENS as f64).min(1.0);

        let eval_chunk = EVAL_PROGRESS_CHUNK.max(1);
        let mut scored: Vec<(f64, f64, Genome)> = Vec::with_capacity(pop.len());
        for (chunk_idx, chunk) in pop.chunks(eval_chunk).enumerate() {
            let mut part: Vec<(f64, f64, Genome)> = chunk
                .par_iter()
                .map(|g| { let (l, a) = eval(g, &ds, &evaluator, inter_weight); (l, a, g.clone()) })
                .collect();
            scored.append(&mut part);
        }
        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let losses: Vec<f64> = scored.iter().map(|x| x.0).collect();
        let genomes_for_training: Vec<Genome> = scored.iter().map(|x| x.2.clone()).collect();
        println!(
            "CMAES_ACC gen={} loss={:.10} acc={:.10}",
            gen, scored[0].0, scored[0].1
        );
        if MIXER_TRAIN_EVERY > 0 && gen % MIXER_TRAIN_EVERY == 0 {
            println!("CMAES_STAGE gen={} stage=train_begin", gen);
            learned_mutator.train_on_population(
                &genomes_for_training,
                &losses,
                &mut mlp_rng,
            );
            println!(
                "CMAES_STAGE gen={} stage=train_end teacher_pairs={} subset={} max_pairs={} max_steps={}",
                gen,
                learned_mutator.trained_pairs_last(),
                MIXER_TRAIN_POP_SUBSET,
                MIXER_MAX_TEACHER_PAIRS,
                MIXER_MAX_MINIBATCHES,
            );
        } else {
            learned_mutator.trained_pairs_last = 0;
        }

        let elites: Vec<Genome> = scored[..ELITE].iter().map(|x| x.2.clone()).collect();
        let (loss_mean, loss_std) = HybridMutationController::loss_stats(&losses);
        let mut next = elites;

        while next.len() < POP_SIZE {
            let i1 = rng.gen_range(0..ELITE);
            let i2 = rng.gen_range(0..ELITE);
            let p1 = &scored[i1].2;
            let p2 = &scored[i2].2;
            let p1_loss = scored[i1].0;

            let classical = match rng.gen_range(0..3u8) {
                0 => p1.mix(&mut rng, p2),
                1 => p1.mutate(&mut rng),
                _ => p1.mix(&mut rng, p2).mutate(&mut rng),
            };

            let loss_cond = HybridMutationController::normalize_loss(p1_loss, loss_mean, loss_std);
            let child = learned_mutator.maybe_mutate_with_model(
                &mut rng,
                p1,
                classical,
                loss_cond,
            );

            next.push(child);
        }
        pop = next;
    }
}

"""


def generate_rust_source(params: dict) -> str:
    """パラメータ辞書から Rust ソースを生成する"""
    n = params["N_LAYERS"]
    llen = params["LAYER_LEN"]
    llen2 = params["LAYER_LEN_LAST"]
    nadf = params["N_ADF_PER_LAYER"]

    layer_len_arr = ", ".join([str(llen)] * (n - 1) + [str(llen2)])

    if n == 1:
        n_adf_arr = ""
    else:
        n_adf_arr = ", ".join([str(nadf)] * (n - 1))

    elite = params["ELITE"]
    pop_size = params["POP_SIZE"]
    elite = min(elite, pop_size // 2)
    elite = max(elite, 2)

    src = RUST_TEMPLATE.format(
        N_LAYERS=n,
        LAYER_LEN_ARR=layer_len_arr,
        N_ADF_ARR=n_adf_arr,
        VEC_LEN=params["VEC_LEN"],
        POP_SIZE=pop_size,
        ELITE=elite,
        PROB_EML=f"{params['PROB_EML']:.8f}",
        A=f"{params['A']:.8f}",
        B=f"{params['B']:.8f}",
        C=f"{params['C']:.8f}",
        D=f"{params['D']:.8f}",
        E=f"{params['E']:.8f}",
        HILO=f"{params['HILO']:.8f}",
        P=f"{params['P']:.8f}",
        MUT_STOP_PROB=f"{params['MUT_STOP_PROB']:.8f}",
        MUT_MAX_TARGETS=int(params["MUT_MAX_TARGETS"]),
        LEARNED_MUTATION_PROB=f"{params['LEARNED_MUTATION_PROB']:.8f}",
        NEAREST_BETTER_CACHE_SIZE=int(params["NEAREST_BETTER_CACHE_SIZE"]),
        MIXER_D1=int(params["MIXER_D1"]),
        MIXER_D2=int(params["MIXER_D2"]),
        MIXER_BLOCKS=int(params["MIXER_BLOCKS"]),
        MIXER_TOKEN_HIDDEN=int(params["MIXER_TOKEN_HIDDEN"]),
        MIXER_CHANNEL_HIDDEN=int(params["MIXER_CHANNEL_HIDDEN"]),
        MIXER_COND_PROJ_HIDDEN=int(params["MIXER_COND_PROJ_HIDDEN"]),
        MIXER_COND_VEC_DIM=int(params["MIXER_COND_VEC_DIM"]),
        MIXER_TRAIN_EVERY=int(params["MIXER_TRAIN_EVERY"]),
        MIXER_TRAIN_EPOCHS=int(params["MIXER_TRAIN_EPOCHS"]),
        MIXER_BATCH_SIZE=int(params["MIXER_BATCH_SIZE"]),
        MIXER_TRAIN_POP_SUBSET=int(params["MIXER_TRAIN_POP_SUBSET"]),
        MIXER_MAX_TEACHER_PAIRS=int(params["MIXER_MAX_TEACHER_PAIRS"]),
        MIXER_MAX_MINIBATCHES=int(params["MIXER_MAX_MINIBATCHES"]),
        MIXER_LR_MATRIX=f"{params['MIXER_LR_MATRIX']:.8f}",
        MIXER_LR_VECTOR=f"{params['MIXER_LR_VECTOR']:.8f}",
        MUON_WEIGHT_DECAY=f"{params['MUON_WEIGHT_DECAY']:.8f}",
        MUON_MOMENTUM=f"{params['MUON_MOMENTUM']:.8f}",
        MUON_NS_STEPS=int(params["MUON_NS_STEPS"]),
        MUON_NESTEROV='true' if params["MUON_NESTEROV"] else 'false',
        ADAMW_BETA1=f"{params['ADAMW_BETA1']:.8f}",
        ADAMW_BETA2=f"{params['ADAMW_BETA2']:.8f}",
        ADAMW_EPS=f"{params['ADAMW_EPS']:.12f}",
        ADAMW_WEIGHT_DECAY=f"{params['ADAMW_WEIGHT_DECAY']:.8f}",
        BODY=RUST_BODY,
    )
    return src


# ─── ビルド & 実行 ────────────────────────────────────────────────────────────

def build_and_run(
    params: dict,
    project_dir: str,
    eval_time: int,
    trial_idx: int,
    out_dir: str,
) -> float:
    src = generate_rust_source(params)
    src_path = os.path.join(project_dir, "src", "main.rs")

    with open(src_path, "w", encoding="utf-8") as f:
        f.write(src)

    log_dir = os.path.join(out_dir, f"trial_{trial_idx:04d}")
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    print(f"  [trial {trial_idx}] Building ...", flush=True)
    build_start = time.time()
    build_result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    build_elapsed = time.time() - build_start

    if build_result.returncode != 0:
        print(f"  [trial {trial_idx}] BUILD FAILED ({build_elapsed:.1f}s)")
        print(build_result.stderr[-2000:])
        with open(os.path.join(log_dir, "build_stderr.txt"), "w") as f:
            f.write(build_result.stderr)
        return 0.0

    print(f"  [trial {trial_idx}] Build OK ({build_elapsed:.1f}s). Running {eval_time}s ...", flush=True)

    cargo_toml = os.path.join(project_dir, "Cargo.toml")
    bin_name = None
    try:
        import tomllib
    except ImportError:
        tomllib = None

    if tomllib and os.path.exists(cargo_toml):
        with open(cargo_toml, "rb") as f:
            toml_data = tomllib.load(f)
        bin_name = toml_data.get("package", {}).get("name", None)
    if bin_name is None:
        if os.path.exists(cargo_toml):
            for line in open(cargo_toml):
                m = re.match(r'\s*name\s*=\s*"([^"]+)"', line)
                if m:
                    bin_name = m.group(1)
                    break
    if bin_name is None:
        bin_name = os.path.basename(project_dir)

    binary = os.path.join(project_dir, "target", "release", bin_name)

    run_proc = subprocess.Popen(
        [binary],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=project_dir,
    )

    import select as _select
    lines = []
    deadline = time.time() + eval_time
    last_output_time = time.time()
    SILENCE_TIMEOUT = 15.0
    timed_out_by_silence = False

    try:
        while time.time() < deadline:
            remaining = min(deadline - time.time(), SILENCE_TIMEOUT - (time.time() - last_output_time))
            if remaining <= 0:
                if time.time() - last_output_time >= SILENCE_TIMEOUT:
                    timed_out_by_silence = True
                break
            ready, _, _ = _select.select([run_proc.stdout], [], [], max(remaining, 0.0))
            if not ready:
                if time.time() - last_output_time >= SILENCE_TIMEOUT:
                    timed_out_by_silence = True
                    break
                continue
            line = run_proc.stdout.readline()
            if not line:
                break
            last_output_time = time.time()
            lines.append(line.rstrip())
            if "CMAES_ACC" in line or "CMAES_STAGE" in line:
                print(f"    {line.rstrip()}", flush=True)
    finally:
        run_proc.terminate()
        try:
            run_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            run_proc.kill()

    with open(os.path.join(log_dir, "stdout.txt"), "w") as f:
        f.write("\n".join(lines))

    if timed_out_by_silence:
        print(f"  [trial {trial_idx}] INVALID: no output for {SILENCE_TIMEOUT:.0f}s → acc = None")
        with open(os.path.join(log_dir, "result.json"), "w") as f:
            json.dump({"acc": None, "invalid": "silence_timeout", **params}, f, indent=2)
        return None

    i = 0.0
    best_acc = 0.0
    for line in lines:
        m = re.search(r"CMAES_ACC.*acc=([0-9.]+)", line)
        if m:
            try:
                best_acc += math.log(max(1 - float(m.group(1)), 1e-24))
                i += 1
            except ValueError:
                pass
    best_acc = 1.0 - math.exp(best_acc / i)

    print(f"  [trial {trial_idx}] acc = {best_acc:.6f}")
    with open(os.path.join(log_dir, "result.json"), "w") as f:
        json.dump({"acc": best_acc, **params}, f, indent=2)

    return best_acc


# ─── CMA-ES ループ ────────────────────────────────────────────────────────────

class CMAESSearcher:
    def __init__(self, project_dir: str, eval_time: int, budget: int, out_dir: str):
        self.project_dir = project_dir
        self.eval_time = eval_time
        self.budget = budget
        self.out_dir = out_dir
        self.history: list[dict] = []
        self.trial_idx = 0

        os.makedirs(out_dir, exist_ok=True)

        x0 = initial_x()
        sigma0 = initial_sigma()

        bounds_lo = []
        bounds_hi = []
        for name, lo, hi, init, is_int, log_scale in PARAM_DEFS:
            if log_scale:
                bounds_lo.append(math.log(lo))
                bounds_hi.append(math.log(hi))
            else:
                bounds_lo.append(float(lo))
                bounds_hi.append(float(hi))

        opts = cma.CMAOptions()
        opts["bounds"] = [bounds_lo, bounds_hi]
        opts["maxfevals"] = budget
        opts["verbose"] = -9
        opts["tolx"] = 1e-4
        #opts["popsize"] = 30
        opts["tolfun"] = 1e-4

        self.es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        print(f"CMA-ES initialized: dim={len(x0)}, sigma0={sigma0:.3f}, budget={budget}")

    def objective(self, x: list[float]) -> float:
        params = decode(x)
        print(f"\n{'='*60}")
        print(f"Trial {self.trial_idx}: {params}")
        acc = build_and_run(params, self.project_dir, self.eval_time, self.trial_idx, self.out_dir)
        self.trial_idx += 1

        if acc is None:
            fitness = 0.0
            self.history.append({"trial": self.trial_idx - 1, "acc": None, "invalid": True, **params})
            self._save_history()
            return fitness

        EPS = 1e-24
        fitness = math.log(max(1.0 - acc, EPS))
        self.history.append({"trial": self.trial_idx - 1, "acc": acc, **params})
        self._save_history()
        return fitness

    def run(self):
        print("\n" + "="*60)
        print("Starting CMA-ES hyperparameter search")
        print("="*60)

        global_best_fitness = float("inf")
        global_best_params = None

        while not self.es.stop() and self.trial_idx < self.budget:
            solutions = self.es.ask()
            fitnesses = [self.objective(x) for x in solutions]
            self.es.tell(solutions, fitnesses)

            iter_best_idx = int(min(range(len(fitnesses)), key=lambda i: fitnesses[i]))
            iter_best_fitness = fitnesses[iter_best_idx]
            iter_best_params = decode(solutions[iter_best_idx])

            if iter_best_fitness < global_best_fitness:
                global_best_fitness = iter_best_fitness
                global_best_params = iter_best_params

            global_best_acc = 1.0 - math.exp(min(global_best_fitness, 0.0))
            iter_best_acc = 1.0 - math.exp(min(iter_best_fitness, 0.0))
            print(
                f"\n>>> iter best: acc≈{iter_best_acc:.6f} | "
                f"global best: acc≈{global_best_acc:.6f} params={global_best_params}"
            )

        print("\n" + "="*60)
        print("CMA-ES search finished.")
        if self.history:
            valid = [d for d in self.history if d.get("acc") is not None]
            if valid:
                best = max(valid, key=lambda d: d["acc"])
                print(f"Best trial #{best['trial']}: acc={best['acc']:.6f}")
                print(f"Best params: { {k: v for k, v in best.items() if k not in ('trial','acc')} }")
                with open(os.path.join(self.out_dir, "best_params.json"), "w") as f:
                    json.dump(best, f, indent=2)
                print(f"Saved to {self.out_dir}/best_params.json")
        return self.history

    def _save_history(self):
        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)


# ─── Cargo.toml テンプレート ─────────────────────────────────────────────────

CARGO_TOML_TEMPLATE = """\
[package]
name = "adf_cgp"
version = "0.1.0"
edition = "2021"

[dependencies]
ahash = "0.8"
dashmap = "5"
ndarray = "0.15"
num-complex = "0.4"
rand = { version = "0.8", features = ["small_rng"] }
rand_distr = "0.4"
rayon = "1"

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
"""


def setup_project(project_dir: str, src_rs_path: Optional[str] = None):
    src_dir = os.path.join(project_dir, "src")
    os.makedirs(src_dir, exist_ok=True)

    cargo_toml = os.path.join(project_dir, "Cargo.toml")
    if not os.path.exists(cargo_toml):
        print(f"Creating new Cargo project at {project_dir}")
        with open(cargo_toml, "w") as f:
            f.write(CARGO_TOML_TEMPLATE)

    src_dest = os.path.join(src_dir, "main.rs")
    if src_rs_path and os.path.exists(src_rs_path):
        shutil.copy2(src_rs_path, src_dest)
        print(f"Copied {src_rs_path} → {src_dest}")
    elif not os.path.exists(src_dest):
        params = {name: init for name, lo, hi, init, is_int, log_scale in PARAM_DEFS}
        with open(src_dest, "w") as f:
            f.write(generate_rust_source(params))
        print(f"Generated initial main.rs at {src_dest}")


# ─── エントリポイント ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CMA-ES hyperparameter search for ADF-CGP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        例:
          # カレントディレクトリが Cargo プロジェクトの場合
          python3 cmaes_search.py --project . --budget 30 --time 30

          # ソースファイルを指定して新規プロジェクト作成
          python3 cmaes_search.py --project ./adf_cgp_proj --src main.rs --budget 30 --time 30
        """),
    )
    parser.add_argument("--project", default=".", help="Cargo プロジェクトのルートディレクトリ")
    parser.add_argument("--src", default=None, help="元の main.rs パス (省略時はプロジェクト内のものを使用)")
    parser.add_argument("--out", default="cmaes_results", help="結果出力ディレクトリ")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="CMA-ES 最大評価回数")
    parser.add_argument("--time", type=int, default=DEFAULT_TIME, help="各評価の実行秒数")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project)
    out_dir = os.path.abspath(args.out)

    print(f"Project dir : {project_dir}")
    print(f"Output dir  : {out_dir}")
    print(f"Budget      : {args.budget} evaluations")
    print(f"Eval time   : {args.time} seconds each")

    setup_project(project_dir, args.src)

    searcher = CMAESSearcher(
        project_dir=project_dir,
        eval_time=args.time,
        budget=args.budget,
        out_dir=out_dir,
    )
    searcher.run()


if __name__ == "__main__":
    main()
    