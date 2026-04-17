use ahash::AHasher;
use dashmap::DashMap;
use num_complex::Complex;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::hash::{Hash, Hasher};

// ─────────────────────────────────────────
// Hyperparams
// ─────────────────────────────────────────
const N_INPUTS: usize = 2;
const MODEL_LEN: usize = 1024;
const POP_SIZE: usize = 4096;
const ELITE: usize = 64;
const N_GEN: usize = 200;
const VEC_LEN: usize = 2048;

// ─────────────────────────────────────────
// Safe ops
// ─────────────────────────────────────────
fn clamp(z: Complex<f64>) -> Complex<f64> {
    let m = 15.0;
    Complex::new(z.re.max(-m).min(m), z.im.max(-m).min(m))
}

fn safe_log(z: Complex<f64>) -> Complex<f64> {
    let eps = 1e-6;
    if z.norm() < eps {
        Complex::new(eps, 0.0).ln()
    } else {
        z.ln()
    }
}

fn eml(a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
    clamp(a).exp() - safe_log(b)
}

fn eml_vec(x: &[Complex<f64>], y: &[Complex<f64>]) -> Vec<Complex<f64>> {
    x.iter().zip(y).map(|(&a, &b)| eml(a, b)).collect()
}

// ─────────────────────────────────────────
// Genome
// ─────────────────────────────────────────
#[derive(Clone)]
struct Genome {
    conn: Vec<[u16; 2]>,
}

impl Genome {
    fn random(rng: &mut SmallRng) -> Self {
        let conn = (0..MODEL_LEN)
            .map(|i| {
                let max = (N_INPUTS + i) as u16;
                [rng.gen_range(0..max), rng.gen_range(0..max)]
            })
            .collect();
        Self { conn }
    }

    // [FIX] active-node aware crossover:
    // activeノードの接続だけ選択的に交換することで
    // 親2の有効サブグラフ構造を保ちやすくする。
    // 非activeノードはどちらでも評価に影響しないので
    // 親1をそのまま使う（どうせactive pruningで無視される）。
    fn mix(&self, rng: &mut SmallRng, g2: &Genome) -> Self {
        let active2 = active_nodes(g2);
        let mut g = self.clone();

        // g2のactiveノードをランダムサンプリングして移植
        let n_copy = rng.gen_range(1..=active2.len().max(1));
        let mut candidates: Vec<usize> = active2
            .into_iter()
            .filter(|&i| i >= N_INPUTS) // 入力ノードは固定
            .collect();

        // Fisher-Yates で先頭n_copy個を選ぶ
        for i in 0..n_copy.min(candidates.len()) {
            let j = rng.gen_range(i..candidates.len());
            candidates.swap(i, j);
            let abs = candidates[i];
            let local = abs - N_INPUTS;
            g.conn[local] = g2.conn[local];
        }
        g
    }

    // [FIX] geometric分布で変異数をサンプリング:
    // 元のuniform(1..MODEL_LEN/2)は大変異に偏りすぎ。
    // p=0.15 のgeometric分布は期待値~6.7変異で
    // 局所探索を維持しつつ稀に大ジャンプも起きる。
    fn mutate(&self, rng: &mut SmallRng) -> Self {
        let mut g = self.clone();
        let p = 0.15_f64;
        // geometric分布: k回失敗してから1回成功
        let n_mut = {
            let mut k = 1usize;
            while rng.gen::<f64>() > p && k < MODEL_LEN / 2 {
                k += 1;
            }
            k
        };
        for _ in 0..n_mut {
            let i = rng.gen_range(0..MODEL_LEN);
            let max = (N_INPUTS + i) as u16;
            g.conn[i][rng.gen_range(0..2)] = rng.gen_range(0..max);
            if (rng.gen::<f64>() < 0.1) {
                g.conn[i][rng.gen_range(0..2)] = rng.gen_range(0..2);
            }
        }
        g
    }
}

// ─────────────────────────────────────────
// Active node pruning
// ─────────────────────────────────────────
fn active_nodes(g: &Genome) -> Vec<usize> {
    let mut active = vec![false; N_INPUTS + MODEL_LEN];
    active[N_INPUTS + MODEL_LEN - 1] = true;

    for i in (0..MODEL_LEN).rev() {
        let idx = N_INPUTS + i;
        if !active[idx] {
            continue;
        }
        let c = g.conn[i];
        active[c[0] as usize] = true;
        active[c[1] as usize] = true;
    }

    (0..N_INPUTS + MODEL_LEN).filter(|&i| active[i]).collect()
}

// ─────────────────────────────────────────
// Dataset (1D → 1D)
// ─────────────────────────────────────────
struct Dataset {
    inputs: Vec<[Vec<Complex<f64>>; N_INPUTS]>,
    targets: Vec<Vec<Complex<f64>>>,
    n: usize,
}

impl Dataset {
    fn build() -> Self {
        let mut inputs = vec![];
        let mut targets = vec![];

        for _ in 0..1 {
            let mut x = vec![];
            let mut one = vec![];

            for i in 0..VEC_LEN {
                let v = -2.0 + 20.0 * i as f64 / VEC_LEN as f64;
                x.push(Complex::new(v, 0.0));
                one.push(Complex::new(1.0, 0.0));
            }

            let target: Vec<_> = x.iter().map(|z| (z * z).sin()).collect();

            inputs.push([x, one]);
            targets.push(target);
        }

        Self {
            n: inputs.len(),
            inputs,
            targets,
        }
    }
}

// ─────────────────────────────────────────
// Cache
// ─────────────────────────────────────────
type Sig = u64;

// node_cache の上限エントリ数。
// 1エントリ = VEC_LEN * 16 bytes = 8192 bytes。
// 65536エントリ ≒ 512 MB。
const NODE_CACHE_CAP: usize = 65_536;

// fitness_cache の上限エントリ数。
// 1エントリ = (f64, f64) = 16 bytes。
// POP_SIZE * 2世代分で十分なhit率が得られる。
const FITNESS_CACHE_CAP: usize = POP_SIZE * 2;

struct Evaluator {
    node_cache: DashMap<Sig, Vec<Complex<f64>>>,
    fitness_cache: DashMap<u64, (f64, f64)>,
}

impl Evaluator {
    fn new() -> Self {
        Self {
            node_cache: DashMap::with_capacity(NODE_CACHE_CAP),
            fitness_cache: DashMap::with_capacity(FITNESS_CACHE_CAP),
        }
    }

    // node_cache を世代ごとにリセットする。
    // 世代をまたいだノードsigのhitは配線変化でほぼ起きないため
    // リセットしてもキャッシュ効率はほぼ低下しない。
    // 一方でリセットしないと世代数×エントリ数がOOMの原因になる。
    fn reset_node_cache(&self) {
        self.node_cache.clear();
    }

    // fitness_cache は上限を超えたらクリア（LRUの簡易代替）。
    // エリートはすぐ再評価されるので実害は少ない。
    fn maybe_trim_fitness_cache(&self) {
        if self.fitness_cache.len() > FITNESS_CACHE_CAP {
            self.fitness_cache.clear();
        }
    }
}

// [FIX] make_sig を非可換化:
// eml(a,b) != eml(b,a) なので sig(a,b) != sig(b,a) でなければならない。
// op_id を混ぜることで将来的な多演算拡張にも対応。
fn make_sig(a: Sig, b: Sig, op_id: u8) -> Sig {
    let mut h = AHasher::default();
    op_id.hash(&mut h);
    a.hash(&mut h);
    b.hash(&mut h);
    h.finish()
}

// [FIX] genome_sig: conn全体をハッシュに含める。
// 元実装はactive_nodesのインデックス集合だけをハッシュしていたが、
// active setが同じでも接続先が違えば出力は異なる → キャッシュ誤hitのバグ。
fn genome_sig(g: &Genome, active: &[usize]) -> u64 {
    let mut h = AHasher::default();
    for &abs in active {
        if abs < N_INPUTS {
            continue;
        }
        let i = abs - N_INPUTS;
        g.conn[i][0].hash(&mut h);
        g.conn[i][1].hash(&mut h);
        abs.hash(&mut h); // ノード位置も含める
    }
    h.finish()
}

// ─────────────────────────────────────────
// Stats
// ─────────────────────────────────────────
fn argsort(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<_> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    idx
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;

    for i in 0..x.len() {
        let a = x[i] - mx;
        let b = y[i] - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }

    if dx < 1e-12 || dy < 1e-12 {
        return 0.0;
    }

    num / (dx.sqrt() * dy.sqrt())
}

fn chatterjee(x: &[f64], y: &[f64]) -> f64 {
    let idx = argsort(x);
    let mut ranks = vec![0; y.len()];
    let ys = argsort(y);
    for (r, &i) in ys.iter().enumerate() {
        ranks[i] = r as i64;
    }

    let mut s = 0;
    for w in idx.windows(2) {
        s += (ranks[w[0]] - ranks[w[1]]).abs();
    }

    let n = x.len() as f64;
    1.0 - 3.0 * s as f64 / (n * n - 1.0)
}

fn decompose(z: &[Complex<f64>]) -> [Vec<f64>; 4] {
    [
        z.iter().map(|c| c.re).collect(),
        z.iter().map(|c| c.im).collect(),
        z.iter().map(|c| c.norm()).collect(),
        z.iter().map(|c| c.arg()).collect(),
    ]
}

// [FIX] 損失関数の再設計:
//
// 旧: score = mean(chatterjee + pearson) / 8  → 両者は高度に相関し情報が重複
//     loss = (1 - score) * (1 - acc)          → accが高いと損失が0に潰れ不安定
//
// 新設計の方針:
//   - "shape" (形状一致) = Pearsonで十分。Chatterjeeは非線形関係の検出に特化させる
//     → re成分だけ両方使い、それ以外はPearsonのみ（計算コスト削減にも）
//   - "scale" (スケール一致) = 正規化MSEで直接評価
//     Pearson/Chatterjeeはスケール不変なのでMSEと補完的
//   - 重み: shape(0.7) + scale(0.3) で形状優先
//   - 最終lossは単純に 1 - total_score。掛け算で潰す操作は廃止。
fn normalized_mse(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;
    let var_t: f64 = {
        let m = target.iter().sum::<f64>() / n;
        target.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / n
    };
    if var_t < 1e-12 {
        return 0.0;
    }
    let mse: f64 = pred
        .iter()
        .zip(target)
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f64>()
        / n;
    // 0.0(完全一致) ～ 1.0(分散と同程度の誤差) にクリップ
    (mse / var_t).min(1.0)
}

fn score(pred: &[Complex<f64>], target: &[Complex<f64>]) -> f64 {
    let p = decompose(pred);
    let t = decompose(target);

    // shape score: re成分はChatterjee+Pearson、それ以外はPearsonのみ
    let shape = {
        let re_shape = (chatterjee(&p[0], &t[0]) + pearson(&p[0], &t[0])) * 0.5;
        let im_shape = pearson(&p[1], &t[1]);
        let norm_shape = pearson(&p[2], &t[2]);
        let arg_shape = pearson(&p[3], &t[3]);
        (re_shape + im_shape + norm_shape + arg_shape) / 4.0
    };

    // scale score: re成分の正規化MSEで直接的なスケール一致を測る
    let scale = 1.0 - normalized_mse(&p[0], &t[0]);

    0.7 * shape + 0.3 * scale
}

// accuracyは表示用のみ（損失計算には使わない）
fn accuracy(pred: &[Complex<f64>], target: &[Complex<f64>]) -> f64 {
    let p = decompose(pred);
    let t = decompose(target);

    let mut s: f64 = 0.0;
    for i in 0..4 {
        s = s.max(pearson(&p[i], &t[i]).abs());
    }
    s
}

fn execute(
    g: &Genome,
    active: &[usize],
    inp: &[Vec<Complex<f64>>; N_INPUTS],
    ev: &Evaluator,
) -> Vec<Complex<f64>> {
    let total = N_INPUTS + MODEL_LEN;
    let mut nodes: Vec<Option<(Sig, Vec<Complex<f64>>)>> = vec![None; total];

    for i in 0..N_INPUTS {
        nodes[i] = Some((i as u64, inp[i].clone()));
    }

    for &abs in active {
        if abs < N_INPUTS {
            continue;
        }

        let i = abs - N_INPUTS;

        let (s1, ref v1) = nodes[g.conn[i][0] as usize]
            .as_ref()
            .expect("dependency not computed");

        let (s2, ref v2) = nodes[g.conn[i][1] as usize]
            .as_ref()
            .expect("dependency not computed");

        // [FIX] op_id=0 を渡して非可換なsigを生成
        let sig = make_sig(*s1, *s2, 0);

        let val = if let Some(v) = ev.node_cache.get(&sig) {
            v.clone()
        } else {
            let out = eml_vec(v1, v2);
            ev.node_cache.insert(sig, out.clone());
            out
        };

        nodes[abs] = Some((sig, val));
    }

    nodes[N_INPUTS + MODEL_LEN - 1].as_ref().unwrap().1.clone()
}

// ─────────────────────────────────────────
// Evaluation
// ─────────────────────────────────────────
fn eval(g: &Genome, ds: &Dataset, ev: &Evaluator) -> (f64, f64) {
    let active = active_nodes(g);

    // [FIX] conn配線を含む正当なキーで fitness をキャッシュ
    let key = genome_sig(g, &active);

    if let Some(v) = ev.fitness_cache.get(&key) {
        return *v;
    }

    let mut loss = 0.0;
    let mut acc = 0.0;

    for i in 0..ds.n {
        let out = execute(g, &active, &ds.inputs[i], ev);
        loss += 1.0 - score(&out, &ds.targets[i]);
        acc += accuracy(&out, &ds.targets[i]);
    }

    // [FIX] 単純な平均loss。掛け算で潰す操作を廃止。
    let l = loss / ds.n as f64;
    let a = acc / ds.n as f64;

    ev.fitness_cache.insert(key, (l, a));
    (l, a)
}

// ─────────────────────────────────────────
// Main
// ─────────────────────────────────────────
fn main() {
    let ds = Dataset::build();
    let mut rng = SmallRng::seed_from_u64(42);

    let mut pop: Vec<_> = (0..POP_SIZE).map(|_| Genome::random(&mut rng)).collect();

    let evaluator = Evaluator::new();

    for gen in 0..N_GEN {
        // 世代頭にnode_cacheをリセット:
        // 前世代の配線とは別のsigが生成されるためhit率が低く
        // リセットしないとメモリが世代数に比例して膨張する。
        evaluator.reset_node_cache();

        // fitness_cacheが膨らみすぎていれば簡易クリア
        evaluator.maybe_trim_fitness_cache();

        let mut scored: Vec<_> = pop
            .par_iter()
            .map(|g| {
                let (l, a) = eval(g, &ds, &evaluator);
                (l, a, g.clone())
            })
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        println!(
            "gen {:4}  loss {:.6}  acc {:.6}",
            gen, scored[0].0, scored[0].1
        );

        // [FIX] ELITE個を次世代に残す（元実装は2個だった）
        let elites: Vec<_> = scored[..ELITE].iter().map(|x| x.2.clone()).collect();

        let mut next = elites.clone();
        while next.len() < POP_SIZE {
            let g = &scored[rng.gen_range(0..ELITE)].2;
            let g2 = &scored[rng.gen_range(0..ELITE)].2;

            // [FIX] 確率的に交叉のみ・突然変異のみ・両方を選ぶ
            let child = match rng.gen_range(0..3) {
                0 => g.mix(&mut rng, g2),                  // crossover only
                1 => g.mutate(&mut rng),                   // mutation only
                _ => g.mix(&mut rng, g2).mutate(&mut rng), // both
            };
            next.push(child);
        }

        pop = next;
    }
}
