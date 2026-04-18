use ahash::AHasher;
use dashmap::DashMap;
use num_complex::Complex;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const N_INPUTS_MAIN: usize = 2;
const N_ADF: usize = 1;
const LEN_LAYER0: usize = 8;
const LEN_LAYER1: usize = 8;
const LEN_LAYER2: usize = 256;
const N_INPUTS_ADF: usize = 3; // スロット 0=x, 1=y, 2=one(定数1ベクタ)
/// ADF の第3入力 (定数1) に割り当てる固定シグネチャ
const ONE_SIG: Sig = 0xFFFF_FFFF_FFFF_FFFFu64;
const VEC_LEN: usize = 1024;
const POP_SIZE: usize = 512;
const ELITE: usize = 16;
const N_GEN: usize = 2000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Layer {
    L0,
    L1,
    L2,
}

impl Layer {
    #[inline]
    fn len(self) -> usize {
        match self {
            Layer::L0 => LEN_LAYER0,
            Layer::L1 => LEN_LAYER1,
            Layer::L2 => LEN_LAYER2,
        }
    }
    #[inline]
    fn n_funcs(self) -> usize {
        match self {
            Layer::L0 => 1,
            Layer::L1 => 1 + N_ADF,
            Layer::L2 => 1 + 2 * N_ADF,
        }
    }
    #[inline]
    fn n_ext(self) -> usize {
        match self {
            Layer::L0 | Layer::L1 => N_INPUTS_ADF,
            Layer::L2 => N_INPUTS_MAIN,
        }
    }
}

#[inline(always)]
fn clamp(z: Complex<f64>) -> Complex<f64> {
    const M: f64 = 15.0;
    Complex::new(z.re.clamp(-M, M), z.im.clamp(-M, M))
}

#[inline(always)]
fn safe_log(z: Complex<f64>) -> Complex<f64> {
    const EPS: f64 = 1e-6;
    if z.norm_sqr() < EPS * EPS {
        Complex::new(EPS, 0.0).ln()
    } else {
        z.ln()
    }
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
    layer: Layer,
    conn: Box<[[u16; 2]]>,
    func: Box<[u8]>,
}

fn selectfunc(max: u8, rng: &mut SmallRng) -> u8 {
    if rng.gen::<f64>() < 0.5 {
        return 0;
    }
    rng.gen_range(0..max)
}

impl Chromosome {
    fn random(layer: Layer, rng: &mut SmallRng) -> Self {
        let n = layer.len();
        let n_ext = layer.n_ext();
        let n_f = layer.n_funcs() as u8;
        let conn = (0..n)
            .map(|i| {
                let m = (n_ext + i) as u16;
                [rng.gen_range(0..m), rng.gen_range(0..m)]
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let func = (0..n)
            .map(|_| selectfunc(n_f, rng))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { layer, conn, func }
    }

    /// active ノードリストと染色体シグネチャを一度の走査で計算
    fn active_and_sig(&self) -> (Vec<usize>, u64) {
        let n = self.layer.len();
        let n_ext = self.layer.n_ext();
        let total = n_ext + n;
        let mut active = vec![false; total];
        active[total - 1] = true;
        for i in (0..n).rev() {
            if !active[n_ext + i] {
                continue;
            }
            active[self.conn[i][0] as usize] = true;
            active[self.conn[i][1] as usize] = true;
        }
        let mut h = AHasher::default();
        (self.layer as u8).hash(&mut h);
        let list: Vec<usize> = (0..total)
            .filter(|&i| active[i])
            .inspect(|&abs| {
                if abs >= n_ext {
                    let i = abs - n_ext;
                    self.conn[i][0].hash(&mut h);
                    self.conn[i][1].hash(&mut h);
                    self.func[i].hash(&mut h);
                    abs.hash(&mut h);
                }
            })
            .collect();
        (list, h.finish())
    }

    fn mutate(&self, rng: &mut SmallRng) -> Self {
        let mut conn = self.conn.to_vec();
        let mut func = self.func.to_vec();
        let n = self.layer.len();
        let n_ext = self.layer.n_ext();
        let n_f = self.layer.n_funcs() as u8;
        let mut n_mut = 1usize;
        while rng.gen::<f64>() > 0.15 && n_mut < n {
            n_mut += 1;
        }
        for _ in 0..n_mut {
            let i = rng.gen_range(0..n);
            let max = (n_ext + i) as u16;
            match rng.gen_range(0..3u8) {
                0 => conn[i][0] = rng.gen_range(0..max),
                1 => conn[i][1] = rng.gen_range(0..max),
                _ => func[i] = selectfunc(n_f, rng),
            }
            if rng.gen::<f64>() < 0.05 {
                conn[i][rng.gen_range(0..2)] = rng.gen_range(0..n_ext as u16);
            }
        }
        Self {
            layer: self.layer,
            conn: conn.into_boxed_slice(),
            func: func.into_boxed_slice(),
        }
    }

    fn mix(&self, rng: &mut SmallRng, other: &Chromosome) -> Self {
        let (active2, _) = other.active_and_sig();
        let n_ext = self.layer.n_ext();
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
        Self {
            layer: self.layer,
            conn: conn.into_boxed_slice(),
            func: func.into_boxed_slice(),
        }
    }
}

// ─── Genome ───────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Genome {
    l0: Vec<Chromosome>,
    l1: Vec<Chromosome>,
    l2: Chromosome,
}

impl Genome {
    fn random(rng: &mut SmallRng) -> Self {
        Self {
            l0: (0..N_ADF)
                .map(|_| Chromosome::random(Layer::L0, rng))
                .collect(),
            l1: (0..N_ADF)
                .map(|_| Chromosome::random(Layer::L1, rng))
                .collect(),
            l2: Chromosome::random(Layer::L2, rng),
        }
    }
    fn mutate(&self, rng: &mut SmallRng) -> Self {
        let mut g = self.clone();
        for t in 0..2 * N_ADF + 1 {
            if t < N_ADF {
                g.l0[t] = self.l0[t].mutate(rng);
            } else if t < 2 * N_ADF {
                g.l1[t - N_ADF] = self.l1[t - N_ADF].mutate(rng);
            } else {
                g.l2 = self.l2.mutate(rng);
            }
        }
        g
    }
    fn mix(&self, rng: &mut SmallRng, other: &Genome) -> Self {
        let mut g = self.clone();
        for k in 0..N_ADF {
            if rng.gen::<f64>() < 0.5 {
                g.l0[k] = self.l0[k].mix(rng, &other.l0[k]);
            }
            if rng.gen::<f64>() < 0.5 {
                g.l1[k] = self.l1[k].mix(rng, &other.l1[k]);
            }
        }
        if rng.gen::<f64>() < 0.5 {
            g.l2 = self.l2.mix(rng, &other.l2);
        }
        g
    }
}

// ─── Signature ────────────────────────────────────────────────────────────────

type Sig = u64;

/// 低コストな非可換ハッシュ (AHasher 構築不要)
#[inline]
fn make_sig(a: Sig, b: Sig, func_id: u8) -> Sig {
    let mut x = a ^ ((func_id as u64).wrapping_mul(0x9e3779b97f4a7c15));
    x ^= b.wrapping_mul(0x6c62272e07bb0142);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn genome_key(l0s: &[Sig], l1s: &[Sig], l2s: Sig) -> u64 {
    let mut h = AHasher::default();
    for &s in l0s {
        s.hash(&mut h);
    }
    for &s in l1s {
        s.hash(&mut h);
    }
    l2s.hash(&mut h);
    h.finish()
}

// ─── Cache & Evaluator ────────────────────────────────────────────────────────

type ArcVec = Arc<[Complex<f64>]>;
type AdfKey = (Sig, Sig, Sig);

// ─── NodeBuf: 実行中の中間値バッファ ─────────────────────────────────────────
// Vec<Option<...>> を避け、(Sig, ArcVec) の平行配列に。
// active_list の順に書くので未初期化アクセスは起きない。

struct NodeBuf {
    sigs: Vec<Sig>,
    vals: Vec<ArcVec>,
}

impl NodeBuf {
    fn new(total: usize) -> Self {
        let dummy: ArcVec = Arc::from(vec![Complex::new(0.0, 0.0); VEC_LEN].into_boxed_slice());
        Self {
            sigs: vec![0u64; total],
            vals: vec![dummy; total],
        }
    }
    #[inline]
    fn set(&mut self, idx: usize, sig: Sig, val: ArcVec) {
        self.sigs[idx] = sig;
        self.vals[idx] = val;
    }
    #[inline]
    fn sig(&self, idx: usize) -> Sig {
        self.sigs[idx]
    }
    #[inline]
    fn val(&self, idx: usize) -> &ArcVec {
        &self.vals[idx]
    }
}

// ─── get-or-compute: read-first で競合を最小化 ───────────────────────────────

#[inline]
fn node_get_or_compute(
    sig: Sig,
    v0: &[Complex<f64>],
    v1: &[Complex<f64>],
    ev: &Evaluator,
) -> ArcVec {
    if let Some(v) = ev.node_cache.get(&sig) {
        return Arc::clone(&*v);
    }
    let out: ArcVec = Arc::from(eml_vec(v0, v1).into_boxed_slice());
    // or_insert_with で他スレッドの先勝ちを尊重しつつ自身の out を返す
    ev.node_cache.entry(sig).or_insert_with(|| Arc::clone(&out));
    out
}

// ─── ADF 実行 (Layer0) ────────────────────────────────────────────────────────

// ─── Dataset ──────────────────────────────────────────────────────────────────

struct Dataset {
    inputs: Vec<[Vec<Complex<f64>>; N_INPUTS_MAIN]>,
    targets: Vec<Vec<Complex<f64>>>,
    n: usize,
}

impl Dataset {
    fn build() -> Self {
        let x: Vec<_> = (0..VEC_LEN)
            .map(|i| Complex::new(-6.0 + 12.0 * i as f64 / VEC_LEN as f64, 0.0))
            .collect();
        let one = vec![Complex::new(1.0, 0.0); VEC_LEN];
        let target: Vec<_> = x.iter().map(|z| (z * z).sin()).collect();
        Self {
            n: 1,
            inputs: vec![[x, one]],
            targets: vec![target],
        }
    }
}

// ─── 評価指標 ─────────────────────────────────────────────────────────────────

fn argsort(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_unstable_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

fn pearson_raw(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut num, mut dx, mut dy) = (0.0f64, 0.0f64, 0.0f64);
    for (&xi, &yi) in x.iter().zip(y) {
        let a = xi - mx;
        let b = yi - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    if dx < 1e-12 || dy < 1e-12 {
        0.0
    } else {
        num / (dx.sqrt() * dy.sqrt())
    }
}

/// argsort 済みインデックスと rank 配列を受け取って Chatterjee を計算
#[inline]
fn chatterjee_from_ranks(sorted_idx: &[usize], ranks: &[i64]) -> f64 {
    let s: i64 = sorted_idx
        .windows(2)
        .map(|w| (ranks[w[0]] - ranks[w[1]]).abs())
        .sum();
    let n = sorted_idx.len() as f64;
    1.0 - 3.0 * s as f64 / (n * n - 1.0)
}

/// score と accuracy を 1 回のループで計算 (中間 Vec を削減)
fn score_and_acc(pred: &[Complex<f64>], target: &[Complex<f64>]) -> (f64, f64) {
    let n = pred.len();
    // 4 チャンネルを同時展開
    let mut p = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    let mut t = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for (pc, tc) in pred.iter().zip(target) {
        p[0].push(pc.re);
        p[1].push(pc.im);
        p[2].push(pc.norm());
        p[3].push(pc.arg());
        t[0].push(tc.re);
        t[1].push(tc.im);
        t[2].push(tc.norm());
        t[3].push(tc.arg());
    }

    let mut total_score = 0.0f64;
    let mut max_pearson = 0.0f64;

    for ch in 0..4 {
        let px = &p[ch];
        let tx = &t[ch];
        let px_idx = argsort(px);
        let tx_idx = argsort(tx);
        let mut px_ranks = vec![0i64; n];
        let mut tx_ranks = vec![0i64; n];
        for (r, &i) in px_idx.iter().enumerate() {
            px_ranks[i] = r as i64;
        }
        for (r, &i) in tx_idx.iter().enumerate() {
            tx_ranks[i] = r as i64;
        }

        let c_fwd = chatterjee_from_ranks(&px_idx, &tx_ranks);
        let c_bwd = chatterjee_from_ranks(&tx_idx, &px_ranks);
        let pe = pearson_raw(px, tx).abs();

        total_score += (c_fwd + c_bwd + pe) / 3.0;
        if pe > max_pearson {
            max_pearson = pe;
        }
    }

    (total_score / 4.0, max_pearson)
}

// ─── Fitness 評価 ─────────────────────────────────────────────────────────────
// --- 既存のインポートと定数は維持 ---

type Score = (f64, f64); // (total_score, max_pearson)

struct Evaluator {
    node_cache: DashMap<Sig, ArcVec>,
    adf_cache: DashMap<AdfKey, (Sig, ArcVec, f64, usize)>, // スコア総和とノード数を追加
    fitness_cache: DashMap<u64, (f64, f64)>,
    /// 計算済みベクトルのスコア (Sig -> Score)
    score_cache: DashMap<Sig, Score>,
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
    fn reset_generation(&self) {
        self.node_cache.clear();
        self.adf_cache.clear();
        // score_cache は世代を跨いで再利用可能（ターゲットが同じなら）
    }

    /// ノードの出力ベクトルに対するスコアを取得（キャッシュ利用）
    fn get_node_score(&self, sig: Sig, val: &ArcVec, target: &[Complex<f64>]) -> Score {
        if let Some(s) = self.score_cache.get(&sig) {
            return *s;
        }
        let s = score_and_acc(val, target);
        self.score_cache.insert(sig, s);
        s
    }
}

// ─── ADF 実行 (Layer0) ───
fn exec_l0(
    c: &Chromosome,
    c_sig: Sig,
    active: &[usize],
    in0_sig: Sig,
    in0: &ArcVec,
    in1_sig: Sig,
    in1: &ArcVec,
    one: &ArcVec,
    target: &[Complex<f64>], // ターゲットを追加
    ev: &Evaluator,
) -> (Sig, ArcVec, f64, usize) {
    let key = (c_sig, in0_sig, in1_sig);
    if let Some(v) = ev.adf_cache.get(&key) {
        return (v.0, Arc::clone(&v.1), v.2, v.3);
    }

    let n_ext = N_INPUTS_ADF;
    let total = n_ext + c.layer.len();
    let mut buf = NodeBuf::new(total);
    buf.set(0, in0_sig, Arc::clone(in0));
    buf.set(1, in1_sig, Arc::clone(in1));
    buf.set(2, ONE_SIG, Arc::clone(one));

    let mut sum_score = 0.0;
    let mut count = 0;

    for &abs in active {
        if abs < n_ext {
            continue;
        }
        let i = abs - n_ext;
        let s0 = buf.sig(c.conn[i][0] as usize);
        let s1 = buf.sig(c.conn[i][1] as usize);
        let sig = make_sig(s0, s1, 0);
        let val = node_get_or_compute(
            sig,
            buf.val(c.conn[i][0] as usize),
            buf.val(c.conn[i][1] as usize),
            ev,
        );

        // 中間ノード評価
        let (s, _) = ev.get_node_score(sig, &val, target);
        sum_score += s;
        count += 1;

        buf.set(abs, sig, val);
    }

    let result = (
        buf.sig(total - 1),
        Arc::clone(buf.val(total - 1)),
        sum_score,
        count,
    );
    ev.adf_cache.entry(key).or_insert_with(|| result.clone());
    result
}

// ─── ADF 実行 (Layer1) ───
fn exec_l1(
    c: &Chromosome,
    c_sig: Sig,
    active: &[usize],
    in0_sig: Sig,
    in0: &ArcVec,
    in1_sig: Sig,
    in1: &ArcVec,
    one: &ArcVec,
    l0_chroms: &[Chromosome],
    l0_sigs: &[Sig],
    l0_acts: &[Vec<usize>],
    target: &[Complex<f64>],
    ev: &Evaluator,
) -> (Sig, ArcVec, f64, usize) {
    let key = (c_sig, in0_sig, in1_sig);
    if let Some(v) = ev.adf_cache.get(&key) {
        return (v.0, Arc::clone(&v.1), v.2, v.3);
    }

    let n_ext = N_INPUTS_ADF;
    let total = n_ext + c.layer.len();
    let mut buf = NodeBuf::new(total);
    buf.set(0, in0_sig, Arc::clone(in0));
    buf.set(1, in1_sig, Arc::clone(in1));
    buf.set(2, ONE_SIG, Arc::clone(one));

    let mut sum_score = 0.0;
    let mut count = 0;

    for &abs in active {
        if abs < n_ext {
            continue;
        }
        let i = abs - n_ext;
        let s0 = buf.sig(c.conn[i][0] as usize);
        let s1 = buf.sig(c.conn[i][1] as usize);
        let f = c.func[i] as usize;

        let (sig, val, s_inc, c_inc) = if f == 0 {
            let sig = make_sig(s0, s1, 0);
            let val = node_get_or_compute(
                sig,
                buf.val(c.conn[i][0] as usize),
                buf.val(c.conn[i][1] as usize),
                ev,
            );
            let (s, _) = ev.get_node_score(sig, &val, target);
            (sig, val, s, 1)
        } else {
            let idx = (f - 1).min(N_ADF - 1);
            exec_l0(
                &l0_chroms[idx],
                l0_sigs[idx],
                &l0_acts[idx],
                s0,
                buf.val(c.conn[i][0] as usize),
                s1,
                buf.val(c.conn[i][1] as usize),
                one,
                target,
                ev,
            )
        };

        sum_score += s_inc;
        count += c_inc;
        buf.set(abs, sig, val);
    }

    let result = (
        buf.sig(total - 1),
        Arc::clone(buf.val(total - 1)),
        sum_score,
        count,
    );
    ev.adf_cache.entry(key).or_insert_with(|| result.clone());
    result
}

// ─── Layer2 実行 ───
fn exec_l2(
    g: &Genome,
    l2_active: &[usize],
    l0_sigs: &[Sig],
    l0_acts: &[Vec<usize>],
    l1_sigs: &[Sig],
    l1_acts: &[Vec<usize>],
    ext: &[Vec<Complex<f64>>; N_INPUTS_MAIN],
    target: &[Complex<f64>],
    ev: &Evaluator,
) -> (ArcVec, f64, usize) {
    let c = &g.l2;
    let n_ext = N_INPUTS_MAIN;
    let total = n_ext + c.layer.len();
    let mut buf = NodeBuf::new(total);
    for i in 0..n_ext {
        buf.set(i, i as Sig + 1, Arc::from(ext[i].as_slice()));
    }
    let one_arc = Arc::from(ext[1].as_slice());

    let mut sum_score = 0.0;
    let mut count = 0;

    for &abs in l2_active {
        if abs < n_ext {
            continue;
        }
        let i = abs - n_ext;
        let c0 = c.conn[i][0] as usize;
        let c1 = c.conn[i][1] as usize;
        let s0 = buf.sig(c0);
        let s1 = buf.sig(c1);
        let f = c.func[i] as usize;

        let (sig, val, s_inc, c_inc) = if f == 0 {
            let sig = make_sig(s0, s1, 0);
            let val = node_get_or_compute(sig, buf.val(c0), buf.val(c1), ev);
            let (s, _) = ev.get_node_score(sig, &val, target);
            (sig, val, s, 1)
        } else if f <= N_ADF {
            let idx = f - 1;
            exec_l0(
                &g.l0[idx],
                l0_sigs[idx],
                &l0_acts[idx],
                s0,
                buf.val(c0),
                s1,
                buf.val(c1),
                &one_arc,
                target,
                ev,
            )
        } else {
            let idx = (f - N_ADF - 1).min(N_ADF - 1);
            exec_l1(
                &g.l1[idx],
                l1_sigs[idx],
                &l1_acts[idx],
                s0,
                buf.val(c0),
                s1,
                buf.val(c1),
                &one_arc,
                &g.l0,
                l0_sigs,
                l0_acts,
                target,
                ev,
            )
        };

        sum_score += s_inc;
        count += c_inc;
        buf.set(abs, sig, val);
    }

    (Arc::clone(buf.val(total - 1)), sum_score, count)
}

// ─── Fitness 評価 (eval) ───
fn eval(g: &Genome, ds: &Dataset, ev: &Evaluator) -> (f64, f64) {
    let l0_data: Vec<(Vec<usize>, Sig)> = g.l0.iter().map(|c| c.active_and_sig()).collect();
    let l1_data: Vec<(Vec<usize>, Sig)> = g.l1.iter().map(|c| c.active_and_sig()).collect();
    let (l2_active, l2_sig) = g.l2.active_and_sig();

    let l0_sigs: Vec<Sig> = l0_data.iter().map(|d| d.1).collect();
    let l0_acts: Vec<Vec<usize>> = l0_data.into_iter().map(|d| d.0).collect();
    let l1_sigs: Vec<Sig> = l1_data.iter().map(|d| d.1).collect();
    let l1_acts: Vec<Vec<usize>> = l1_data.into_iter().map(|d| d.0).collect();

    let gkey = genome_key(&l0_sigs, &l1_sigs, l2_sig);
    if let Some(v) = ev.fitness_cache.get(&gkey) {
        return *v;
    }

    let mut total_loss = 0.0;
    let mut total_acc = 0.0;

    for ds_idx in 0..ds.n {
        let target = &ds.targets[ds_idx];
        let (pred, sum_score, count) = exec_l2(
            g,
            &l2_active,
            &l0_sigs,
            &l0_acts,
            &l1_sigs,
            &l1_acts,
            &ds.inputs[ds_idx],
            target,
            ev,
        );

        // 最終出力のスコア
        let (final_s, final_a) = ev.get_node_score(l2_sig, &pred, target);

        // 中間表現全体の平均スコア
        let avg_inter_score = if count > 0 {
            sum_score / count as f64
        } else {
            0.0
        };

        // 損失計算: 最終出力の精度を優先しつつ、中間層がターゲットに近い特徴を持つほどボーナス
        // 以下の式では、中間表現の寄与率を 10% (0.1) としています
        //let combined_score = final_s * 0.9 + avg_inter_score * 0.1;

        total_loss += (1.0 - (avg_inter_score + final_s) / 2.0) * (1.0 - final_a);
        total_acc += final_a;
    }

    let res = (total_loss / ds.n as f64, total_acc / ds.n as f64);
    ev.fitness_cache.insert(gkey, res);
    res
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!(
        "ADF-CGP  N_ADF={N_ADF}  L0={LEN_LAYER0}  L1={LEN_LAYER1}  L2={LEN_LAYER2}  POP={POP_SIZE}"
    );

    let ds = Dataset::build();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut pop: Vec<Genome> = (0..POP_SIZE).map(|_| Genome::random(&mut rng)).collect();
    let evaluator = Evaluator::new();

    for gen in 0..N_GEN {
        evaluator.reset_generation();

        let mut scored: Vec<(f64, f64, Genome)> = pop
            .par_iter()
            .map(|g| {
                let (l, a) = eval(g, &ds, &evaluator);
                (l, a, g.clone())
            })
            .collect();

        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        println!(
            "gen {:4}  loss {:.6}  acc {:.6}",
            gen, scored[0].0, scored[0].1
        );

        let elites: Vec<Genome> = scored[..ELITE].iter().map(|x| x.2.clone()).collect();
        let mut next = elites;
        while next.len() < POP_SIZE {
            let p1 = &scored[rng.gen_range(0..ELITE)].2;
            let p2 = &scored[rng.gen_range(0..ELITE)].2;
            let child = match rng.gen_range(0..3u8) {
                0 => p1.mix(&mut rng, p2),
                1 => p1.mutate(&mut rng),
                _ => p1.mix(&mut rng, p2).mutate(&mut rng),
            };
            next.push(child);
        }
        pop = next;
    }
}
