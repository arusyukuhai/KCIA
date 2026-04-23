use ndarray::{s, Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

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
    RawScalar,
    SinCos,
}

#[derive(Clone)]
struct CondEmbedding {
    kind: CondEncodingKind,
    dim: usize,
    max_period: f32,
}

#[derive(Clone)]
struct CondEmbeddingCache {
    cond: f32,
    emb: Array1<f32>,
}

impl CondEmbedding {
    fn new_raw_scalar() -> Self {
        Self {
            kind: CondEncodingKind::RawScalar,
            dim: 1,
            max_period: 10000.0,
        }
    }

    fn new_sincos(dim: usize, max_period: f32) -> Self {
        assert!(dim >= 2);
        Self {
            kind: CondEncodingKind::SinCos,
            dim,
            max_period,
        }
    }

    fn output_dim(&self) -> usize {
        self.dim
    }

    fn forward(&self, cond: f32) -> (Array1<f32>, CondEmbeddingCache) {
        let emb = match self.kind {
            CondEncodingKind::RawScalar => Array1::from_vec(vec![cond]),
            CondEncodingKind::SinCos => {
                let half = self.dim / 2;
                let mut out = Array1::<f32>::zeros(self.dim);

                for i in 0..half {
                    let frac = i as f32 / half as f32;
                    let freq = self.max_period.powf(-frac);
                    let x = cond * freq;
                    out[i] = x.sin();
                    out[half + i] = x.cos();
                }

                if self.dim % 2 == 1 {
                    out[self.dim - 1] = 0.0;
                }
                out
            }
        };

        (emb.clone(), CondEmbeddingCache { cond, emb })
    }

    fn backward(&self, cache: &CondEmbeddingCache, demb: &Array1<f32>) -> f32 {
        match self.kind {
            CondEncodingKind::RawScalar => demb[0],
            CondEncodingKind::SinCos => {
                let half = self.dim / 2;
                let mut dcond = 0.0;
                for i in 0..half {
                    let frac = i as f32 / half as f32;
                    let freq = self.max_period.powf(-frac);
                    let x = cache.cond * freq;
                    dcond += demb[i] * x.cos() * freq;
                    dcond += demb[half + i] * (-x.sin()) * freq;
                }
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
    n_mats: Vec<Array2<f32>>, // each is [2n, n]
    m_mat: Array2<f32>,       // [2m, m]
    cond: f32,
}

#[derive(Clone)]
struct Targets {
    n_targets: Vec<Array2<f32>>, // each is [2n, n]
    m_target: Array2<f32>,       // [2m, m]
}

struct ModelOutput {
    n_logits: Vec<Array2<f32>>,
    m_logits: Array2<f32>,
}

struct ForwardCache {
    enc_n_caches: Vec<BilinearCache>,
    enc_m_cache: BilinearCache,
    cond_cache: CondEmbeddingCache,
    block_caches: Vec<MixerBlockCache>,
    dec_n_caches: Vec<BilinearCache>,
    dec_m_cache: BilinearCache,
}

// ============================================================
// full model
// ============================================================

struct MixerModel {
    num_n: usize,
    n: usize,
    m: usize,
    d1: usize,
    d2: usize,

    enc_n: BilinearMap, // [2n, n] -> [d1, d2]
    dec_n: BilinearMap, // [d1, d2] -> [2n, n]
    enc_m: BilinearMap, // [2m, m] -> [d1, d2]
    dec_m: BilinearMap, // [d1, d2] -> [2m, m]

    cond_embed: CondEmbedding,
    blocks: Vec<MixerBlock>,
}

impl MixerModel {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rng: &mut StdRng,
        num_n: usize,
        n: usize,
        m: usize,
        d1: usize,
        d2: usize,
        num_blocks: usize,
        token_hidden: usize,
        channel_hidden: usize,
        cond_embed: CondEmbedding,
        cond_proj_hidden: usize,
        cond_vec_dim: usize,
    ) -> Self {
        let scale_n = (2.0 / ((2 * n) + d1 + n + d2) as f32).sqrt();
        let scale_m = (2.0 / ((2 * m) + d1 + m + d2) as f32).sqrt();
        let scale_inner = 0.05;

        let enc_n = BilinearMap::new(rng, 2 * n, n, d1, d2, scale_n);
        let dec_n = BilinearMap::new(rng, d1, d2, 2 * n, n, scale_n);

        let enc_m = BilinearMap::new(rng, 2 * m, m, d1, d2, scale_m);
        let dec_m = BilinearMap::new(rng, d1, d2, 2 * m, m, scale_m);

        let tokens = (num_n + 1) * d1;
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
            num_n,
            n,
            m,
            d1,
            d2,
            enc_n,
            dec_n,
            enc_m,
            dec_m,
            cond_embed,
            blocks,
        }
    }

    fn zero_grad(&mut self) {
        self.enc_n.zero_grad();
        self.dec_n.zero_grad();
        self.enc_m.zero_grad();
        self.dec_m.zero_grad();
        for b in &mut self.blocks {
            b.zero_grad();
        }
    }

    fn scale_gradients(&mut self, s: f32) {
        self.enc_n.scale_grad(s);
        self.dec_n.scale_grad(s);
        self.enc_m.scale_grad(s);
        self.dec_m.scale_grad(s);
        for b in &mut self.blocks {
            b.scale_grad(s);
        }
    }

    fn step(&mut self, hp: OptimHyper) {
        self.enc_n.step(hp);
        self.dec_n.step(hp);
        self.enc_m.step(hp);
        self.dec_m.step(hp);
        for b in &mut self.blocks {
            b.step(hp);
        }
    }

    fn forward(&self, sample: &Sample) -> (ModelOutput, ForwardCache) {
        assert_eq!(sample.n_mats.len(), self.num_n);

        let mut parts: Vec<Array2<f32>> = Vec::new();
        let mut enc_n_caches = Vec::new();

        for x in &sample.n_mats {
            let (blk, cache) = self.enc_n.forward(x);
            parts.push(blk);
            enc_n_caches.push(cache);
        }

        let (m_blk, enc_m_cache) = self.enc_m.forward(&sample.m_mat);
        parts.push(m_blk);

        let total_tokens = (self.num_n + 1) * self.d1;
        let mut z = Array2::<f32>::zeros((total_tokens, self.d2));
        for (i, part) in parts.iter().enumerate() {
            let start = i * self.d1;
            let end = start + self.d1;
            z.slice_mut(s![start..end, ..]).assign(part);
        }

        let (cond_emb, cond_cache) = self.cond_embed.forward(sample.cond);

        let mut block_caches = Vec::new();
        for block in &self.blocks {
            let (next, cache) = block.forward(&z, &cond_emb);
            z = next;
            block_caches.push(cache);
        }

        let mut n_logits = Vec::new();
        let mut dec_n_caches = Vec::new();

        for i in 0..self.num_n {
            let start = i * self.d1;
            let end = start + self.d1;
            let blk = z.slice(s![start..end, ..]).to_owned();
            let (mat, cache) = self.dec_n.forward(&blk);
            n_logits.push(mat);
            dec_n_caches.push(cache);
        }

        let start = self.num_n * self.d1;
        let end = start + self.d1;
        let blk_m = z.slice(s![start..end, ..]).to_owned();
        let (m_logits, dec_m_cache) = self.dec_m.forward(&blk_m);

        (
            ModelOutput { n_logits, m_logits },
            ForwardCache {
                enc_n_caches,
                enc_m_cache,
                cond_cache,
                block_caches,
                dec_n_caches,
                dec_m_cache,
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
        let mut dz = Array2::<f32>::zeros(((self.num_n + 1) * self.d1, self.d2));

        for i in 0..self.num_n {
            let (loss_i, dlogits) =
                Self::bce_with_logits_loss(&output.n_logits[i], &target.n_targets[i]);
            total_loss += loss_i;

            let dblk = self.dec_n.backward(&cache.dec_n_caches[i], &dlogits);
            let start = i * self.d1;
            let end = start + self.d1;
            let mut sl = dz.slice_mut(s![start..end, ..]);
            sl += &dblk;
        }

        let (loss_m, dlogits_m) = Self::bce_with_logits_loss(&output.m_logits, &target.m_target);
        total_loss += loss_m;

        let dblk_m = self.dec_m.backward(&cache.dec_m_cache, &dlogits_m);
        let start_m = self.num_n * self.d1;
        let end_m = start_m + self.d1;
        {
            let mut sl = dz.slice_mut(s![start_m..end_m, ..]);
            sl += &dblk_m;
        }

        let mut dcond_emb = Array1::<f32>::zeros(self.cond_embed.output_dim());

        for i in (0..self.blocks.len()).rev() {
            let (next_dz, dc) = self.blocks[i].backward(&cache.block_caches[i], &dz);
            dz = next_dz;
            dcond_emb += &dc;
        }

        let _dcond_scalar = self.cond_embed.backward(&cache.cond_cache, &dcond_emb);

        for i in 0..self.num_n {
            let start = i * self.d1;
            let end = start + self.d1;
            let dblk = dz.slice(s![start..end, ..]).to_owned();
            let _dx = self.enc_n.backward(&cache.enc_n_caches[i], &dblk);
        }

        let dblk_m2 = dz.slice(s![start_m..end_m, ..]).to_owned();
        let _dx_m = self.enc_m.backward(&cache.enc_m_cache, &dblk_m2);

        let denom = (self.num_n + 1) as f32;
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
        let ns = out
            .n_logits
            .into_iter()
            .map(|x| x.mapv(sigmoid_scalar))
            .collect::<Vec<_>>();
        let m = out.m_logits.mapv(sigmoid_scalar);
        (ns, m)
    }
}

// ============================================================
// dummy data
// ============================================================

fn make_dummy_data(
    num_n: usize,
    n: usize,
    m: usize,
    num_samples: usize,
    rng: &mut StdRng,
) -> Vec<(Sample, Targets)> {
    let mut out = Vec::new();

    for _ in 0..num_samples {
        let mut cond: f32 = StandardNormal.sample(rng);
        cond = cond.tanh();

        let mut n_mats = Vec::new();
        let mut n_targets = Vec::new();

        for _ in 0..num_n {
            let x = Array2::from_shape_fn((2 * n, n), |_| {
                let z: f32 = StandardNormal.sample(rng);
                z
            });
            let tgt = x.mapv(|v| if v + 0.25 * cond > 0.0 { 1.0 } else { 0.0 });
            n_mats.push(x);
            n_targets.push(tgt);
        }

        let m_mat = Array2::from_shape_fn((2 * m, m), |_| {
            let z: f32 = StandardNormal.sample(rng);
            z
        });
        let m_target = m_mat.mapv(|v| if v - 0.15 * cond > 0.0 { 1.0 } else { 0.0 });

        out.push((
            Sample {
                n_mats,
                m_mat,
                cond,
            },
            Targets {
                n_targets,
                m_target,
            },
        ));
    }

    out
}

// ============================================================
// main
// ============================================================

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    let num_n = 3;
    let n = 256; // repeated inputs are [2n, n] = [16, 8]
    let m = 128; // special input is [2m, m] = [10, 5]

    let d1 = 64;
    let d2 = 64;

    let num_blocks = 2;
    let token_hidden = 256;
    let channel_hidden = 256;

    // condition embedding / projection
    let cond_embed = CondEmbedding::new_sincos(32, 10000.0);
    let cond_proj_hidden = 64; // per-block projection hidden size
    let cond_vec_dim = 32; // projected condition size fed to each AdaLN

    let epochs = 20;
    let batch_size = 8;

    let hp = OptimHyper {
        // Muon for 2D params
        lr_matrix: 2e-2,
        muon_weight_decay: 1e-2,
        muon_momentum: 0.95,
        muon_ns_steps: 5,
        muon_nesterov: true,

        // AdamW for 1D params
        lr_vector: 1e-3,
        adamw_beta1: 0.9,
        adamw_beta2: 0.95,
        adamw_eps: 1e-8,
        adamw_weight_decay: 1e-2,
    };

    let mut model = MixerModel::new(
        &mut rng,
        num_n,
        n,
        m,
        d1,
        d2,
        num_blocks,
        token_hidden,
        channel_hidden,
        cond_embed,
        cond_proj_hidden,
        cond_vec_dim,
    );

    let train = make_dummy_data(num_n, n, m, 64, &mut rng);

    for epoch in 0..epochs {
        let mut loss_sum = 0.0;
        let mut num_batches = 0usize;

        for batch in train.chunks(batch_size) {
            loss_sum += model.train_minibatch_step(batch, hp);
            num_batches += 1;
        }

        println!("epoch={epoch:02} loss={:.6}", loss_sum / num_batches as f32);
    }

    let (sample, _) = &train[0];
    let pred = model.predict(sample);
    let (n_probs, m_probs) = model.predict_probs(sample);

    println!("n_logits[0].shape = {:?}", pred.n_logits[0].raw_dim());
    println!("m_logits.shape    = {:?}", pred.m_logits.raw_dim());
    println!("n_probs[0][0,0]   = {:.4}", n_probs[0][[0, 0]]);
    println!("m_probs[0,0]      = {:.4}", m_probs[[0, 0]]);
}
