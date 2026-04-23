use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct NearestBetterResult {
    /// ans[i] = 自分より高スコアでハミング距離最小の個体 index
    /// 候補がなければ None
    pub ans: Vec<Option<usize>>,
}

#[inline]
fn hamming_distance(a: &[usize], b: &[usize]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

#[inline]
fn freq_bucket(f: usize) -> u8 {
    if f == 0 {
        0
    } else {
        // 1 -> 1, 2..3 -> 2, 4..7 -> 3, ...
        (usize::BITS - f.leading_zeros()) as u8
    }
}

/// strict: score[j] > score[i] のみ許可
///
/// 制約:
/// - 全配列の長さは同じ
/// - arrs[k][i] は 0..=i の範囲
pub fn nearest_better_hamming_pruned_cached(
    arrs: &[Vec<usize>],
    scores: &[f64],
    max_cache_size: usize,
) -> NearestBetterResult {
    let n = arrs.len();
    assert_eq!(n, scores.len(), "arrs.len() must equal scores.len()");
    if n == 0 {
        return NearestBetterResult { ans: vec![] };
    }

    let l = arrs[0].len();
    for a in arrs.iter() {
        assert_eq!(a.len(), l, "all arrays must have the same length");
    }
    for x in arrs.iter() {
        for (i, &v) in x.iter().enumerate() {
            assert!(
                v <= i,
                "arr value out of range: x[{}]={}, expected 0..={}",
                i,
                v,
                i
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

    // postings[i][v] = 既登録個体 index の配列
    let mut postings: Vec<Vec<Vec<usize>>> =
        (0..l).map(|i| vec![Vec::<usize>::new(); i + 1]).collect();

    // freqs[i][v] = 現在までに insert された個体のうち、座標 i が v の数
    let mut freqs: Vec<Vec<usize>> = (0..l).map(|i| vec![0usize; i + 1]).collect();

    // query ごとの遅延初期化
    let mut count: Vec<usize> = vec![0; n];
    let mut seen_ver: Vec<u32> = vec![0; n];
    let mut active_mark: Vec<u32> = vec![0; n];
    let mut query_ver: u32 = 0;

    // signature -> coords order
    let mut order_cache: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();

    let mut get_coord_order = |x: &[usize],
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

    let mut query_one = |x: &[usize],
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

        // 先に検索
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

        // 後で一括挿入
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

pub fn verify_answers_bruteforce_strict(
    arrs: &[Vec<usize>],
    scores: &[f64],
    ans: &[Option<usize>],
) -> bool {
    let n = arrs.len();
    if scores.len() != n || ans.len() != n {
        return false;
    }

    for i in 0..n {
        let mut candidates = Vec::new();
        for j in 0..n {
            if i != j && scores[j] > scores[i] {
                candidates.push(j);
            }
        }

        if candidates.is_empty() {
            if ans[i].is_some() {
                return false;
            }
            continue;
        }

        let best_d = candidates
            .iter()
            .map(|&j| hamming_distance(&arrs[i], &arrs[j]))
            .min()
            .unwrap();

        let valid = candidates
            .iter()
            .any(|&j| Some(j) == ans[i] && hamming_distance(&arrs[i], &arrs[j]) == best_d);

        if !valid {
            return false;
        }
    }

    true
}

// 簡単な乱数インスタンス生成
pub fn make_random_instance(n: usize, l: usize, seed: u64) -> (Vec<Vec<usize>>, Vec<f64>) {
    // 外部 crate を使わず簡易 LCG
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);

    fn next_u64(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    let mut arrs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut x = Vec::with_capacity(l);
        for i in 0..l {
            let v = (next_u64(&mut state) % ((i + 1) as u64)) as usize;
            x.push(v);
        }
        arrs.push(x);
    }

    let mut scores = Vec::with_capacity(n);
    for _ in 0..n {
        let r = next_u64(&mut state);
        let s = (r as f64) / (u64::MAX as f64);
        scores.push(s);
    }

    (arrs, scores)
}

fn main() {
    let (arrs, scores) = make_random_instance(2048, 2048, 42);

    let time1 = std::time::Instant::now();
    let result = nearest_better_hamming_pruned_cached(&arrs, &scores, 16384);
    let time2 = std::time::Instant::now();
    println!("time: {:?}", time2 - time1);

    let ok = verify_answers_bruteforce_strict(&arrs, &scores, &result.ans);
    println!("verified: {}", ok);

    /*for i in 0..1024 {
        match result.ans[i] {
            Some(j) => {
                let d = hamming_distance(&arrs[i], &arrs[j]);
                println!(
                    "{} -> {}  score[i]={:.6} score[j]={:.6} hamming={}",
                    i, j, scores[i], scores[j], d
                );
            }
            None => {
                println!("{} -> None", i);
            }
        }
    }*/
}
