// M6: Token sampling for autoregressive generation.
//
// Provides greedy (argmax) and stochastic sampling with temperature,
// top-k, and top-p (nucleus) filtering. Uses a simple XorShift RNG
// to avoid adding the `rand` crate dependency.

/// Sampling configuration for text generation.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for logit scaling. 0.0 = greedy (argmax).
    pub temperature: f32,
    /// Top-K: keep only the top-k highest probability tokens. 0 = disabled.
    pub top_k: usize,
    /// Top-P (nucleus): cumulative probability cutoff. 1.0 = disabled.
    pub top_p: f32,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        }
    }
}

/// Simple XorShift64 RNG to avoid adding the `rand` crate dependency.
pub struct XorShiftRng {
    state: u64,
}

impl XorShiftRng {
    /// Create a new RNG from a seed. Seed of 0 is adjusted to 1.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generate the next u64 value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a random f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Return the index of the maximum value in the logits.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Sample a token from logits using the given configuration.
///
/// Steps:
/// 1. If temperature == 0.0, return argmax (greedy).
/// 2. Apply temperature scaling: logit / temperature.
/// 3. Top-K: sort, keep top-k candidates.
/// 4. Softmax over remaining candidates.
/// 5. Top-P: cumulative probability cutoff, renormalize.
/// 6. Sample from categorical distribution.
pub fn sample_token(logits: &[f32], config: &SamplingConfig, rng: &mut XorShiftRng) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    // Greedy
    if config.temperature <= 0.0 {
        return argmax(logits);
    }

    // Build (index, logit) candidates
    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l / config.temperature))
        .collect();

    // Top-K: sort by logit descending, keep top_k
    if config.top_k > 0 && config.top_k < candidates.len() {
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(config.top_k);
    }

    // Softmax over candidates
    let max_logit = candidates.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(u32, f32)> = candidates
        .iter()
        .map(|&(idx, logit)| (idx, (logit - max_logit).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|c| c.1).sum();
    for c in &mut probs {
        c.1 /= sum;
    }

    // Top-P (nucleus): sort by probability descending, cumulative cutoff
    if config.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumulative = 0.0f32;
        let mut cutoff_idx = probs.len();
        for (i, &(_, p)) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= config.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        probs.truncate(cutoff_idx);

        // Renormalize
        let sum2: f32 = probs.iter().map(|c| c.1).sum();
        for c in &mut probs {
            c.1 /= sum2;
        }
    }

    // Sample from categorical distribution
    let r = rng.next_f32();
    let mut cumulative = 0.0f32;
    for &(idx, p) in &probs {
        cumulative += p;
        if r < cumulative {
            return idx;
        }
    }

    // Fallback: return last candidate
    probs.last().map(|c| c.0).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_basic() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[1.0, 2.0, 5.0]), 2);
    }

    #[test]
    fn test_argmax_negative() {
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
    }

    #[test]
    fn test_argmax_single() {
        assert_eq!(argmax(&[42.0]), 0);
    }

    #[test]
    fn test_greedy_sampling() {
        let logits = vec![1.0, 5.0, 2.0, 3.0];
        let config = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = XorShiftRng::new(42);
        assert_eq!(sample_token(&logits, &config, &mut rng), 1);
    }

    #[test]
    fn test_greedy_deterministic() {
        let logits = vec![1.0, 5.0, 2.0, 3.0];
        let config = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng1 = XorShiftRng::new(42);
        let mut rng2 = XorShiftRng::new(99);
        // Greedy ignores RNG
        assert_eq!(
            sample_token(&logits, &config, &mut rng1),
            sample_token(&logits, &config, &mut rng2)
        );
    }

    #[test]
    fn test_temperature_sampling_deterministic_with_seed() {
        let logits = vec![1.0, 5.0, 2.0, 3.0];
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(42),
        };

        let mut rng1 = XorShiftRng::new(42);
        let mut rng2 = XorShiftRng::new(42);

        let t1 = sample_token(&logits, &config, &mut rng1);
        let t2 = sample_token(&logits, &config, &mut rng2);
        assert_eq!(t1, t2, "Same seed should produce same token");
    }

    #[test]
    fn test_top_k_limits_candidates() {
        // With top_k=1, it should always pick the highest logit
        let logits = vec![1.0, 10.0, 2.0, 3.0];
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            seed: None,
        };
        let mut rng = XorShiftRng::new(42);
        // top_k=1 means only the top token is kept
        assert_eq!(sample_token(&logits, &config, &mut rng), 1);
    }

    #[test]
    fn test_top_p_limits_candidates() {
        // Very low top_p should select only the highest-probability token
        let logits = vec![0.0, 100.0, 0.0, 0.0];
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.01,
            seed: None,
        };
        let mut rng = XorShiftRng::new(42);
        assert_eq!(sample_token(&logits, &config, &mut rng), 1);
    }

    #[test]
    fn test_temperature_affects_distribution() {
        // Very high temperature should make distribution more uniform
        // Very low temperature should make it more peaked
        let logits = vec![1.0, 2.0, 3.0, 4.0];

        // Low temperature: should almost always pick highest
        let config_low = SamplingConfig {
            temperature: 0.01,
            top_k: 0,
            top_p: 1.0,
            seed: Some(42),
        };
        let mut rng = XorShiftRng::new(42);
        let mut count_top = 0;
        for _ in 0..100 {
            if sample_token(&logits, &config_low, &mut rng) == 3 {
                count_top += 1;
            }
        }
        assert!(count_top > 90, "Low temp should favor top token, got {}/100", count_top);
    }

    #[test]
    fn test_combined_top_k_top_p() {
        let logits = vec![1.0, 10.0, 9.0, 0.0];
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 2,
            top_p: 0.5,
            seed: None,
        };
        let mut rng = XorShiftRng::new(42);
        // top_k=2 keeps indices 1,2 (logits 10,9)
        // top_p=0.5 further limits; token 1 has ~73% probability, so it alone exceeds 0.5
        let token = sample_token(&logits, &config, &mut rng);
        assert!(token == 1 || token == 2, "Should only pick from top-2, got {}", token);
    }

    #[test]
    fn test_xorshift_rng_produces_different_values() {
        let mut rng = XorShiftRng::new(42);
        let a = rng.next_u64();
        let b = rng.next_u64();
        let c = rng.next_u64();
        assert_ne!(a, b);
        assert_ne!(b, c);
    }

    #[test]
    fn test_xorshift_rng_f32_range() {
        let mut rng = XorShiftRng::new(42);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0, "next_f32 out of range: {}", v);
        }
    }

    #[test]
    fn test_empty_logits() {
        let config = SamplingConfig::default();
        let mut rng = XorShiftRng::new(42);
        assert_eq!(sample_token(&[], &config, &mut rng), 0);
    }

    #[test]
    fn test_sampling_returns_valid_index() {
        let logits = vec![1.0; 100];
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        let mut rng = XorShiftRng::new(42);
        for _ in 0..100 {
            let token = sample_token(&logits, &config, &mut rng);
            assert!((token as usize) < 100, "Token out of range: {}", token);
        }
    }
}
