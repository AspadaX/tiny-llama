use anyhow::{Result, anyhow};

/// Computes Root Mean Square Layer Normalization (RMSNorm) for a 1D activation vector.
///
/// RMSNorm is a lightweight alternative to LayerNorm that preserves re-scaling invariance
/// while omitting the mean-centering step. It stabilizes training with lower computational overhead.
///
/// Algorithm flow:
/// 1. Calculate the mean of squared values from the input
/// 2. Add epsilon for numerical stability, then take the square root to get the RMS scalar
/// 3. Divide every input element by the RMS value to normalize scale
/// 4. Optionally apply element-wise learnable gain weights
///
/// # Arguments
/// * `vector` - Input activation slice to normalize
/// * `weight` - Optional learnable gain parameters, must match the input vector length
/// * `epsilon` - Optional small constant to avoid division by zero (defaults to 1e-6)
///
/// # Returns
/// Normalized output vector, or an error if input validation fails
pub fn calculate_rms_norm(
    vector: &[f32],
    weight: Option<&[f32]>,
    epsilon: Option<f32>,
) -> Result<Vec<f32>> {
    if vector.is_empty() {
        return Err(anyhow!("Vector length can't be 0"));
    }

    if let Some(weight) = weight {
        if weight.len() != vector.len() {
            return Err(anyhow!("Lengths of weight and vector mismatched"));
        }
    }

    let epsilon: f32 = match epsilon {
        Some(result) => result,
        None => 1e-6,
    };

    let mean_squre: f32 = vector.iter().map(|item| item * item).sum::<f32>() / vector.len() as f32; // Divded by vector len is the difference between L2Norm and RMSNorm
    let root_mean_square: f32 = (mean_squre + epsilon).sqrt(); // Use the epsilon to prevent invalid calculations

    let vector: Vec<f32> = vector.iter().map(|item| item / root_mean_square).collect();

    Ok(match weight {
        Some(weight) => vector
            .iter()
            .zip(weight)
            .map(|(vector, weight)| vector * weight)
            .collect(),
        None => vector,
    })
}
