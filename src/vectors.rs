use anyhow::{Result, anyhow};

pub fn are_vectors_sizes_match(a: &[f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("Vector lengths must match"));
    }

    Ok(())
}

/// Calculates the dot product of two vectors:
///
/// `a · b = (a₁ × b₁) + (a₂ × b₂) + (a₃ × b₃) + ...`
pub fn calculate_dot_product(a: &[f32], b: &[f32]) -> Result<f32> {
    are_vectors_sizes_match(a, b)?;
    Ok(a.iter().zip(b).map(|(a, b)| a * b).sum())
}

/// Calculates the element-wise sum of two vectors:
///
/// `result = [a₁ + b₁, a₂ + b₂, a₃ + b₃, ...]`
///
/// If the input vectors have different lengths, the result length matches the
/// shorter vector because elements are paired using `zip`.
pub fn sum_vectors(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    are_vectors_sizes_match(a, b)?;
    Ok(a.iter().zip(b).map(|(a, b)| a + b).collect())
}
