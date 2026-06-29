use anyhow::Result;

/// Calculates the RoPE rotation angle θ.
///
/// θ = m * 10000^(-2j / d_head)
///
/// `m = token_position`, `j = pair_position`, `d_head` = feature dimension per attention head.
pub fn calculate_theta(token_position: usize, pair_position: usize, d_head: usize) -> f32 {
    token_position as f32 * f32::powf(10000.0, (-2.0 * pair_position as f32) / d_head as f32)
}

/// Precomputes cos/sin lookup tables for fast RoPE application.
///
/// For every token position and dimension pair:
///
/// cos = cos(m * 10000^(-2j / d_head))
/// sin = sin(m * 10000^(-2j / d_head))
///
/// Linear table index: `index = token_position * num_pairs + pair_position`
/// where `num_pairs = d_head / 2`.
pub fn precompute_theta_tables(max_sequence_len: usize, d_head: usize) -> (Vec<f32>, Vec<f32>) {
    let num_pairs = d_head / 2;
    let mut cos_table = Vec::with_capacity(max_sequence_len * num_pairs);
    let mut sin_table = Vec::with_capacity(max_sequence_len * num_pairs);

    // token_position is the m
    for token_position in 0..max_sequence_len {
        // pair_position is the j, computed by d_head / 2
        for pair_position in 0..num_pairs {
            let theta = calculate_theta(token_position, pair_position, d_head);
            let cos = f32::cos(theta);
            let sin = f32::sin(theta);

            cos_table.push(cos);
            sin_table.push(sin);
        }
    }

    (cos_table, sin_table)
}

/// Applies Rotary Position Embedding to a single attention head vector.
///
/// Rotates each adjacent 2D dimension pair with the precomputed angle:
///
/// x' = x * cos(θ) - y * sin(θ)
/// y' = x * sin(θ) + y * cos(θ)
pub fn calculate_rotary_position_embeddings(
    token_position: usize,
    vector: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
) -> Result<Vec<f32>> {
    let mut vector = vector.to_vec();

    // equal to d_head / 2
    let num_pairs = vector.len() / 2;

    vector
        .chunks_exact_mut(2)
        .enumerate()
        .for_each(|(pair_position, pair)| {
            let index = token_position * num_pairs + pair_position;

            let cos = cos_table[index];
            let sin = sin_table[index];

            let x = pair[0];
            let y = pair[1];

            pair[0] = x * cos - y * sin;
            pair[1] = x * sin + y * cos;
        });

    Ok(vector)
}