use anyhow::{Result, anyhow};

use crate::vectors::{calculate_dot_product, sum_vectors};

/// inputs: x, a vector of a token of the entire sequence; w, num of output size vectors with each having a size of input size; b, a vector represents the bias, which is sized to the output size

/// w looks like this:
/// [
/// [input_size], -> output_size of such vectors in w
/// ...
/// ]

/// 1. perform a dot product calculation between x and w. each vector in w will dot product the x and result in a scalar value. this process will eventually output a vector with output_size, new_x
/// 2. new_x will plus b. this is the result to return
pub fn calculate_linear_layer(vector: &[f32], weight: &[&[f32]], bias: &[f32]) -> Result<Vec<f32>> {
    // Get vector size as the input_size
    let input_size = vector.len();

    // Get weight size as the output_size
    let output_size = weight.len();

    // Validate the size between the output_size and bias
    if output_size != bias.len() {
        return Err(anyhow!("Bias size is different from output size"));
    }

    // Validate the input_size between each vector of weight
    for weight_vector in weight {
        if input_size != weight_vector.len() {
            return Err(anyhow!("Weight vector size is different from input size"));
        }
    }

    // Perform dot product calculations between vector and weight, get a new vector with output_size
    let mut new_vector: Vec<f32> = Vec::with_capacity(output_size);
    for weight_vector in weight {
        new_vector.push(calculate_dot_product(vector, weight_vector)?);
    }

    // Plus the new vector with bias, get a new vector
    // Return the new vector
    Ok(sum_vectors(&new_vector, bias)?)
}
