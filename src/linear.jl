"""
    linear_model_fitting(X::Matrix{Float64}, y::Vector{Float64}) -> Vector{Float64}

Perform linear regression to fit a linear model y = X Theta.

# Arguments
- `X`: Design matrix (predictor variables).
- `y`: Response vector.

# Returns
- `Θ`: Vector of fitted model coefficients.
"""
function linear_model_fitting(X::Matrix{Float64}, y::Vector{Float64})
    Θ = X \ y
    return Θ
end