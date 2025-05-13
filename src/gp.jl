"""
    morelli_mean(α::Float64, δₑ::Float64, Q::Float64, M::Float64, V::Float64) -> Float64

Compute the pitch moment coefficient C_m using the Morelli generic global aerodynamic model.

# Arguments
- `α`: Angle of attack (radians).
- `δₑ`: Elevator deflection (radians).
- `Q`: Pitch rate (rad per second).
- `M`: Mach number.
- `V`: Airspeed (ft/s).

# Returns
- `C_m`: Pitch moment coefficient.

# Notes
Uses parameters from the Morelli model for the A-7 (or F-16) aircraft.
"""
function morelli_mean(α, δₑ, Q, M, V)
    # Source: Grauer, Jared A., and Eugene A. Morelli. "Generic global aerodynamic model for aircraft." Journal of Aircraft 52.1 (2015): 13-20.

    # Constants

    # Parameters for the F-16C
    # weight = 20500 # weight [lb]
    # g = 32.2       # acceleration due to gravity [ft/s^2]
    # Ixx = 9496     # moment of inertia about x-axis [slug ft^2]
    # Iyy = 55814    # moment of inertia about y-axis [slug ft^2]
    # Izz = 63100    # moment of inertia about z-axis [slug ft^2]
    # Ixz = 982      # moment of inertia about xz-axis [slug ft^2]
    # S = 300        # wing reference area [ft^2]
    # c̄ = 11.32      # mean aerodynamic chord [ft]
    # b = 30         # wingspan [ft]
    # xcm_c̄ = 0.25 
    # xref_c̄ = 0.35

    # θ = [0.034, -0.005, 20.77, 0.177, 1.285, -19.97, 0.756, 5.887, 55.59, −5.155, -1.146, -0.188, 0.876, 0.060, 0.164, 0.074, 4.458, 29.90, 0.412, −5.538, -2.477, −1.101, 1.906,
    #     -0.071, -0.445, 0.058, -0.143, 0.023, -0.024, -0.288, -8.267, -0.563, -5.513, 9.793, -1.057, -2.018, 1.897, -0.094, 0.234, 0.056, -0.418, -0.034, -0.085, 0.372, -0.725]

    # Parameters for the A-7
    weight = 22699 # weight [lb]
    g = 32.2       # acceleration due to gravity [ft/s^2]
    Ixx = 16970    # moment of inertia about x-axis [slug ft^2]
    Iyy = 65430   # moment of inertia about y-axis [slug ft^2]
    Izz = 76130    # moment of inertia about z-axis [slug ft^2]
    Ixz = 4030      # moment of inertia about xz-axis [slug ft^2]
    S = 375        # wing reference area [ft^2]
    c̄ = 10.8      # mean aerodynamic chord [ft]
    b = 38.7       # wingspan [ft]
    xcm_c̄ = 0.30
    xref_c̄ = 0.35

    θ = [0.006, 0.32, 0.0, 0.074, -2.519, 0.0, 0.439, 19.76, 0.0, -22.1, -1.084, 0.03, 0.059, 0.099, 0.268, -0.093, 4.412, 0.0, 0.549, 0.0, 0.817, -9.833, 3.851, -0.054, 
        -0.313, 0.031, -0.137, 0.004, -0.023, -0.810, -7.033, -1.032, 0.502, 8.007, 1.215, 17.15, -1.278, -1.969, 0.102, 0.06, -0.294, -0.02, -0.121, 0.557, -0.923] 
    
    q̃ = Q*c̄/(2*V)

    # Calculate aerodynamic coefficients
    C_m = θ[29] + θ[30]*α + θ[31]*q̃ + θ[32]*δₑ + θ[33]*q̃*α + θ[34]*q̃*α^2 + θ[35]*δₑ*α^2 + θ[36]*q̃*α^3 + θ[37]*δₑ*α^3 + θ[38]*α^4
    # moment = C_m * S * c̄ * (0.5 * V^2)
    # C_m = C_m + (C_L * (xref_c̄ - xcm_c̄))
    return C_m
end


"""
    GP_regression(X::Matrix{Float64}, y::Vector{Float64}, morelli_mean_function::Function) -> (...)

Perform Gaussian Process regression on the data with a custom mean function based on the Morelli model.

# Arguments
- `X`: Design matrix (predictor variables).
- `y`: Response vector.
- `morelli_mean_function`: Function to compute the mean (Morelli aerodynamic model).

# Returns
- `p_fx`: Posterior GP model.
- `custom_mean_function`: The custom mean function used in the GP.
- `X_train`: Training input data (scaled).
- `scaler`: Scaler object used for data normalization.
- Scaling functions `cm_scale_x`, `cm_unscale_x`, `scale_Cm`, `unscale_Cm`.
- `cm_scale_factor`: Scaling factor for the response variable.
"""
function GP_regression(X::Matrix{Float64}, y::Vector{Float64}, morelli_mean_function::Function)
    # Data scaling
    Z = hcat(y, X)
    scaler = fit(UnitRangeTransform, Z, dims=1)
    # scaler = fit(ZScoreTransform, Z, dims=1)

    Z_scaled = StatsBase.transform(scaler, Z)
    X_scaled = Z_scaled[:, 2:end]
    y_scaled = Z_scaled[:, 1]

    # Scaling functions
    cm_scale_x(x) = scaler isa UnitRangeTransform ? (x .- scaler.min[2:end]) .* scaler.scale[2:end] : (x .- scaler.mean[2:end]) ./ scaler.scale[2:end]
    cm_unscale_x(x) = scaler isa UnitRangeTransform ? x ./ scaler.scale[2:end] .+ scaler.min[2:end] : x .* scaler.scale[2:end] .+ scaler.mean[2:end]
    scale_Cm(Cm) = scaler isa UnitRangeTransform ? (Cm .- scaler.min[1]) * scaler.scale[1] : (Cm .- scaler.mean[1]) / scaler.scale[1]
    unscale_Cm(Cm) = scaler isa UnitRangeTransform ? (Cm / scaler.scale[1] .+ scaler.min[1]) : (Cm * scaler.scale[1] .+ scaler.mean[1])
    cm_scale_factor = scaler isa UnitRangeTransform ? scaler.scale[1] : 1 / scaler.scale[1]
   
    # Create custom mean function
    function create_custom_mean_function(morelli_mean::Function)
        return AbstractGPs.CustomMean((x::AbstractVector) -> begin
            x_unscaled = cm_unscale_x(x)
            # Get airspeed in ft/s from dynamic pressure
            rho = x_unscaled[2]
            dyn_press = x_unscaled[3]
            airspeed = sqrt(2 * dyn_press / rho)
            Cm = morelli_mean(x_unscaled[7], x_unscaled[8], x_unscaled[5], x_unscaled[1], airspeed)
            Cm_scaled = scale_Cm(Cm)
            return Cm_scaled
        end)
    end

    # Loss function for optimization
    # function loss_function(x, y, custom_mean_function)
    #     function negativelogmarginallikelihood(params)
    #         kernel = NeuralNetworkKernel()
    #         f = GP(custom_mean_function, kernel)
    #         fx = f(x, 0.1)
    #         return -logpdf(fx, y)
    #     end
    #     return negativelogmarginallikelihood
    # end

    X_train = RowVecs(X_scaled)
    Y_train = y_scaled
    θ0 = randn(2)
    custom_mean_function = create_custom_mean_function(morelli_mean_function)
    # opt = Optim.optimize(loss_function(X_train, Y_train, custom_mean_function), θ0, LBFGS(), Optim.Options(time_limit=60))


    # Build GP model
    kernel = NeuralNetworkKernel()
    noise_var = 0.1
    f = AbstractGPs.GP(custom_mean_function, kernel)
    fx = f(X_train, noise_var)
    p_fx = AbstractGPs.posterior(fx, Y_train)

    # println("Optimized: ", logpdf(fx, Y_train))

    return p_fx, custom_mean_function, X_train, scaler, cm_scale_x, cm_unscale_x, scale_Cm, unscale_Cm, cm_scale_factor
end

"""
    compute_Cz_alpha(Z::Vector{Float64}, X::Matrix{Float64}, left_fuel_qty::Vector{Float64},
                     right_fuel_qty::Vector{Float64}, dyn_press::Vector{Float64}, S::Float64,
                     trim_state::Vector{Float64}) -> (...)

Compute the aerodynamic force coefficient C_Z and its derivative with respect to angle of attack using both linear regression and Gaussian Process regression.

# Arguments
- `Z`: Vector of normal accelerations.
- `X`: Design matrix (predictor variables).
- `left_fuel_qty`, `right_fuel_qty`: Fuel quantities (lbs).
- `dyn_press`: Dynamic pressure values.
- `S`: Wing reference area (ft²).
- `trim_state`: State vector at trim condition.

# Returns
- Functions `μf_unscaled_cz`, `σf_unscaled_cz`, `μf_scaled_cz`, `σf_scaled_cz`: Mean and standard deviation functions for C_Z.
- `Z_alpha_gp`: Stability derivative Z_alpha from GP.
- `Z_alpha_lin`: Stability derivative Z_alpha from linear model.
- `Cz_alpha_lin`: C_{Z_alpha} from linear regression.
- Scaling functions `cz_scale_x`, `cz_unscale_x`.
- `cz_scale_factor`: Scaling factor for C_Z.
- `scaler_cz`: Scaler object used for C_Z data normalization.
"""
function compute_Cz_alpha(Z::Vector{Float64}, X::Matrix{Float64}, left_fuel_qty::Vector{Float64}, right_fuel_qty::Vector{Float64}, dyn_press::Vector{Float64}, S::Float64, trim_state::Vector{Float64})
    m = empty_weight .+ left_fuel_qty .+ right_fuel_qty
    Z_force = -g .* Z .* m
    C_Z = Z_force ./ (dyn_press .* S)

    # Linear model fitting
    X_cz = X
    Θ_cz = X_cz \ C_Z
    Cz_alpha_lin = Θ_cz[7]
    Cz_δe_lin = Θ_cz[8]

    # Gaussian Process Regression for C_Z
    Z_cz = hcat(C_Z, X_cz)
    scaler_cz = fit(UnitRangeTransform, Z_cz, dims=1)
    # scaler_cz = fit(ZScoreTransform, Z_cz, dims=1)

    Z_cz_scaled = StatsBase.transform(scaler_cz, Z_cz)
    X_cz_scaled = Z_cz_scaled[:, 2:end]
    y_cz_scaled = Z_cz_scaled[:, 1]

    # Scaling functions
    cz_scale_x(x) = scaler_cz isa UnitRangeTransform ? (x .- scaler_cz.min[2:end]) .* scaler_cz.scale[2:end] : (x .- scaler_cz.mean[2:end]) ./ scaler_cz.scale[2:end]
    cz_unscale_x(x) = scaler_cz isa UnitRangeTransform ? x ./ scaler_cz.scale[2:end] .+ scaler_cz.min[2:end] : x .* scaler_cz.scale[2:end] .+ scaler_cz.mean[2:end]
    scale_Cz(Cz) = scaler_cz isa UnitRangeTransform ? (Cz .- scaler_cz.min[1]) * scaler_cz.scale[1] : (Cz .- scaler_cz.mean[1]) / scaler_cz.scale[1]
    unscale_Cz(Cz) = scaler_cz isa UnitRangeTransform ? Cz / scaler_cz.scale[1] .+ scaler_cz.min[1] : Cz * scaler_cz.scale[1] .+ scaler_cz.mean[1]
    cz_scale_factor = scaler_cz isa UnitRangeTransform ? scaler_cz.scale[1] : 1 / scaler_cz.scale[1]

    # GP model for C_Z
    X_cz_train = RowVecs(X_cz_scaled)
    Y_cz_train = y_cz_scaled
    θ0_cz = randn(2)
    custom_mean_function_cz = AbstractGPs.ZeroMean()
    # function loss_function_cz(x, y, custom_mean_function)
    #     function negativelogmarginallikelihood(params)
    #         kernel = NeuralNetworkKernel()
    #         f = GP(custom_mean_function, kernel)
    #         fx = f(x, 0.1)
    #         return -logpdf(fx, y)
    #     end
    #     return negativelogmarginallikelihood
    # end
    # opt_cz = Optim.optimize(loss_function_cz(X_cz_train, Y_cz_train, custom_mean_function_cz), θ0_cz, LBFGS(), Optim.Options(time_limit=60))

    kernel_cz = NeuralNetworkKernel()
    noise_var = 0.1
    f_cz = AbstractGPs.GP(custom_mean_function_cz, kernel_cz)
    fx_cz = f_cz(X_cz_train, noise_var)
    p_fx_cz = posterior(fx_cz, Y_cz_train)

    # Define mean function and its gradient
    μf_unscaled_cz(x) = unscale_Cz(mean(p_fx_cz([cz_scale_x(x)]))[1])
    μf_scaled_cz(x) = mean(p_fx_cz([x]))[1]
    σf_unscaled_cz(x) = sqrt.(var(p_fx_cz([cz_scale_x(x)]))[1]) / cz_scale_factor
    σf_scaled_cz(x) = sqrt.(var(p_fx_cz([x]))[1])

    # Compute gradients
    grad_μ_cz = Zygote.gradient(μf_unscaled_cz, trim_state)[1]
    Cz_alpha_gp = grad_μ_cz[7]
    Cz_δe_gp = grad_μ_cz[8]
    Cz_q_gp = grad_μ_cz[5]

    Z_alpha_gp = Cz_alpha_gp * dyn_press[1] * S / m[1]
    Z_alpha_lin = Cz_alpha_lin * dyn_press[1] * S / m[1]

    return μf_unscaled_cz, σf_unscaled_cz, μf_scaled_cz, σf_scaled_cz, Z_alpha_gp, Z_alpha_lin, Cz_alpha_lin, cz_scale_x, cz_unscale_x, cz_scale_factor, scaler_cz
end

"""
    compute_force_coefficients(data::DataFrame, X::Matrix{Float64}, left_fuel_qty::Vector{Float64},
                              right_fuel_qty::Vector{Float64}, dyn_press::Vector{Float64}, S::Float64,
                              trim_state::Vector{Float64}) -> (C_X, C_Y, C_Z)

Compute the aerodynamic force coefficients C_X, C_Y, and C_Z from accelerations.

# Arguments
- `data`: DataFrame containing the flight data.
- `X`: Design matrix (predictor variables).
- `left_fuel_qty`, `right_fuel_qty`: Fuel quantities (lbs).
- `dyn_press`: Dynamic pressure values.
- `S`: Wing reference area (ft²).
- `trim_state`: State vector at trim condition.

# Returns
- `C_X`, `C_Y`, `C_Z`: Force coefficients.
"""
function compute_force_coefficients(data::DataFrame, X::Matrix{Float64}, left_fuel_qty::Vector{Float64},
                                  right_fuel_qty::Vector{Float64}, dyn_press::Vector{Float64}, S::Float64,
                                  trim_state::Vector{Float64})
    m = empty_weight .+ left_fuel_qty .+ right_fuel_qty

    # Compute forces from accelerations
    X_force = g .* data[!, "NX_LONG_ACCEL"] .* m
    Y_force = g .* data[!, "NY_LATERAL_ACCEL"] .* m
    Z_force = -g .* data[!, "NZ_NORMAL_ACCEL"] .* m

    # Compute force coefficients
    C_X = X_force ./ (dyn_press .* S)
    C_Y = Y_force ./ (dyn_press .* S)
    C_Z = Z_force ./ (dyn_press .* S)

    return C_X, C_Y, C_Z
end
