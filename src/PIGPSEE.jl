module PIGPSEE

using CSV
using DataFrames
using Interpolations
using Plots
using Statistics
using ISAData, Unitful
using LaTeXStrings
using StatsBase
using AbstractGPs
using Optim
using PrettyTables
using Zygote
using PlotlyJS
using LinearAlgebra
using Distributions 
import AbstractGPs: mean_vector
import Unitful: ft, lb, deg2rad

# Constants
const γ = 1.4           # Ratio of specific heats for air
const R = 287.05        # Specific gas constant for air in J/(kg·K)
const g = 32.1740485564 # Acceleration due to gravity in ft/s²
const knot_to_ft_s = 1.68781 # Conversion factor from knots to ft/s
const empty_weight = 8750.0 # lbs
const S = 170.0  # Wing area in square feet
const c̄ = 7.79   # Mean Aerodynamic Chord (MAC) in feet
const event_numbers = [2, 7, 12, 20, 23] # Events to analyze
const mps_to_ft_s = 3.28084 # Conversion factor from m/s to ft/s
const psf_to_pa = 47.880258888889 # Conversion factor from psf to Pa
const kgm3_to_slugft3 = 0.0019403203 # convert kg/m³ to slug/ft³

include("gp.jl")
include("linear.jl")
include("t38.jl")
include("trim.jl")
include("utils.jl")


"""
    run()

Main function that performs the entire analysis pipeline:
- Loads and preprocesses data.
- Computes moments of inertia and moments.
- Fits linear and Gaussian Process models.
- Computes aerodynamic derivatives.
- Generates tables and plots.

# Arguments
- None.

# Returns
- None.
"""
function run()
    # Load and preprocess data
    file_path = joinpath(@__DIR__, "data", "Rollercoasters_20240821.csv")
    filename = file_path
    data_tuple = load_and_preprocess_data(filename, event_numbers)
    (data, time, alpha_rad, mach, airspeed, roll_rad, pitch_rad, P_rad, Q_rad, R_rad, stab_pos_rad,
     press_alt_ic, AMB_AIR_TEMP_C, left_fuel_qty, right_fuel_qty, V_ft_s, rho,
     dyn_press, fuel_weight, total_weight) = data_tuple

    # Compute moments of inertia
    Ixx, Iyy, Izz, Ixz = compute_inertia(left_fuel_qty, right_fuel_qty)

    # Compute Q_dot
    Q_dot = compute_Q_dot(Q_rad, time)

    # Compute moments
    moments = compute_moments(Ixx, Iyy, Izz, Ixz, Q_dot, P_rad, R_rad)
    C_m = moments ./ (dyn_press .* S .* c̄)

    # Linear model fitting
    X = hcat(mach, rho, dyn_press, P_rad, Q_rad, R_rad, alpha_rad, stab_pos_rad)
    Θ = linear_model_fitting(X, C_m)
    Cm_alpha_lin = Θ[7]
    Cm_δe_lin = Θ[8]
    U1 = V_ft_s[1]
    Θ5_lin = Θ[5]
    Cm_q_lin = Θ5_lin * 2 * U1 / c̄

    # Gaussian Process Regression for moment coefficient
    p_fx, custom_mean_function, X_train, scaler, cm_scale_x, cm_unscale_x, scale_Cm, unscale_Cm, cm_scale_factor = GP_regression(X, C_m, morelli_mean)

    # Define mean function and its gradient for moment coefficient
    μf_unscaled(x) = unscale_Cm(mean(p_fx([cm_scale_x(x)]))[1])
    μf_scaled(x) = mean(p_fx([x]))[1]
    σf_unscaled(x) = sqrt.(var(p_fx([cm_scale_x(x)]))[1]) / cm_scale_factor
    σf_scaled(x) = sqrt.(var(p_fx([x]))[1])

    # Point at which to compute the gradient
    trim_state = X[1, :]
    grad_μ = Zygote.gradient(μf_unscaled, trim_state)[1]
    Cm_alpha_gp = grad_μ[7]
    Cm_δe_gp = grad_μ[8]
    Cm_q_gp = grad_μ[5] * 2 * U1 / c̄

    # Compute force coefficients
    C_X, C_Y, C_Z = compute_force_coefficients(data, X, left_fuel_qty, right_fuel_qty, dyn_press, S, trim_state)
    force_coefficient_plot(data, event_numbers, time, C_X, C_Y, C_Z)

    # Compute C_z_alpha
    Cz_μf_unscaled, Cz_σf_unscaled, Cz_μf_scaled, Cz_σf_scaled, Z_alpha_gp, Z_alpha_lin, Cz_alpha_lin, cz_scale_x, cz_unscale_x, cz_scale_factor, cz_scaler = compute_Cz_alpha(data[!, "NZ_NORMAL_ACCEL"], X, left_fuel_qty, right_fuel_qty, dyn_press, S, trim_state)

    make_table(Cm_alpha_lin, Cm_δe_lin, Cm_q_lin, Cm_alpha_gp, Cm_δe_gp, Cm_q_gp, Z_alpha_gp, Z_alpha_lin, Cz_alpha_lin, trim_state, S, c̄, Iyy[1])

    # Empirical Data
    machs_emp = [0.71, 0.71, 1.08, 0.91, 0.9, 0.7, 0.5, 0.9, 0.7, 0.5, 0.9, 0.7, 0.35, 0.35, 0.6, 0.81, 0.89, 0.92, 0.9, 1.06, 1.09]
    dps_emp = [513.0, 203.0, 458.0, 498.0, 551.0, 334.0, 170.0, 551.0, 334.0, 170.0, 551.0, 334.0, 125.0, 125.0, 198.0, 141.0, 807.0, 182.0, 445.0, 242.0, 653.0]
    ω_sps_emp = [0.72, 0.38, 0.9, 0.86, 0.64, 0.46, 0.34, 0.66, 0.47, 0.34, 0.65, 0.49, 0.37, 0.37, 0.36, 0.38, 0.83, 0.42, 0.67, 0.56, 0.83]
    ζ_sps_emp = [0.32, 0.29, 0.22, 0.32, 0.36, 0.31, 0.31, 0.38, 0.34, 0.35, 0.33, 0.31, 0.30, 0.23, 0.27, 0.26, 0.35, 0.18, 0.30, 0.13, 0.17]

    dps = collect(LinRange(100, 900, 100))

    # Plot results
    plot_results(dps_emp, ω_sps_emp, ζ_sps_emp, machs_emp, dps, μf_unscaled, Cz_μf_unscaled,
                 μf_scaled, σf_scaled, Cz_μf_scaled, Cz_σf_scaled,
                 Cz_alpha_lin, Θ5_lin, Cm_alpha_lin, empty_weight, left_fuel_qty, right_fuel_qty,
                 S, c̄, Iyy, X_train, cm_scale_x, cz_scale_x, cm_unscale_x, cz_unscale_x, cm_scale_factor,
                 cz_scale_factor, scaler, cz_scaler)

    # 3D Surface plot
    surface_3D_plot(μf_unscaled, σf_unscaled)

    # CM Prediction plot
    CM_prediction_plot(data, event_numbers, time, X, Θ, μf_unscaled, σf_unscaled, C_m)
end

export
    run,
    load_and_preprocess_data,
    compute_inertia,
    compute_Q_dot,
    compute_moments,
    linear_model_fitting,
    GP_regression,
    morelli_mean,
    make_table,
    compute_Cz_alpha,
    compute_force_coefficients,
    plot_results,
    surface_3D_plot,
    CM_prediction_plot


end # module