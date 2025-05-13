"""
    compute_speed_of_sound(T::Float64) -> Float64

Compute the speed of sound (in ft/s) for a given temperature in Kelvin.

# Arguments
- `T`: Temperature in Kelvin.

# Returns
- Speed of sound in ft/s.
"""
function compute_speed_of_sound(T::Float64)
    return sqrt(γ * R * T)*mps_to_ft_s # Convert m/s to ft/s
end

"""
    load_and_preprocess_data(filename::String, event_numbers::Vector{Int}) -> (data, ...)

Load and preprocess flight test data from a CSV file, filtering by event numbers.

# Arguments
- `filename`: Path to the CSV file containing the data.
- `event_numbers`: Vector of event numbers to filter the data.

# Returns
- `data`: DataFrame containing the filtered data with additional computed columns.
- Extracted and computed variables such as `time`, `alpha_rad`, `mach`, `airspeed`, etc.
"""
function load_and_preprocess_data(filename::String, event_numbers::Vector{Int})
    # Load data
    data = CSV.read(filename, DataFrame)
    # Filter events
    event_name = "event_2_7_12_20_23"
    data = data[data[!, "EVENT"] .== 2 .|| data[!, "EVENT"] .== 7 .|| data[!, "EVENT"] .== 12 .|| data[!, "EVENT"] .== 20 .|| data[!, "EVENT"] .== 23, :]

    # Extract data
    time = data[!, "Delta_Irig"]             # Seconds
    alpha = data[!, "AOA"]                   # Degrees
    mach = data[!, "MACH_IC"]                # Mach number
    airspeed = data[!, "ADC_TRUE_AIRSPEED"]  # Knots
    roll = data[!, "EGI_ROLL_ANGLE"]         # Degrees
    pitch = data[!, "EGI_PITCH_ANGLE"]       # Degrees
    P = data[!, "EGI_ROLL_RATE_P"]           # Degrees per second
    Q = data[!, "EGI_PITCH_RATE_Q"]          # Degrees per second
    R = data[!, "EGI_YAW_RATE_R"]            # Degrees per second
    stab_pos = data[!, "STAB_POS"]           # Degrees
    press_alt_ic = data[!, "PRESS_ALT_IC"]   # Feet
    AMB_AIR_TEMP_C = data[!, "AMB_AIR_TEMP_C"] # Celsius
    left_fuel_qty = data[!, "EED_LEFT_FUEL_QTY"]   # Pounds
    right_fuel_qty = data[!, "EED_RIGHT_FUEL_QTY"] # Pounds

    # Convert to radians
    alpha_rad = deg2rad.(alpha)
    roll_rad = deg2rad.(roll)
    pitch_rad = deg2rad.(pitch)
    P_rad = deg2rad.(P)
    Q_rad = deg2rad.(Q)
    R_rad = deg2rad.(R)
    stab_pos_rad = deg2rad.(stab_pos)

    # Compute airspeed in ft/s
    V_ft_s = airspeed * knot_to_ft_s

    # Compute density altitude
    ISA_temp = [ISAdata(press_alt_ic[i]u"ft")[3].val - 273.15 for i in 1:length(press_alt_ic)]
    density_alt = press_alt_ic + (120 * (AMB_AIR_TEMP_C .- ISA_temp))

    # Compute air density (slug/ft³)
    rho = [ISAdata(density_alt[i]u"ft")[1].val for i in 1:length(density_alt)] .* kgm3_to_slugft3

    # Compute dynamic pressure
    dyn_press = 0.5 .* rho .* V_ft_s.^2 

    # Total weight
    fuel_weight = left_fuel_qty .+ right_fuel_qty
    total_weight = empty_weight .+ fuel_weight

    # Remove bias from P, Q, R
    P_rad .-= mean(P_rad[1:10])
    Q_rad .-= mean(Q_rad[1:10])
    R_rad .-= mean(R_rad[1:10])

    # Add computed variables to data DataFrame 
    data[!, :alpha_rad] = alpha_rad
    data[!, :P_rad] = P_rad
    data[!, :Q_rad] = Q_rad
    data[!, :R_rad] = R_rad
    data[!, :stab_pos_rad] = stab_pos_rad
    data[!, :rho] = rho
    data[!, :dyn_press] = dyn_press
    
    return (data, time, alpha_rad, mach, airspeed, roll_rad, pitch_rad, P_rad, Q_rad, R_rad, stab_pos_rad,
            press_alt_ic, AMB_AIR_TEMP_C, left_fuel_qty, right_fuel_qty, V_ft_s, rho,
            dyn_press, fuel_weight, total_weight)
end

"""
    compute_Q_dot(Q::Vector{Float64}, time::Vector{Float64}) -> Vector{Float64}

Compute the time derivative of the pitch rate Q (i.e., Q_dot).

# Arguments
- `Q`: Vector of pitch rates (radians per second).
- `time`: Vector of time stamps (seconds).

# Returns
- Vector of Q_dot values (radians per second squared).
"""
function compute_Q_dot(Q::Vector{Float64}, time::Vector{Float64})
    # Q_dot = [0.0; diff(Q) ./ diff(time)]
    Q_dot = zeros(length(Q))
    Q_dot[2:end-1] = (Q[3:end] - Q[1:end-2]) ./ (time[3:end] - time[1:end-2])
    Q_dot[1] = (Q[2] - Q[1]) / (time[2] - time[1])  # Forward difference for the first point
    Q_dot[end] = (Q[end] - Q[end-1]) / (time[end] - time[end-1])  # Backward difference for the last point

    return Q_dot
end

"""
    compute_ωsp_ζsp(Z_alpha::Float64, Cm_q::Float64, dp::Float64, S::Float64, 
                    c_bar::Float64, Iyy::Float64, U1::Float64, Cm_alpha::Float64)

Compute the short-period natural frequency `ω_sp` (in Hz) and damping ratio `ζ_sp` for an aircraft using longitudinal stability derivatives.

# Arguments
- `Z_alpha::Float64`: Stability derivative associated with pitching acceleration per change in angle of attack.
- `Cm_q::Float64`: Pitch moment coefficient due to rate of pitch.
- `dp::Float64`: Dynamic pressure.
- `S::Float64`: Reference wing area.
- `c_bar::Float64`: Mean aerodynamic chord of the wing.
- `Iyy::Float64`: Aircraft moment of inertia around the y-axis.
- `U1::Float64`: Steady flight velocity.
- `Cm_alpha::Float64`: Pitch moment coefficient due to angle of attack.

# Returns
- `ω_sp::Float64`: Short-period natural frequency in Hz.
- `ζ_sp::Float64`: Short-period damping ratio (dimensionless).
"""
function compute_ωsp_ζsp(Z_alpha::Float64, Cm_q::Float64, dp::Float64, S::Float64,
                         c_bar::Float64, Iyy::Float64, U1::Float64, Cm_alpha::Float64)
    
    # Calculate stability derivatives
    M_q = (Cm_q * dp * S * c_bar^2) / (2 * Iyy * U1)
    M_alpha = (dp * S * c_bar * Cm_alpha) / Iyy
    # Compute the natural frequency ω_sp (rad/s)
    omega_sp_rad = sqrt((Z_alpha * M_q) / U1 - M_alpha) 
    # Compute damping ratio ζ_sp
    ζ_sp = - (M_q + (M_q/3) + Z_alpha / U1) / (2 * omega_sp_rad)
    # Convert ω_sp to Hz
    ω_sp = omega_sp_rad / (2 * π)
    return ω_sp, ζ_sp
end

"""
    ω_sp_plot(dps::Vector{Float64}, mach::Float64,
              μf_unscaled::Function, Cz_μf_unscaled::Function,
              μf_scaled::Function, σf_scaled::Function, Cz_μf_scaled::Function, Cz_σf_scaled::Function,
              Cz_alpha_lin::Float64, Θ5_lin::Float64, Cm_alpha_lin::Float64,
              empty_weight::Float64, left_fuel_qty::Vector{Float64},
              right_fuel_qty::Vector{Float64}, S::Float64, c̄::Float64,
              Iyy::Vector{Float64}, X_train, cm_scale_x, cz_scale_x, cm_unscale_x,
              cz_unscale_x, cm_scale_factor, cz_scale_factor, cm_scaler, cz_scaler) -> (...)

Compute and return arrays of short-period natural frequencies and damping ratios for both GP and linear models over a range of dynamic pressures.

# Arguments
- `dps`: Vector of dynamic pressures.
- `mach`: Mach number.
- Functions `μf_unscaled`, `Cz_μf_unscaled`, etc.: Mean and variance functions for C_m and C_Z.
- `Cz_alpha_lin`, `Θ5_lin`, `Cm_alpha_lin`: Linear model coefficients.
- `empty_weight`: Empty weight of the aircraft (lbs).
- `left_fuel_qty`, `right_fuel_qty`: Fuel quantities (lbs).
- `S`: Wing reference area (ft²).
- `c̄`: Mean aerodynamic chord (ft).
- `Iyy`: Aircraft moment of inertia around the y-axis (slug·ft²).
- Scaling functions and scalers.

# Returns
- `ω_sps_gp`, `ζ_sps_gp`: Arrays of omega_{sp} and zeta_{sp} from GP model.
- `ω_sps_lin`, `ζ_sps_lin`: Arrays of omega_{sp} and zeta_{sp} from linear model.
- `U1s`: Array of flight speeds.
- `Z_alphas`: Array of Z_alpha derivatives.
- `M_qs`: Array of M_q derivatives.
- `M_alphas`: Array of M_alpha derivatives.
"""
function ω_sp_plot(dps::Vector{Float64}, mach::Float64,
    μf_unscaled::Function, Cz_μf_unscaled::Function,
    μf_scaled::Function, σf_scaled::Function, Cz_μf_scaled::Function, Cz_σf_scaled::Function,
    Cz_alpha_lin::Float64, Θ5_lin::Float64, Cm_alpha_lin::Float64,
    empty_weight::Float64, left_fuel_qty::Vector{Float64},
    right_fuel_qty::Vector{Float64}, S::Float64, c̄::Float64,
    Iyy::Vector{Float64}, X_train, cm_scale_x, cz_scale_x, cm_unscale_x, 
    cz_unscale_x, cm_scale_factor, cz_scale_factor, cm_scaler, cz_scaler)

    # Initialize arrays to store results
    ω_sps_gp = zeros(length(dps))
    ω_sps_lin = zeros(length(dps))
    ζ_sps_gp = zeros(length(dps))
    ζ_sps_lin = zeros(length(dps))
    U1s = zeros(length(dps))
    Z_alphas = zeros(length(dps))
    M_qs = zeros(length(dps))
    M_alphas = zeros(length(dps))


    # Total mass
    m = empty_weight .+ left_fuel_qty .+ right_fuel_qty

    altitudes = collect(1:1:100000)
    pressures = [ISAdata(alt*1u"ft")[2].val for alt in altitudes]

    for (ii, dp) in enumerate(dps)
        Press = (2*dp / (γ * mach^2)) * psf_to_pa # convert psf to Pa
        # find index of pressures that is closest to the current pressure
        idx = argmin(abs.(pressures .- Press))
        altitude = altitudes[idx]        
        rho = ISAdata(altitude *1u"ft")[1].val .* kgm3_to_slugft3 # convert kg/m³ to slug/ft³
        Temp_K = ISAdata(altitude *1u"ft")[3].val
        a = compute_speed_of_sound(Temp_K)

        # Compute true airspeed
        U1 = sqrt(2 * dp / rho)
        # U1 = mach * a

        # Compute trim conditions
        alpha = trim_aoa(dp)
        stab_pos = trim_stab_pos(dp)
        # State vector: mach, rho, dyn_press, P, Q, R, alpha, stab_pos
        X_query = [mach, rho, dp, 0.0, 0.0, 0.0, alpha, stab_pos]

        # Compute gradients using Zygote
        grad_Cm = Zygote.gradient(μf_unscaled, X_query)[1]
        grad_Cz = Zygote.gradient(Cz_μf_unscaled, X_query)[1]

        # Extract aerodynamic derivatives
        Cm_alpha_gp = grad_Cm[7]
        Cm_q_gp = grad_Cm[5] * 2 * U1 / c̄
        Cz_alpha_gp = grad_Cz[7]
        Z_alpha_gp = Cz_alpha_gp * dp * S / m[1]

        # Compute ω_sp and ζ_sp for the GP model
        ω_sp_gp, ζ_sp_gp = compute_ωsp_ζsp(Z_alpha_gp, Cm_q_gp, dp, S, c̄, Iyy[1], U1, Cm_alpha_gp)

    
        # Compute ω_sp and ζ_sp for the linear model
        Z_alpha_lin = Cz_alpha_lin * dp * S / m[1]
        M_q_lin = Θ5_lin * 2 * U1 / c̄
        ω_sp_lin, ζ_sp_lin = compute_ωsp_ζsp(Z_alpha_lin, M_q_lin, dp, S, c̄, Iyy[1], U1, Cm_alpha_lin)

        if 2 < altitude < 50000
            ω_sps_gp[ii] = ω_sp_gp
            ζ_sps_gp[ii] = ζ_sp_gp
            ω_sps_lin[ii] = ω_sp_lin
            ζ_sps_lin[ii] = ζ_sp_lin
            U1s[ii] = U1
            Z_alphas[ii] = Z_alpha_gp
            M_qs[ii] = (Cm_q_gp * dp * S * c̄^2) / (2 * Iyy[1] * U1)
            M_alphas[ii] = (dp * S * c̄ * Cm_alpha_gp) / Iyy[1]
        end
    end

    return ω_sps_gp, ζ_sps_gp, ω_sps_lin, ζ_sps_lin, U1s, Z_alphas, M_qs, M_alphas
end

"""
    make_table(Cm_alpha_lin::Float64, Cm_δe_lin::Float64, Cm_q_lin::Float64,
               Cm_alpha_gp::Float64, Cm_δe_gp::Float64, Cm_q_gp::Float64,
               Z_alpha_gp::Float64, Z_alpha_lin::Float64, Cz_alpha_lin::Float64,
               trim_state::Vector{Float64}, S::Float64, c̄::Float64, Iyy::Float64)

Create and display a table comparing the aerodynamic derivatives and short-period parameters from linear and GP models.

# Arguments
- `Cm_alpha_lin`, `Cm_δe_lin`, `Cm_q_lin`: Linear model coefficients.
- `Cm_alpha_gp`, `Cm_δe_gp`, `Cm_q_gp`: GP model coefficients.
- `Z_alpha_gp`, `Z_alpha_lin`, `Cz_alpha_lin`: Stability derivatives.
- `trim_state`: State vector at trim condition.
- `S`: Wing reference area (ft²).
- `c̄`: Mean aerodynamic chord (ft).
- `Iyy`: Aircraft moment of inertia around the y-axis (slug·ft²).

# Returns
- None. Displays a table using `pretty_table`.
"""
function make_table(Cm_alpha_lin, Cm_δe_lin, Cm_q_lin, Cm_alpha_gp, Cm_δe_gp, Cm_q_gp, Z_alpha_gp, Z_alpha_lin, Cz_alpha_lin, trim_state, S, c̄, Iyy)
    # State vector: mach, rho, dyn_press, P, Q, R, alpha, stab_pos
    dp = trim_state[3]
    U1 = sqrt(2 * dp / trim_state[2])
    ω_sp_gp, ζ_sp_gp = compute_ωsp_ζsp(Z_alpha_gp, Cm_q_gp, dp, S, c̄, Iyy, U1, Cm_alpha_gp)
    ω_sp_lin, ζ_sp_lin = compute_ωsp_ζsp(Z_alpha_lin, Cm_q_lin, dp, S, c̄, Iyy, U1, Cm_alpha_lin)

    # Data at Point 2 (0.7M)
    res_data = [
    ["Linear Model", Cm_alpha_lin, Cm_δe_lin, Cm_q_lin, ω_sp_lin, ζ_sp_lin],
    ["GP estimates", Cm_alpha_gp, Cm_δe_gp, Cm_q_gp, ω_sp_gp, ζ_sp_gp],
    ["Paper Estimates", -5.624e-1, -1.285e0, -1.272e1, 0.38, 0.29]
    ]

    # Convert the vector of vectors into a 3x4 Matrix
    matrix_data = Matrix{Any}(undef, 3, 6)

    # Fill the matrix with the values
    for i in 1:3
        matrix_data[i, :] = res_data[i]
    end
    pretty_table(matrix_data; header=["Method", "Cm_alpha", "Cm_δe", "Cm_q", "ω_sp", "ζ_sp"])
end

"""
    compute_rmse(dps_model, omega_model, zeta_model, dps_emp, omega_emp, zeta_emp; dp_tol=40)

Computes the root-mean-square error (RMSE) between empirical and model data for short period frequency and damping.

# Arguments
- `dps_model: Vector of dynamic pressure values from the model.
- `omega_model`: Vector of frequency values corresponding to `dps_model`.
- `zeta_model`: Vector of damping values corresponding to `dps_model`.
- `dps_emp`: Vector of dynamic pressure values from empirical data.
- `omega_emp`: Vector of frequency values corresponding to `dps_emp`.
- `zeta_emp`: Vector of damping ratio values corresponding to `dps_emp`.
- `dp_tol` (optional): Tolerance for grouping empirical data by dynamic pressure. Defaults to 40.

# Returns
A tuple `(rmse_data_omega, rmse_data_zeta, rmse_pred_omega, rmse_pred_zeta)` where:
- `rmse_data_omega`: Average RMSE of the empirical frequency values grouped by dynamic pressure.
- `rmse_data_zeta`: Average RMSE of the empirical damping values grouped by dynamic pressure.
- `rmse_pred_omega`: RMSE between empirical and model frequency values.
- `rmse_pred_zeta`: RMSE between empirical and model damping values.

# Details
1. The function first computes `rmse_data` by grouping empirical data based on unique dynamic pressure values.
   - For each dynamic pressure group, it calculates the RMSE of frequency and damping.
2. The function then computes `rmse_pred` by comparing the model predictions with the empirical data.
   - For each empirical data point, the nearest dynamic pressure value in the model data is found, 
     and the corresponding model predictions are used to calculate the RMSE.
"""
function compute_rmse(dps_model, omega_model, zeta_model, dps_emp, omega_emp, zeta_emp, dp_tol=40)
    # First, compute rmse_data by grouping empirical data by dynamic pressure
    unique_dps = unique(dps_emp)
    rmse_list_omega = []
    rmse_list_zeta = []

    for dp in unique_dps
        # Find indices where dynamic pressure matches
        indices = findall(x -> abs.(x .- dp) .< dp_tol, dps_emp)
        if length(indices) > 1
            # Multiple measurements at this dynamic pressure
            omega_vals = omega_emp[indices]
            zeta_vals = zeta_emp[indices]
            # Compute mean values
            omega_mean = mean(omega_vals)
            zeta_mean = mean(zeta_vals)
            # Compute RMSE for omega and zeta
            omega_rmse = sqrt(mean((omega_vals .- omega_mean).^2))
            zeta_rmse = sqrt(mean((zeta_vals .- zeta_mean).^2))
            # Collect RMSEs
            push!(rmse_list_omega, omega_rmse)
            push!(rmse_list_zeta, zeta_rmse)
        end
    end

    # Average RMSEs for the empirical data
    if length(rmse_list_omega) == 0
        rmse_data_omega = NaN
        rmse_data_zeta = NaN
    else
        rmse_data_omega = mean(rmse_list_omega)
        rmse_data_zeta = mean(rmse_list_zeta)
    end

    # Now compute rmse_pred by comparing model predictions to empirical data
    omega_model_at_emp = zeros(length(dps_emp))
    zeta_model_at_emp = zeros(length(dps_emp))

    for (i, dp_emp) in enumerate(dps_emp)
        # Find the closest dynamic pressure in the model data
        idx = findmin(abs.(dps_model .- dp_emp))[2]
        omega_model_at_emp[i] = omega_model[idx]
        zeta_model_at_emp[i] = zeta_model[idx]
    end

    # Compute RMSE between empirical data and model predictions
    rmse_pred_omega = sqrt(mean((omega_emp .- omega_model_at_emp).^2))
    rmse_pred_zeta = sqrt(mean((zeta_emp .- zeta_model_at_emp).^2))

    return rmse_data_omega, rmse_data_zeta, rmse_pred_omega, rmse_pred_zeta
end

"""
    plot_results(dps_emp::Vector{Float64}, ω_sps_emp::Vector{Float64}, ζ_sps_emp::Vector{Float64},
                 machs_emp::Vector{Float64}, dps::Vector{Float64}, μf_unscaled::Function,
                 Cz_μf_unscaled::Function, μf_scaled::Function, σf_scaled::Function,
                 Cz_μf_scaled::Function, Cz_σf_scaled::Function, Cz_alpha_lin::Float64,
                 Θ5_lin::Float64, Cm_alpha_lin::Float64, empty_weight::Float64,
                 left_fuel_qty::Vector{Float64}, right_fuel_qty::Vector{Float64},
                 S::Float64, c̄::Float64, Iyy::Vector{Float64}, X_train,
                 cm_scale_x, cz_scale_x, cm_unscale_x, cz_unscale_x, cm_scale_factor,
                 cz_scale_factor, cm_scaler, cz_scaler)

Plot the short-period natural frequency omega_{sp} and damping ratio zeta_{sp} versus dynamic pressure for GP and linear models, and compare with empirical data.

# Arguments
- `dps_emp`, `ω_sps_emp`, `ζ_sps_emp`, `machs_emp`: Empirical data points.
- `dps`: Vector of dynamic pressures.
- Functions `μf_unscaled`, `Cz_μf_unscaled`, etc.: Mean and variance functions for C_m and C_Z.
- `Cz_alpha_lin`, `Θ5_lin`, `Cm_alpha_lin`: Linear model coefficients.
- `empty_weight`, `left_fuel_qty`, `right_fuel_qty`: Aircraft weights.
- `S`, `c̄`, `Iyy`: Aircraft parameters.
- Other arguments related to data scaling.

# Returns
- None. Generates and saves plots.
"""
function plot_results(dps_emp, ω_sps_emp, ζ_sps_emp, machs_emp, dps, μf_unscaled, Cz_μf_unscaled,
                      μf_scaled, σf_scaled, Cz_μf_scaled, Cz_σf_scaled,
                      Cz_alpha_lin, Θ5_lin, Cm_alpha_lin, empty_weight, left_fuel_qty, right_fuel_qty,
                      S, c̄, Iyy, X_train, cm_scale_x, cz_scale_x, cm_unscale_x, cz_unscale_x, cm_scale_factor,
                      cz_scale_factor, cm_scaler, cz_scaler)
    # Plot ω_sp vs Dynamic Pressure
    gr()
    plt1 = Plots.plot()
    all_U1s = []
    all_Z_alphas = []
    all_M_qs = []
    all_M_alphas = []
    all_ω_sps_gp = []
    all_ω_sps_lin = []
    all_ζ_sps_gp = []
    all_ζ_sps_lin = []

    data_path = joinpath(@__DIR__, "data")
    figure_path = joinpath(@__DIR__, "figures")

    all_rmse_data_omega = []
    all_rmse_data_zeta = []
    all_rmse_pred_omega = []
    all_rmse_pred_zeta = []

    for mach in [0.5, 0.7, 0.9, 1.08]
        ω_sps_gp, ζ_sps_gp, ω_sps_lin, ζ_sps_lin, U1s, Z_alphas, M_qs, M_alphas = ω_sp_plot(dps, mach, μf_unscaled, Cz_μf_unscaled,
                                                              μf_scaled, σf_scaled, Cz_μf_scaled, Cz_σf_scaled,
                                                              Cz_alpha_lin, Θ5_lin, Cm_alpha_lin,
                                                              empty_weight, left_fuel_qty, right_fuel_qty, S, c̄, Iyy, X_train, cm_scale_x, cz_scale_x, cm_unscale_x, 
                                                              cz_unscale_x, cm_scale_factor, cz_scale_factor, cm_scaler, cz_scaler)

        valid = ω_sps_gp .!= 0.0
        # Plot only valid data
        Plots.plot!(plt1, dps[valid], ω_sps_gp[valid],
            xlabel=L"\mathrm{Dynamic\ Pressure}\ (\mathrm{lb}/\mathrm{ft}^2)",
            ylabel=L"\omega_{sp}", title=L"\omega_{sp}\ \mathrm{vs.\ Dynamic\ Pressure}",
            label="GP $(mach)M")
        Plots.plot!(plt1, dps[valid], ω_sps_lin[valid], label="Linear Model $(mach)M", linestyle=:dash)

        ω_sps_emp_valid = ω_sps_emp[abs.(machs_emp .- mach) .< 0.1]
        ζ_sps_emp_valid = ζ_sps_emp[abs.(machs_emp .- mach) .< 0.1]
        dps_emp_valid = dps_emp[abs.(machs_emp .- mach) .< 0.1]

        rmse_data_omega, rmse_data_zeta, rmse_pred_omega, rmse_pred_zeta = compute_rmse(dps[valid], ω_sps_gp[valid], ζ_sps_gp[valid], dps_emp_valid, ω_sps_emp_valid, ζ_sps_emp_valid)
        push!(all_rmse_data_omega, rmse_data_omega)
        push!(all_rmse_data_zeta, rmse_data_zeta)
        push!(all_rmse_pred_omega, rmse_pred_omega)
        push!(all_rmse_pred_zeta, rmse_pred_zeta)

        df = DataFrame(dps=dps, ω_sps_gp=ω_sps_gp, ω_sps_lin=ω_sps_lin, ζ_sps_gp=ζ_sps_gp, ζ_sps_lin=ζ_sps_lin) 
        CSV.write(data_path * "/sp_results_$(mach)M.csv", df)

        push!(all_U1s, U1s)
        push!(all_Z_alphas, Z_alphas)
        push!(all_M_qs, M_qs)
        push!(all_M_alphas, M_alphas)
        push!(all_ω_sps_gp, ω_sps_gp)
        push!(all_ω_sps_lin, ω_sps_lin)
        push!(all_ζ_sps_gp, ζ_sps_gp)
        push!(all_ζ_sps_lin, ζ_sps_lin)
    end
    scatter!(plt1, dps_emp, ω_sps_emp, marker_z=machs_emp, label="Empirical Data", ylims=(0, 1.0))

    # Make a table of results where the rows are mach numbers and the columns are RMSE values
    rmse_matrix = Matrix{Any}(undef, 4, 5)
    machs = [0.5, 0.7, 0.9, 1.08]
    for i in 1:4
        rmse_matrix[i, 1] = machs[i]
        rmse_matrix[i, 2] = all_rmse_pred_omega[i] 
        rmse_matrix[i, 3] = all_rmse_data_omega[i]
        rmse_matrix[i, 4] = all_rmse_pred_zeta[i]
        rmse_matrix[i, 5] = all_rmse_data_zeta[i]
    end
    pretty_table(round.(rmse_matrix, digits=3); header=["Mach", "RMSE Pred Omega", "RMSE Data Omega", "RMSE Pred Zeta", "RMSE Data Zeta"])


    # Plot ζ_sp vs Dynamic Pressure
    plt2 = Plots.plot()
    for (ii, mach) in enumerate([0.5, 0.7, 0.9, 1.08])        
        ω_sps_gp = all_ω_sps_gp[ii]
        ω_sps_lin = all_ω_sps_lin[ii]
        ζ_sps_gp = all_ζ_sps_gp[ii]
        ζ_sps_lin = all_ζ_sps_lin[ii]

        valid = ω_sps_gp .!= 0.0

        Plots.plot!(plt2, dps[valid], ζ_sps_gp[valid], xlabel=L"\mathrm{Dynamic\ Pressure}\ (\mathrm{lb}/\mathrm{ft}^2)",
            ylabel=L"\zeta_{sp}", title=L"\zeta_{sp}\ \mathrm{vs.\ Dynamic\ Pressure}",
            label="GP $(mach)M")
        Plots.plot!(plt2, dps[valid], ζ_sps_lin[valid], label="Linear Model $(mach)M", linestyle=:dash)
    end
    Plots.scatter!(plt2, dps_emp, ζ_sps_emp, marker_z=machs_emp, label="Empirical Data", ylims=(0, 1.0))

    Plots.display(Plots.plot(plt1, plt2, layout=(2, 1), size=(800, 800)))
    Plots.savefig(figure_path * "/sp_results.pdf")
end

"""
    surface_3D_plot(μf_unscaled::Function, σf_unscaled::Function)

Create a 3D surface plot of the pitch moment coefficient C_m versus angle of attack and pitch rate, along with the associated uncertainties.

# Arguments
- `μf_unscaled`: Function to compute the mean C_m from the GP model.
- `σf_unscaled`: Function to compute the standard deviation of C_m.

# Returns
- None. Displays and saves the 3D plot.
"""
function surface_3D_plot(μf_unscaled, σf_unscaled)
    plotlyjs()

    # Define the meshgrid function
    function meshgrid(x, y)
        nx, ny = length(x), length(y)
        x = reshape(x, nx, 1)
        y = reshape(y, 1, ny)
        return repeat(x, 1, ny), repeat(y, nx, 1)
    end

    # Generate x1 and x2
    x1 = deg2rad.(collect(LinRange(-30, 30, 50)))
    x2 = deg2rad.(collect(LinRange(-30, 30, 50)))

    # Create the meshgrid
    alpha_surface_query, Q_surface_query = meshgrid(x1, x2)

    # Flatten the arrays for computation
    alpha_flat = vec(alpha_surface_query)
    Q_flat = vec(Q_surface_query)

    # Create other variables
    n_points = length(alpha_flat)
    mach_flat = fill(0.5, n_points)
    rho_flat = fill(0.001143, n_points)
    dyn_press_flat = fill(512.5, n_points)
    P_flat = zeros(n_points)
    R_flat = zeros(n_points)
    stab_pos_flat = fill(deg2rad(-2), n_points)

    # State vector: mach, rho, dyn_press, P, Q, R, alpha, stab_pos
    X_surface_query = hcat(
        mach_flat,
        rho_flat,
        dyn_press_flat,
        P_flat,
        Q_flat,
        R_flat,
        alpha_flat,
        stab_pos_flat
    )

    # Compute CM_plot and σ_CM_plot
    CM_plot = [μf_unscaled(X_surface_query[i, :]) for i in 1:n_points]
    σ_CM_plot = [σf_unscaled(X_surface_query[i, :]) for i in 1:n_points]

    # Reshape CM_plot into a 50x50 matrix
    CM_plot_matrix = reshape(CM_plot, size(alpha_surface_query))
    σ_CM_plot_matrix = reshape(σ_CM_plot, size(alpha_surface_query))


    # Create the surface plot using PlotlyJS
    p = PlotlyJS.plot(
        PlotlyJS.surface(
            x=rad2deg.(alpha_surface_query),
            y=rad2deg.(Q_surface_query),
            z=CM_plot_matrix,
            surfacecolor=σ_CM_plot_matrix,
            colorscale="Viridis",
            colorbar=attr(title="sigma_{Cm}")
        ),
        Layout(
            scene=attr(
                xaxis=attr(title="Alpha (deg)"),
                yaxis=attr(title="Q (deg/s)"),
                zaxis=attr(title="C_m")
            )
        )
    )

    df = DataFrame(alpha=rad2deg.(vec(alpha_surface_query)), Q=rad2deg.(vec(Q_surface_query)), CM=CM_plot, σ_CM=σ_CM_plot)
    data_path = joinpath(@__DIR__, "data")
    CSV.write(data_path * "/surface_plot.csv", df)

    display(p)
    # PlotlyJS.savefig(p, "surface_plot.pdf")
end

"""
    CM_prediction_plot(data::DataFrame, event_numbers::Vector{Int}, time::Vector{Float64},
                       X::Matrix{Float64}, Θ::Vector{Float64}, μf_unscaled::Function,
                       σf_unscaled::Function, C_m::Vector{Float64})

Plot the predicted pitch moment coefficient C_m over time for specified events, comparing the GP model, linear model, and empirical data.

# Arguments
- `data`: DataFrame containing the data.
- `event_numbers`: Vector of event numbers to plot.
- `time`: Vector of time stamps.
- `X`: Design matrix (predictor variables).
- `Θ`: Coefficients from linear model fitting.
- `μf_unscaled`, `σf_unscaled`: Functions from GP model to compute mean and standard deviation of C_m.
- `C_m`: Empirical C_m values.

# Returns
- None. Generates and saves plots for each event.
"""
function CM_prediction_plot(data, event_numbers, time, X, Θ, μf_unscaled, σf_unscaled, C_m)
    # find index corresponding to the first time event starts
    event_start_idxs = [findfirst(data[!, "EVENT"] .== event_number) for event_number in event_numbers]
    event_end_idxs = [findlast(data[!, "EVENT"] .== event_number) for event_number in event_numbers]
    gr()

    C_m_lin_pred = X * Θ
    C_m_gp_pred = [μf_unscaled(X[i, :]) for i in 1:size(X, 1)]
    σ_C_m_gp_pred = [σf_unscaled(X[i, :]) for i in 1:size(X, 1)]

    data_path = joinpath(@__DIR__, "data")
    figure_path = joinpath(@__DIR__, "figures")
    
    for (ii, event_number) in enumerate(event_numbers)
        Plots.plot(time[event_start_idxs[ii]:event_end_idxs[ii]], C_m_gp_pred[event_start_idxs[ii]:event_end_idxs[ii]], ribbon=1.96 .* σ_C_m_gp_pred[event_start_idxs[ii]:event_end_idxs[ii]], label="GP Prediction")
        Plots.plot!(time[event_start_idxs[ii]:event_end_idxs[ii]], C_m_lin_pred[event_start_idxs[ii]:event_end_idxs[ii]], label="Linear Model Prediction")
        Plots.plot!(time[event_start_idxs[ii]:event_end_idxs[ii]], C_m[event_start_idxs[ii]:event_end_idxs[ii]], label="Empirical Data", color=:black)
        Plots.plot!(xlabel="Time (s)", ylabel=L"C_m", title="C_m Prediction for Event $(event_number)")
        Plots.savefig(figure_path * "/C_m_prediction_$(event_number).pdf")

        df = DataFrame(time=time[event_start_idxs[ii]:event_end_idxs[ii]], C_m=C_m[event_start_idxs[ii]:event_end_idxs[ii]], C_m_lin_pred=C_m_lin_pred[event_start_idxs[ii]:event_end_idxs[ii]], C_m_gp_pred=C_m_gp_pred[event_start_idxs[ii]:event_end_idxs[ii]], CI=1.96 .* σ_C_m_gp_pred[event_start_idxs[ii]:event_end_idxs[ii]])
        CSV.write(data_path * "/C_m_prediction_$(event_number).csv", df)
    end
end

"""
    force_coefficient_plot(data::DataFrame, event_numbers::Vector{Int}, time::Vector{Float64},
                          C_X::Vector{Float64}, C_Y::Vector{Float64}, C_Z::Vector{Float64})

Plot the force coefficients (C_X, C_Y, C_Z) over time for each event.

# Arguments
- `data`: DataFrame containing the data.
- `event_numbers`: Vector of event numbers to plot.
- `time`: Vector of time stamps.
- `C_X`, `C_Y`, `C_Z`: Force coefficients.
"""
function force_coefficient_plot(data::DataFrame, event_numbers::Vector{Int}, time::Vector{Float64},
                              C_X::Vector{Float64}, C_Y::Vector{Float64}, C_Z::Vector{Float64})
    # find index corresponding to the first time event starts
    event_start_idxs = [findfirst(data[!, "EVENT"] .== event_number) for event_number in event_numbers]
    event_end_idxs = [findlast(data[!, "EVENT"] .== event_number) for event_number in event_numbers]
    gr()

    data_path = joinpath(@__DIR__, "data")
    figure_path = joinpath(@__DIR__, "figures")
    
    for (ii, event_number) in enumerate(event_numbers)
        # Create a figure with three subplots
        p = Plots.plot(layout=(3,1), size=(800, 1200))

        # Plot each force coefficient
        Plots.plot!(p[1], time[event_start_idxs[ii]:event_end_idxs[ii]], 
              C_X[event_start_idxs[ii]:event_end_idxs[ii]], 
              label="C_X", 
              title="Axial Force Coefficient", 
              xlabel="Time (s)", 
              ylabel=L"C_X")
        
        Plots.plot!(p[2], time[event_start_idxs[ii]:event_end_idxs[ii]], 
              C_Y[event_start_idxs[ii]:event_end_idxs[ii]], 
              label="C_Y", 
              title="Side Force Coefficient", 
              xlabel="Time (s)", 
              ylabel=L"C_Y")
        
        Plots.plot!(p[3], time[event_start_idxs[ii]:event_end_idxs[ii]], 
              C_Z[event_start_idxs[ii]:event_end_idxs[ii]], 
              label="C_Z", 
              title="Normal Force Coefficient", 
              xlabel="Time (s)", 
              ylabel=L"C_Z")

        # Save the plot
        Plots.savefig(p, figure_path * "/force_coefficients_$(event_number).pdf")

        # Save the data
        df = DataFrame(
            time=time[event_start_idxs[ii]:event_end_idxs[ii]],
            C_X=C_X[event_start_idxs[ii]:event_end_idxs[ii]],
            C_Y=C_Y[event_start_idxs[ii]:event_end_idxs[ii]],
            C_Z=C_Z[event_start_idxs[ii]:event_end_idxs[ii]]
        )
        CSV.write(data_path * "/force_coefficients_$(event_number).csv", df)
    end
end