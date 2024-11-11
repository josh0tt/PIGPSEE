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
    load_and_preprocess_data_pixhawk(pixhawk::String, logical_files::Vector{Int}) -> (data, ...)

Load and preprocess flight test data from a CSV file, filtering by event numbers.

# Arguments
- `pixhawk`: Path to the CSV file containing the pixhawk data.
- 'logical_files': Vector of paths to csv files containing supplemental GPS-timed data
- `event_numbers`: Vector of event numbers to filter the data.

# Returns
- `data`: DataFrame containing the filtered data with additional computed columns.
- Extracted and computed variables such as `time`, `alpha_rad`, `mach`, `airspeed`, etc.
"""
function load_and_preprocess_data_pixhawk(
        pixhawk::String, 
        tags_file::String,
        constants_file::String,
        logical_files::Vector{String},
        event_numbers::Vector{Int}
    )
    # Open and read the single pixhawk file
    disorganized = CSV.read(joinpath(@__DIR__, "data", "pixhawk", pixhawk), DataFrame)
    # Open and read each logical csv
    logical = CSV.read.(joinpath.(@__DIR__, "data", "pixhawk", logical_files), DataFrame)
    # Open and read the tags csv
    tags = CSV.read(joinpath.(@__DIR__, "data", "pixhawk", tags_file), DataFrame)
    # Open and read constants file and process into dictionary
    constants_frame = CSV.read(joinpath.(@__DIR__, "data", "pixhawk", constants_file), DataFrame)
    constants = Dict(name => constants_frame[!, name][1] for name = names(constants_frame))


    # Parameters of interest in pixhawk data file
    # Format (RowName, [(Col1Index, Param1Alias), (Col2Index, Param2Alias)...])
    dis_interest = Dict{String, Vector{Tuple{Int64, String}}}([
        ("IMU", [(1, "P"), (2, "Q"), (3, "R"), (6, "z_accel")]),
        ("ATT", [(2, "roll"), (4, "pitch")]),
        ("BARO", [(1, "press_alt")]),
        ("GPS", [(9, "gs"), (10, "track")])
    ])

    # Parameters of interest in the logical csv input files, regardless of which file they're in
    # Format (CSVColumnName, ParameterAlias)
    log_interest = Dict{String, String}([
        ("stab_pos", "stab_pos"),
        ("aoa", "aoa"),
        ("left_fuel_qty", "left_fuel_qty"),
        ("right_fuel_qty", "right_fuel_qty"),
        ("temp", "temp")
    ])

    # Define computed properties from extracted data
    function computed_properties!(data)
        # Atmospheric properties
        data["isa_temp"] = [ISAdata(alt)[3] - 273.15 for alt = data["press_alt"]]
        data["density_alt"] = data["press_alt"] + (120 * (data["temp"] .- data["isa_temp"]))
        data["rho"] = [ISAdata(data["density_alt"][i])[1] for i in 1:length(data["density_alt"])] .* kgm3_to_slugft3

        # Convert extracted data to radians
        data["P_rad"] = deg2rad.(data["P"])
        data["Q_rad"] = deg2rad.(data["Q"])
        data["R_rad"] = deg2rad.(data["R"])
        data["roll_rad"] = deg2rad.(data["roll"])
        data["pitch_rad"] = deg2rad.(data["pitch"])
        data["alpha_rad"] = deg2rad.(data["aoa"])
        data["stab_pos_rad"] = deg2rad.(data["stab_pos"])

        # Remove bias from P, Q, and R
        data["P_rad"] .-= mean(data["P_rad"][1:10])
        data["Q_rad"] .-= mean(data["Q_rad"][1:10])
        data["R_rad"] .-= mean(data["R_rad"][1:10])

        # Calculate airspeed from gs and wind
        wind_run = [
            [data["gs"][i] * cosd(data["track"][i]), data["gs"][i] * sind(data["track"][i])]
            for (i, event) = enumerate(data["event"]) if event == 6155797356
        ]
        windless_vector = [
            constants["wr_airspeed"] * cosd(constants["wr_heading"]),
            constants["wr_airspeed"] * sind(constants["wr_heading"])
        ]
        wind_vectors = [[track_vector[1] - windless_vector[1], track_vector[2] - windless_vector[2]] for track_vector = wind_run]
        wind_vector = [mean([w[1] for w = wind_vectors]), mean([w[2] for w = wind_vectors])]
        airspeed_vectors = [
            [gs * cosd(track) - wind_vector[1], gs * sind(track) - wind_vector[2]]
            for (gs, track) = zip(data["gs"], data["track"])
        ]
        #println(airspeed_vectors)
        data["airspeed"] = [sqrt(v[1]^2 + v[2]^2) for v = airspeed_vectors]

        # Compute dynamic pressure from density and airspeed in ft/s
        data["V_ft_s"] = data["airspeed"] .* knot_to_ft_s
        data["dyn_press"] = 0.5 .* data["rho"] .* (data["V_ft_s"]).^2

        # Calculate mach from airspeed and speed of sound
        data["mach"] = data["airspeed"] ./ compute_speed_of_sound.(data["temp"] .+ 273.15)

        # Calculate weights from fuel burns and empty weight
        data["fuel_weight"] = data["right_fuel_qty"] .+ data["left_fuel_qty"]
        data["total_weight"] = data["fuel_weight"] .+ constants["unfueled_weight"]
    end

    # System to GPS time offset
    offset = missing;

    # Reorganize data points for each interest parameter into vectors of (time, data) ordered pairs
    aligned = Dict{String, Vector{Tuple{Float64, Float64}}}()
    keys = Vector{String}()
    for row = eachrow(disorganized)

        # Pull the time and row name for the current row
        id = row[1]
        time = row[2]

        # Find first GPS data point and calculate offset
        if (id == "GPS") && (offset === missing)
            gps_ms = parse(Float64, row[4])
            offset = gps_ms * 1000 - time
        end

        # Check whether any parameters of interest are associated with this row
        # If so, extract the data point for that parameter and add to the vector
        for item = get(dis_interest, id, Vector{Tuple{Int64, String}}([]))
            value::Float64 = parse(Float64, row[item[1] + 2])
            name = item[2]

            # Check for key and initialize vector if not already present, otherwise add to vector
            if get(aligned, name, missing) !== missing
                push!(aligned[name], (time, value))
            else
                aligned[name] = [(time, value)]
                push!(keys, name)
            end
        end
    end

    # Iterate through logical csv input file data frames
    for log = logical
        time = log[!, "time"]

        # Loop through data frame's columns
        for name = names(log)

            # Check whether current column is an interest parameter, if so add to the data vector
            value = get(log_interest, name, missing)
            if value !== missing

                # Extract column's data and pair it with corresponding timestamps
                col = log[!, name]
                aligned[value] = [(Float64(time[i]), Float64(col[i])) for i = 1:length(col)]
                push!(keys, value)
            end
        end
    end

    # Convert all timestamps to weekly GPS time in seconds
    for key = keys
        col = aligned[key]
        aligned[key] = [((col[i][1] + offset) / 10^6, col[i][2]) for i = 1:length(col)]
    end

    # Find the minimum timestep of all data columns
    step = minimum([minimum([col[i + 1][1] - col[i][1] for i = 1:length(col) - 1]) for (_, col) = aligned])
    # Find first timestamp of all data
    start = minimum([col[1][1] for (_, col) = aligned])
    # Find last
    stop = maximum([col[end][1] for (_, col) = aligned])

    # Generate time series for all time values between start and stop with step interval
    time = collect(start:step:stop)
    # Final interpolated data dictionary
    data = Dict{String, Vector{Float64}}()
    data["time"] = time

    # Loop through all parameter keys, building interpolated data vector aligned with selected time values
    for key = keys
        col = Vector{Float64}()
        values = aligned[key]

        # Linearly interpolate between nearest two data points for each selected time value
        i = 2
        before = values[1]
        after = values[2]
        for t = start:step:stop
            while t > after[1] && i != length(values)
                i += 1
                before = after
                after = values[i]
            end

            if t < before[1]
                push!(col, before[2])
            elseif t > after[1]
                push!(col, after[2])
            else
                # Linear interpolation
                interp = ((t - before[1]) / (after[1] - before[1])) * (after[2] - before[2]) + before[2]
                push!(col, interp)
            end
        end

        # Add to final data dictionary
        data[key] = col
    end

    # Associate tags with each timestep
    data["event"] = [0 for i = 1:length(data["time"])]
    for event = eachrow(tags)
        id = event[1]
        first = event[2]
        last = event[3]
        start_idx = floor(Int, (first - start) / step) + 1
        end_idx = floor(Int, (last - start) / step) + 1
        data["event"][start_idx:end_idx] .= id
    end

    # Calculate computed properties based on time-aligned data
    computed_properties!(data)

    # Print for debugging
    #println(data)

    # Return in same format as primary loading function
    return (data, time, data["alpha_rad"], data["mach"], data["airspeed"], data["roll_rad"], data["pitch_rad"], 
            data["P_rad"], data["Q_rad"], data["R_rad"], data["stab_pos_rad"],
            data["press_alt"], data["temp"], data["left_fuel_qty"], data["right_fuel_qty"], data["V_ft_s"], data["rho"],
            data["dyn_press"], data["fuel_weight"], data["total_weight"])
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