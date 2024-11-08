"""
    T38_moment(Ixx::Float64, Iyy::Float64, Izz::Float64, Ixz::Float64,
               Q_dot::Float64, P::Float64, R::Float64) -> Float64

Compute the T-38 aircraft pitch moment.

# Arguments
- `Ixx`, `Iyy`, `Izz`, `Ixz`: Moments of inertia.
- `Q_dot`: Pitch rate derivative (radians per second squared).
- `P`, `R`: Roll and yaw rates (radians per second).

# Returns
- Pitch moment.
"""
function T38_moment(Ixx::Float64, Iyy::Float64, Izz::Float64, Ixz::Float64,
                    Q_dot::Float64, P::Float64, R::Float64)
    return Iyy * Q_dot + (Ixx - Izz) * P * R + Ixz * (P^2 - R^2)
end

"""
    T38_mass_properties(WFL::Float64, WFR::Float64) -> (Ixx, Iyy, Izz, Ixz, CG, CG_in, mass, DeltaXcg_in, DeltaZcg_in)

Compute the mass properties of the T-38 aircraft, including moments of inertia and center of gravity, based on fuel weights.

# Arguments
- `WFL`: Left fuel weight (lbs).
- `WFR`: Right fuel weight (lbs).

# Returns
- `Ixx`, `Iyy`, `Izz`, `Ixz`: Moments of inertia (slug·ft²).
- `CG`: Center of gravity (% MAC).
- `CG_in`: Center of gravity (inches).
- `mass`: Mass (slugs).
- `DeltaXcg_in`: X-axis CG deviation from reference (inches).
- `DeltaZcg_in`: Z-axis CG deviation from reference (inches).
"""
function T38_mass_properties(WFL::Float64, WFR::Float64)
    # Source: Shepherd, Michael J., Timothy R. Jorris, and William R. Gray. 
    # "Limited aerodynamic System Identification of the T-38A using SIDPAC software." 2010 IEEE Aerospace Conference. IEEE, 2010.

    # Operating Weight and Moment
    OPW = 8750.0  # Operating Weight
    OPM = 3039.6  # Operating Moment
    Mom_Sim = 1000.0  # Moment simplifier (constant to keep moments simple to use)
    MAC = 92.76
    LEMAC = 331.20

    # Computer Gross Weight
    GW = OPW + WFL + WFR
    mass = GW / 32.174

    # X-moments
    wfllx = 1.40309e-5 * WFL^2 + 0.2856289 * WFL + 4.008761
    wfrlx = -7.705063e-6 * WFR^2 + 0.3971317 * WFR + 5.0

    # Z-moments
    wfllz = 0.01126431823 * (729.8419938 - WFR) * (26.56136596 + WFL)
    wfrlz = 2.917364207e-9 * (1357.678895 - WFR) * (0.005072708037 + WFR) *
            (3.133863746e6 - 2003.023886 * WFR + WFR^2)

    # Ixx
    ixxfl = 2.43186279e-8 * (55.74415428 + WFL) * (450464.1502 - 1336.852141 * WFL + WFL^2)
    ixxfr = ifelse(WFR <= 1000,
                   1.429572953e-8 * (-1479.23189 + WFR) * (-1108.152712 + WFR) * (-2.971601441 + WFR),
                   5.738417234e-14 * (2722.258505 - WFR) * (-1350.000002 + WFR) *
                   (-1349.999998 + WFR) * (1.307231912e6 - 2246.212161 * WFR + WFR^2))

    # Izz
    izzfl = 1.319074486e-10 * (0.005072772213 + WFL) * (337.372297 + WFL) *
            (7.20647684e6 - 5137.151321 * WFL + WFL^2)
    izzfr = 1.090972066e-10 * (2827.719311 - WFR) * (0.005072711818 + WFR) *
            (3.315606698e6 - 2680.168835 * WFR + WFR^2)

    # Ixz
    ixzfl = 4.43948009e-8 * (-4021.220951 + WFL) * (-732.6733307 + WFL) * (-10.50598976 + WFL)
    ixzfr = 3.149396586e-11 * (-1348.969404 + WFR) * (0.005072705039 + WFR) *
            (-3.644552245e6 + 2929.66091 * WFR - WFR^2)

    # Inertia calculations
    Ixx = 1553.6 + ixxfl + ixxfr
    Izz = 29266 + izzfl + izzfr
    Iyy = 28333.72 + (ixxfl + izzfl) + (ixxfr + izzfr)
    Ixz = 47.25 + ixzfl + ixzfr

    # CG calculations (reference cg: FS354.39, WL100)
    CG = (((OPM + wfllx + wfrlx) * Mom_Sim / GW) - LEMAC) * 100 / MAC
    CG_in = ((OPM + wfllx + wfrlx) * Mom_Sim / GW) - LEMAC
    DeltaXcg_in = (((OPM + wfllx + wfrlx) * Mom_Sim / GW) - 354.39)
    DeltaZcg_in = ((-15852 + wfllz + wfrlz) / GW)

    return Ixx, Iyy, Izz, Ixz, CG, CG_in, mass, DeltaXcg_in, DeltaZcg_in
end

"""
    compute_inertia(left_fuel_qty::Vector{Float64}, right_fuel_qty::Vector{Float64}) -> (Ixx, Iyy, Izz, Ixz)

Compute the aircraft moments of inertia for each time step based on fuel quantities.

# Arguments
- `left_fuel_qty`: Vector of left fuel tank quantities (lbs).
- `right_fuel_qty`: Vector of right fuel tank quantities (lbs).

# Returns
- `Ixx`, `Iyy`, `Izz`, `Ixz`: Vectors of moments of inertia (slug·ft²) for each time step.
"""
function compute_inertia(left_fuel_qty::Vector{Float64}, right_fuel_qty::Vector{Float64})
    n = length(left_fuel_qty)
    Ixx = zeros(n)
    Iyy = zeros(n)
    Izz = zeros(n)
    Ixz = zeros(n)
    for i in 1:n
        Ixx[i], Iyy[i], Izz[i], Ixz[i], _, _, _, _, _ = T38_mass_properties(left_fuel_qty[i], right_fuel_qty[i])
    end
    return Ixx, Iyy, Izz, Ixz
end


"""
    compute_moments(Ixx::Vector{Float64}, Iyy::Vector{Float64}, Izz::Vector{Float64},
                    Ixz::Vector{Float64}, Q_dot::Vector{Float64}, P::Vector{Float64},
                    R::Vector{Float64}) -> Vector{Float64}

Compute the pitch moments for each time step based on moments of inertia and angular rates.

# Arguments
- `Ixx`, `Iyy`, `Izz`, `Ixz`: Vectors of moments of inertia (slug·ft²).
- `Q_dot`: Vector of pitch accelerations (radians per second squared).
- `P`, `R`: Vectors of roll and yaw rates (radians per second).

# Returns
- Vector of pitch moments.
"""
function compute_moments(Ixx::Vector{Float64}, Iyy::Vector{Float64}, Izz::Vector{Float64},
                         Ixz::Vector{Float64}, Q_dot::Vector{Float64}, P::Vector{Float64},
                         R::Vector{Float64})
    n = length(Ixx)
    moments = zeros(n)
    for i in 1:n
        moments[i] = T38_moment(Ixx[i], Iyy[i], Izz[i], Ixz[i], Q_dot[i], P[i], R[i])
    end
    return moments
end