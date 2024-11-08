"""
    trim_aoa(dyn_press::Float64) -> Float64

Compute the trim angle of attack (in radians) as a function of dynamic pressure. Trim function was fit at Mach 0.7.

# Arguments
- `dyn_press`: Dynamic pressure (psf).

# Returns
- Trim angle of attack in radians.
"""
function trim_aoa(dyn_press::Float64)
    # Valid at 0.7 Mach
    a = 8.869e+00
    b = 9.383e-05
    return deg2rad(a * exp(-b * dyn_press))
end

"""
    trim_stab_pos(dyn_press::Float64) -> Float64

Compute the trim stabilizer position (in radians) as a function of dynamic pressure. Trim function was fit at Mach 0.7.

# Arguments
- `dyn_press`: Dynamic pressure (psf).

# Returns
- Trim stabilizer position in radians.
"""
function trim_stab_pos(dyn_press::Float64)
    # Valid at 0.7 Mach
    intercept = -10.182059
    slope = 0.931074
    return deg2rad(intercept + slope * log(dyn_press))
end