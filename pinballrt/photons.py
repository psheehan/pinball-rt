import warp as wp

@wp.struct
class PhotonList:
    position: wp.array(dtype=wp.vec3)
    direction: wp.array(dtype=wp.vec3)
    indices: wp.array2d(dtype=int)
    frequency: wp.array(dtype=float)
    energy: wp.array(dtype=float)
    in_grid: wp.array(dtype=bool)
    do_ml_step: wp.array(dtype=bool)
    
    deposited_energy: wp.array(dtype=float)

    density: wp.array(dtype=float)
    temperature: wp.array(dtype=float)
    alpha: wp.array(dtype=float)
    kabs: wp.array(dtype=float)
    ksca: wp.array(dtype=float)
    albedo: wp.array(dtype=float)
    kext: wp.array2d(dtype=float)
    ray_albedo: wp.array2d(dtype=float)
    absorb: wp.array(dtype=bool)
    amax: wp.array(dtype=float)
    p: wp.array(dtype=float)

    tau: wp.array(dtype=float)
    total_tau_abs: wp.array(dtype=float)

    intensity: wp.array2d(dtype=float)
    tau_intensity: wp.array2d(dtype=float)
    image_ix: wp.array(dtype=int)
    image_iy: wp.array(dtype=int)
    pixel_too_large: wp.array(dtype=bool)

    radius: wp.array(dtype=float)
    theta: wp.array(dtype=float)
    sin_theta: wp.array(dtype=float)
    cos_theta: wp.array(dtype=float)
    phi: wp.array(dtype=float)
    sin_phi: wp.array(dtype=float)
    cos_phi: wp.array(dtype=float)