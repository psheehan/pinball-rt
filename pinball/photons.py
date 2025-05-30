import warp as wp

@wp.struct
class PhotonList:
    position: wp.array(dtype=wp.vec3)
    direction: wp.array(dtype=wp.vec3)
    indices: wp.array2d(dtype=int)
    frequency: wp.array(dtype=float)
    energy: wp.array(dtype=float)
    in_grid: wp.array(dtype=bool)

    intensity: wp.array2d(dtype=float)
    tau: wp.array(dtype=float)
    image_ix: wp.array(dtype=int)
    image_iy: wp.array(dtype=int)
    pixel_too_large: wp.array(dtype=bool)