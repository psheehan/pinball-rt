import warp as wp
import torch

EPSILON = 1.0e-5

@wp.func
def equal(x: float,
          y: float,
          tol: float):
    if (wp.abs(x-y) < wp.abs(y)*tol):
        return True
    else:
        return False
    
@wp.func
def equal_zero(x: float,
            tol: float):
    if (wp.abs(x) < tol):
        return True
    else:
        return False

@wp.func
def planck_function(nu: float,
                    temperature: float):
    #h = 6.6260755e-27
    #k_B = 1.380658e-16
    #c_l = 2.99792458e10
    # nu in units of GHz
    # output in Jy

    #return 2.0*h*nu*nu*nu/(c_l*c_l)*1.0/(wp.exp(h*nu/(k_B*temperature))-1.0);
    return 1474.49946476 * nu * 1.0/(wp.exp(0.04799243*nu/temperature)-1.0);

@wp.kernel
def log_uniform_interp(x: wp.array(dtype=float),
                       xp: wp.array(dtype=float),
                       fp: wp.array(dtype=float),
                       f: wp.array(dtype=float)):
     
    ip = wp.tid()

    index = int((wp.log10(x[ip]) - wp.log10(xp[0])) / (wp.log10(xp[1]) - wp.log10(xp[0])))

    f[ip] = (x[ip] - xp[index]) * (fp[index+1] - fp[index]) / \
        (xp[index+1] - xp[index]) + fp[index]
