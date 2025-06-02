import warp as wp

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
    h = 6.6260755e-27
    k_B = 1.380658e-16
    c_l = 2.99792458e10

    return 2.0*h*nu*nu*nu/(c_l*c_l)*1.0/(wp.exp(h*nu/(k_B*temperature))-1.0);