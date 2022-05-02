import numpy as np

def bissection_onestep(f,a,b):
    if not np.all(f(a)*f(b) <= 0):
        raise ValueError("No sign change")
    else:
        mid_point = (a + b)/2
        mid_value = f(mid_point)
        new_a = a
        new_b = b
        indxs_a = np.nonzero(mid_value*f(b) <= 0)
        indxs_b = np.nonzero(mid_value*f(a) <= 0)
        if indxs_a[0].size != 0 and indxs_a[1].size != 0:
            new_a[indxs_a[0],indxs_a[1]] = mid_point[indxs_a[0],indxs_a[1]]
        if indxs_b[0].size != 0 and indxs_b[0].size != 0:
            new_b[indxs_b[0],indxs_b[1]] = mid_point[indxs_b[0],indxs_b[1]]
        return new_a,new_b

def vec_bissection(f,a,b,iter_max = 100,tol = 1E-11):
    i = 1
    err = 1
    while i < iter_max and err > tol:
        a,b = bissection_onestep(f,a,b)
        err = np.max(np.abs(a - b))
        i += 1
    return a, b
