import astropy.units as u
import astropy.constants as const
import warp as wp
import numpy as np

from .utils import EPSILON

class Image:
    def __init__(self, nx, ny, pixel_size, lam):
        self.nx = nx
        self.ny = ny

        x = (np.arange(nx) - nx / 2)*pixel_size
        y = (np.arange(ny) - ny / 2)*pixel_size

        self.x, self.y = np.meshgrid(x, y)

        self.pixel_size = pixel_size

        self.lam = lam.copy()
        self.nu = const.c.cgs.value / lam

        self.intensity = wp.array3d(np.zeros((nx, ny, lam.size), dtype=float), dtype=float)

class Camera:
    def __init__(self, grid):
        self.grid = grid

    def set_orientation(self, incl, pa, dpc):
        # Set viewing angle parameters.

        #self.r = (dpc*u.pc).cgs.value;
        self.r = self.grid.grid_size()
        self.incl = incl * np.pi/180.
        self.pa = pa * np.pi/180.

        phi = -np.pi/2 - self.pa

        self.i = np.array([self.r*np.sin(self.incl)*np.cos(phi), \
                self.r*np.sin(self.incl)*np.sin(phi), \
                self.r*np.cos(self.incl)])

        self.ex = np.array([-np.sin(phi), np.cos(phi), 0.0])
        self.ey = np.array([-np.cos(self.incl)*np.cos(phi), \
                -np.cos(self.incl)*np.sin(phi), \
                np.sin(self.incl)])
        self.ez = np.array([-np.sin(self.incl)*np.cos(phi), \
                -np.sin(self.incl)*np.sin(phi), \
                -np.cos(self.incl)])

    def emit_rays(self, x, y, nu, nx, ny, pixel_size):
        intensity = np.zeros(x.shape+(nu.size,), dtype=np.float32)
        tau = np.zeros(x.shape+(nu.size,), dtype=float)
        #image_ix, image_iy = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1]))
        image_ix = (x / pixel_size + nx / 2).astype(np.int32)
        image_iy = (y / pixel_size + ny / 2).astype(np.int32)

        pixel_too_large = np.zeros(x.shape).astype(bool)

        position = np.broadcast_to(self.i, x.shape+(3,)) + np.expand_dims(x, axis=-1)*self.ex + np.expand_dims(y, axis=-1)*self.ey
        direction = np.broadcast_to(self.ez, x.shape+(3,))
        direction = np.where(np.abs(direction) < EPSILON, 0., direction)

        position = wp.array2d(position, dtype=wp.vec3)
        direction = wp.array2d(direction, dtype=wp.vec3)

        return intensity, tau, pixel_too_large, position, direction, image_ix, image_iy

    @wp.kernel
    def put_intensity_in_image(image_ix: wp.array(dtype=int),
                               image_iy: wp.array(dtype=int),
                               ray_intensity: wp.array2d(dtype=float),
                               image_intensity: wp.array3d(dtype=float)):

        ir, inu = wp.tid()

        ix, iy = image_ix[ir], image_iy[ir]

        image_intensity[ix, iy, inu] = ray_intensity[ir,inu]

    def make_image(self, nx, ny, pixel_size, lam, incl, pa, dpc):
        self.set_orientation(incl, pa, dpc)

        image = Image(nx, ny, (pixel_size*u.arcsecond*dpc*u.pc).cgs.value, lam)

        nrays = nx*ny
        new_x, new_y = image.x.flatten(), image.y.flatten()
        pixel_size = image.pixel_size

        while nrays > 0:
            intensity, tau, pixel_too_large, position, direction, image_ix, image_iy = \
                    self.emit_rays(new_x, new_y, image.nu, image.nx, image.ny, image.pixel_size)

            s = wp.zeros(new_x.shape, dtype=float)
        
            wp.launch(kernel=self.grid.outer_wall_distance,
                    dim=new_x.shape,
                    inputs=[position, direction, self.grid.grid, s],
                    device='cpu')

            s = s.numpy()
            will_be_in_grid = s < np.inf

            position = position.numpy()[will_be_in_grid]
            direction = direction.numpy()[will_be_in_grid]
            s = s[will_be_in_grid]

            position = position + s[:,np.newaxis]*direction

            position = wp.array(position, dtype=wp.vec3)
            direction = wp.array(direction, dtype=wp.vec3)
            intensity = intensity[will_be_in_grid]
            tau = wp.array(tau[will_be_in_grid], dtype=float)
            image_ix = image_ix[will_be_in_grid]
            image_iy = image_iy[will_be_in_grid]
            pixel_too_large = pixel_too_large[will_be_in_grid]

            nrays = will_be_in_grid.sum()
            iray = np.arange(nrays, dtype=np.int32)

            indices = wp.zeros((nrays, 3), dtype=int)
            wp.launch(kernel=self.grid.photon_loc,
                    dim=(nrays,),
                    inputs=[position, direction, self.grid.grid, indices, iray],
                    device='cpu')

            self.grid.propagate_rays(position, direction, indices, intensity, tau, image.nu, pixel_size, pixel_too_large)

            wp.launch(kernel=self.put_intensity_in_image, 
                    dim=(nrays, image.lam.size),
                    inputs=[image_ix, image_iy, intensity, image.intensity])

            new_x, new_y = [], []
            for i in range(nrays):
                if pixel_too_large[i]:
                    for j in range(4):
                        new_x.append(image_ix[i] + (-1)**j * pixel_size/4)
                        new_y.append(image_iy[i] + (-1)**(int(j/2)) * pixel_size/4)
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            pixel_size = pixel_size / 2
            nrays = len(new_x)

        return image