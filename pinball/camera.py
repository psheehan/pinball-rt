from .photons import PhotonList
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
        xflat, yflat = x.flatten(), y.flatten()
        intensity = np.zeros(xflat.shape+(nu.size,), dtype=np.float32)
        tau = np.zeros(xflat.shape+(nu.size,), dtype=float)
        #image_ix, image_iy = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1]))
        image_ix = (xflat / pixel_size + nx / 2).astype(np.int32)
        image_iy = (yflat / pixel_size + ny / 2).astype(np.int32)

        pixel_too_large = np.zeros(xflat.shape).astype(bool)

        position = np.broadcast_to(self.i, xflat.shape+(3,)) + np.expand_dims(xflat, axis=-1)*self.ex + np.expand_dims(yflat, axis=-1)*self.ey
        direction = np.broadcast_to(self.ez, xflat.shape+(3,))
        direction = np.where(np.abs(direction) < EPSILON, 0., direction)

        ray_list = PhotonList()
        ray_list.position = wp.array(position, dtype=wp.vec3)
        ray_list.direction = wp.array(direction, dtype=wp.vec3)
        ray_list.indices = wp.zeros(xflat.shape+(3,), dtype=int)
        ray_list.intensity = wp.array2d(intensity, dtype=float)
        ray_list.tau = wp.array2d(tau, dtype=float)
        ray_list.image_ix = wp.array(image_ix, dtype=int)
        ray_list.image_iy = wp.array(image_iy, dtype=int)
        ray_list.pixel_too_large = wp.array(pixel_too_large, dtype=bool)

        return ray_list

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
            ray_list = self.emit_rays(new_x, new_y, image.nu, image.nx, image.ny, image.pixel_size)

            s = wp.zeros(new_x.shape, dtype=float)

            wp.launch(kernel=self.grid.outer_wall_distance,
                    dim=new_x.shape,
                    inputs=[ray_list, self.grid.grid, s],
                    device='cpu')

            s = s.numpy()
            will_be_in_grid = s < np.inf
            iwill_be_in_grid = np.arange(nrays, dtype=np.int32)[will_be_in_grid]

            wp.launch(kernel=self.grid.move,
                      dim=iwill_be_in_grid.shape,
                      inputs=[ray_list, s, iwill_be_in_grid],
                      device='cpu')

            ray_list.position = wp.array(ray_list.position.numpy()[will_be_in_grid], dtype=wp.vec3)
            ray_list.direction = wp.array(ray_list.direction.numpy()[will_be_in_grid], dtype=wp.vec3)
            ray_list.intensity = wp.array(ray_list.intensity.numpy()[will_be_in_grid], dtype=float)
            ray_list.tau = wp.array(ray_list.tau.numpy()[will_be_in_grid], dtype=float)
            ray_list.image_ix = wp.array(ray_list.image_ix.numpy()[will_be_in_grid], dtype=int)
            ray_list.image_iy = wp.array(ray_list.image_iy.numpy()[will_be_in_grid], dtype=int)
            ray_list.pixel_too_large = wp.array(ray_list.pixel_too_large.numpy()[will_be_in_grid], dtype=bool)

            nrays = will_be_in_grid.sum()
            iray = np.arange(nrays, dtype=np.int32)

            indices = wp.zeros((nrays, 3), dtype=int)
            wp.launch(kernel=self.grid.photon_loc,
                    dim=(nrays,),
                    inputs=[ray_list, self.grid.grid, iray],
                    device='cpu')

            self.grid.propagate_rays(ray_list, image.nu, pixel_size)

            wp.launch(kernel=self.put_intensity_in_image, 
                    dim=(nrays, image.lam.size),
                    inputs=[ray_list.image_ix, ray_list.image_iy, ray_list.intensity, image.intensity])

            new_x, new_y = [], []
            for i in range(nrays):
                if ray_list.pixel_too_large.numpy()[i]:
                    for j in range(4):
                        new_x.append(ray_list.image_ix.numpy()[i] + (-1)**j * pixel_size/4)
                        new_y.append(ray_list.image_iy.numpy()[i] + (-1)**(int(j/2)) * pixel_size/4)
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            pixel_size = pixel_size / 2
            nrays = len(new_x)

        return image