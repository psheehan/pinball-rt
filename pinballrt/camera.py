from .photons import PhotonList
import astropy.units as u
import astropy.constants as const
import warp as wp
import numpy as np
import xarray as xr
import torch

from .utils import EPSILON

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
        tau_intensity = np.zeros(xflat.shape+(nu.size,), dtype=float)
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
        ray_list.tau_intensity = wp.array2d(tau_intensity, dtype=float)
        ray_list.image_ix = wp.array(image_ix, dtype=int)
        ray_list.image_iy = wp.array(image_iy, dtype=int)
        ray_list.pixel_too_large = wp.array(pixel_too_large, dtype=bool)

        ray_list.density = wp.zeros(xflat.size, dtype=float)
        ray_list.temperature = wp.zeros(xflat.size, dtype=float)
        ray_list.amax = wp.zeros(xflat.size, dtype=float)
        ray_list.p = wp.zeros(xflat.size, dtype=float)

        ray_list.radius = wp.array(np.zeros(xflat.shape), dtype=float)
        ray_list.theta = wp.zeros(xflat.shape, dtype=float)
        ray_list.phi = wp.zeros(xflat.shape, dtype=float)
        ray_list.sin_theta = wp.zeros(xflat.shape, dtype=float)
        ray_list.cos_theta = wp.zeros(xflat.shape, dtype=float)
        ray_list.phi = wp.zeros(xflat.shape, dtype=float)
        ray_list.sin_phi = wp.zeros(xflat.shape, dtype=float)
        ray_list.cos_phi = wp.zeros(xflat.shape, dtype=float)

        return ray_list

    @wp.kernel
    def put_intensity_in_image(image_ix: wp.array(dtype=int),
                               image_iy: wp.array(dtype=int),
                               ray_intensity: wp.array2d(dtype=float),
                               image_intensity: wp.array3d(dtype=float)):

        ir, inu = wp.tid()

        ix, iy = image_ix[ir], image_iy[ir]

        image_intensity[ix, iy, inu] += ray_intensity[ir,inu]

    def raytrace(self, new_x, new_y, nx, ny, image_pixel_size, nu):
        with wp.ScopedDevice(self.grid.device):
            nrays = new_x.size
            pixel_size = image_pixel_size

            intensity = wp.array3d(np.zeros((nx, ny, nu.size), dtype=np.float32), dtype=float)

            while nrays > 0:
                print(nrays)
                ray_list = self.emit_rays(new_x, new_y, nu, nx, ny, image_pixel_size)

                s = wp.zeros(new_x.shape, dtype=float)

                wp.launch(kernel=self.grid.outer_wall_distance,
                        dim=new_x.shape,
                        inputs=[ray_list, self.grid.grid, s])

                s = wp.to_torch(s)
                will_be_in_grid = s < torch.inf
                iwill_be_in_grid = torch.arange(nrays, dtype=torch.int32, device=wp.device_to_torch(wp.get_device()))[will_be_in_grid]

                wp.launch(kernel=self.grid.move,
                          dim=iwill_be_in_grid.shape,
                          inputs=[ray_list, s, iwill_be_in_grid])

                ray_list.position = wp.array(wp.to_torch(ray_list.position)[will_be_in_grid], dtype=wp.vec3)
                ray_list.direction = wp.array(wp.to_torch(ray_list.direction)[will_be_in_grid], dtype=wp.vec3)
                ray_list.intensity = wp.array(wp.to_torch(ray_list.intensity)[will_be_in_grid], dtype=float)
                ray_list.tau_intensity = wp.array(wp.to_torch(ray_list.tau_intensity)[will_be_in_grid], dtype=float)
                ray_list.image_ix = wp.array(wp.to_torch(ray_list.image_ix)[will_be_in_grid], dtype=int)
                ray_list.image_iy = wp.array(wp.to_torch(ray_list.image_iy)[will_be_in_grid], dtype=int)
                ray_list.pixel_too_large = wp.array(wp.to_torch(ray_list.pixel_too_large)[will_be_in_grid], dtype=bool)

                ray_list.density = wp.array(wp.to_torch(ray_list.density)[will_be_in_grid], dtype=float)
                ray_list.temperature = wp.array(wp.to_torch(ray_list.temperature)[will_be_in_grid], dtype=float)
                ray_list.amax = wp.array(wp.to_torch(ray_list.amax)[will_be_in_grid], dtype=float)
                ray_list.p = wp.array(wp.to_torch(ray_list.p)[will_be_in_grid], dtype=float)

                nrays = will_be_in_grid.sum()
                iray = torch.arange(nrays, dtype=torch.int32, device=wp.device_to_torch(wp.get_device()))

                indices = wp.zeros((nrays, 3), dtype=int)
                wp.launch(kernel=self.grid.photon_loc,
                        dim=(nrays,),
                        inputs=[ray_list, self.grid.grid, iray])

                #print(nrays)
                self.grid.propagate_rays(ray_list, nu.values, pixel_size)

                wp.launch(kernel=self.put_intensity_in_image, 
                        dim=(nrays, nu.size),
                        inputs=[ray_list.image_ix, ray_list.image_iy, ray_list.intensity, intensity])

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

        return intensity

    def raytrace_sources(self, x, y, nx, ny, nu, dpc, nrays=1000):
        with wp.ScopedDevice(self.grid.device):
            intensity = wp.array3d(np.zeros((nx, ny, nu.size), dtype=np.float32), dtype=float)
    
            # Also propagate rays from any sources in the grid.
    
            ray_list = self.grid.star.emit_rays(nu, self.grid.distance_unit, self.ez, nrays, dpc, device=self.grid.device)
            iray = torch.arange(nrays, dtype=torch.int32, device=wp.device_to_torch(wp.get_device()))
    
            wp.launch(kernel=self.grid.photon_loc,
                        dim=(nrays,),
                        inputs=[ray_list, self.grid.grid, iray])
            
            self.grid.propagate_rays_from_source(ray_list, nu.values)
    
            ximage = np.dot(ray_list.position.numpy(), self.ey)
            yimage = np.dot(ray_list.position.numpy(), self.ex)
    
            image_ix = (nx * (ximage + x.values.max()) / (2 * x.values.max()) + 0.5).astype(int)
            image_iy = (ny * (yimage + y.values.max()) / (2 * y.values.max()) + 0.5).astype(int)
    
            ray_list.image_ix = wp.array(image_ix, dtype=int)
            ray_list.image_iy = wp.array(image_iy, dtype=int)
    
            wp.launch(kernel=self.put_intensity_in_image,
                        dim=(nrays, nu.size),
                        inputs=[ray_list.image_ix, ray_list.image_iy, ray_list.intensity, intensity])

        return intensity
