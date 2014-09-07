from ..sources import Star
from ..mcrt3d import lib
from ..dust import Dust
import decimal
import numpy
import h5py

class Grid:
    def __init__(self):
        self.density = []
        self.mass = []
        self.temperature = []
        self.dust = []
        self.sources = []

    def add_density(self, density, dust):
        self.density.append(density)
        self.mass.append(density * self.volume)
        self.temperature.append(numpy.ones(density.shape, dtype=float))
        self.dust.append(dust)

    def add_source(self, source):
        self.sources.append(source)

    def set_cartesian_grid(self, w1, w2, w3):
        self.obj = lib.new_CartesianGrid()
        self.coordsystem = "cartesian"

        self.x = 0.5*(w1[0:w1.size-1] + w1[1:w1.size])
        self.y = 0.5*(w2[0:w2.size-1] + w2[1:w2.size])
        self.z = 0.5*(w3[0:w3.size-1] + w3[1:w3.size])

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.volume = numpy.zeros((w1.size-1, w2.size-1, w3.size-1), dtype=float)
        for i in range(self.volume.shape[0]):
            for j in range(self.volume.shape[1]):
                for k in range(self.volume.shape[2]):
                    self.volume[i,j,k] = (self.w1[i+1] - self.w1[i])* \
                        (self.w2[j+1] - self.w2[j])*(self.w3[k+1] - self.w3[k])

        lib.set_walls(self.obj, w1.size-1, w2.size-1, w3.size-1, \
                w1.size, w2.size, w3.size, w1, w2, w3, self.volume)

    def set_cylindrical_grid(self, w1, w2, w3):
        self.obj = lib.new_CylindricalGrid()
        self.coordsystem = "cylindrical"

        self.rho = 0.5*(w1[0:w1.size-1] + w1[1:w1.size])
        self.phi = 0.5*(w2[0:w2.size-1] + w2[1:w2.size])
        self.z = 0.5*(w3[0:w3.size-1] + w3[1:w3.size])

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.volume = numpy.zeros((w1.size-1, w2.size-1, w3.size-1), dtype=float)
        for i in range(self.volume.shape[0]):
            for j in range(self.volume.shape[1]):
                for k in range(self.volume.shape[2]):
                    self.volume[i,j,k] = (self.w1[i+1]**2 - self.w1[i]**2)* \
                        (self.w2[j+1]-self.w2[j]) * (self.w3[k+1]-self.w3[k])/2

        lib.set_walls(self.obj, w1.size-1, w2.size-1, w3.size-1, \
                w1.size, w2.size, w3.size, w1, w2, w3, self.volume)

    def set_spherical_grid(self, w1, w2, w3):
        self.obj = lib.new_SphericalGrid()
        self.coordsystem = "spherical"

        self.r = 0.5*(w1[0:w1.size-1] + w1[1:w1.size])
        self.theta = 0.5*(w2[0:w2.size-1] + w2[1:w2.size])
        self.phi = 0.5*(w3[0:w3.size-1] + w3[1:w3.size])

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.volume = numpy.zeros((w1.size-1, w2.size-1, w3.size-1), dtype=float)
        for i in range(self.volume.shape[0]):
            for j in range(self.volume.shape[1]):
                for k in range(self.volume.shape[2]):
                    self.volume[i,j,k] = (self.w1[i+1]**3 - self.w1[i]**3)* \
                        (self.w3[k+1] - self.w3[k])* \
                        (numpy.cos(self.w2[j]) - numpy.cos(self.w2[j+1]))/3

        lib.set_walls(self.obj, w1.size-1, w2.size-1, w3.size-1, \
                w1.size, w2.size, w3.size, w1, w2, w3, self.volume)

    def set_physical_properties(self):
        lib.create_dust_array(self.obj, len(self.dust))
        lib.create_physical_properties_arrays(self.obj, len(self.dust))
        for i in range(len(self.dust)):
            lib.set_dust(self.obj, self.dust[i].obj, i)

            lib.set_physical_properties(self.obj, self.density[i], \
                    self.temperature[i], self.mass[i], i)

        lib.create_sources_array(self.obj, len(self.sources))
        for i in range(len(self.sources)):
            lib.set_sources(self.obj, self.sources[i].obj, i)

        decimal.getcontext().prec = 80
        y = [i * decimal.Decimal(1) / decimal.Decimal(100) for i in range(101)]
        f = []
        for i in range(len(y)-1):
            n = 1
            while True:
                if n == 1:
                    f.append((-1)**(n+1) * y[i]**(n**2))
                else:
                    f[-1] += (-1)**(n+1) * y[i]**(n**2)
                    if (abs((f[-1] - old_f)) <= decimal.Decimal(1.0e-80) * \
                            abs(f[-1])):
                        break

                n = n+1
                old_f = f[-1]

        f.append(decimal.Decimal(1) / decimal.Decimal(2))
        f = decimal.Decimal(2) * numpy.array(f)

        dydf = numpy.diff(y) / numpy.diff(f)

        self.y = numpy.array(y, dtype=float)
        self.f = numpy.array(f, dtype=float)
        self.dydf = numpy.array(dydf, dtype=float)

        lib.set_mrw_tables(self.obj, self.y, self.f, self.dydf, self.y.size)

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        coordsystem = f['coordsystem'].value
        w1 = f['w1'].value
        w2 = f['w2'].value
        w3 = f['w3'].value

        if (coordsystem == 'cartesian'):
            self.set_cartesian_grid(w1, w2, w3)
        elif (coordsystem == 'cylindrical'):
            self.set_cylindrical_grid(w1, w2, w3)
        elif (coordsystem == 'spherical'):
            self.set_spherical_grid(w1, w2, w3)

        density = f['Density']
        for name in density:
            self.density.append(density[name].value)

        dust = f['Dust']
        for name in dust:
            d = Dust()
            d.set_properties_from_file(usefile=dust[name])
            self.dust.append(d)

        temperature = f['Temperature']
        for name in temperature:
            self.temperature.append(temperature[name].value)

        sources = f['Sources']
        for name in sources:
            source = Source()
            source.read(usefile=sources[name])
            self.sources.append(source)

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        f['coordsystem'] = self.coordsystem
        w1_dset = f.create_dataset("w1", (self.w1.size,), dtype='f')
        w1_dset[...] = self.w1
        w2_dset = f.create_dataset("w2", (self.w2.size,), dtype='f')
        w2_dset[...] = self.w2
        w3_dset = f.create_dataset("w3", (self.w3.size,), dtype='f')
        w3_dset[...] = self.w3

        density = f.create_group("Density")
        density_dsets = []
        for i in range(len(self.density)):
            density_dsets.append(density.create_dataset( \
                    "Density{0:d}".format(i), self.density[i].shape, dtype='f'))
            density_dsets[i][...] = self.density[i]

        dust = f.create_group("Dust")
        dust_groups = []
        for i in range(len(self.dust)):
            dust_groups.append(dust.create_group("Dust{0:d}".format(i)))
            self.dust[i].write(usefile=dust_groups[i])

        temperature = f.create_group("Temperature")
        temperature_dsets = []
        for i in range(len(self.temperature)):
            temperature_dsets.append(temperature.create_dataset( \
                    "Temperature{0:d}".format(i), self.temperature[i].shape, \
                    dtype='f'))
            temperature_dsets[i][...] = self.temperature[i]

        sources = f.create_group("Sources")
        sources_groups = []
        for i in range(len(self.sources)):
            sources_groups.append(sources.create_group("Star{0:d}".format(i)))
            self.sources[i].write(usefile=sources_groups[i])

        if (usefile == None):
            f.close()
