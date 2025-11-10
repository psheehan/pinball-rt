import astropy.constants as const
import astropy.units as u
import urllib
import requests
import numpy
import os

class Gas:

    def set_properties_from_lambda(self, filename):
        if not os.path.exists(filename):
            if os.path.exists(os.environ["HOME"]+"/.pinballrt/data/gas/"+filename):
                filename = os.environ["HOME"]+"/.pinballrt/data/gas/"+filename
            else:
                web_data_location = 'https://home.strw.leidenuniv.nl/~moldata/datafiles/'+filename
                response = requests.get(web_data_location)
                if response.status_code == 200:
                    if not os.path.exists(os.environ["HOME"]+"/.pinballrt/data/gas"):
                        os.makedirs(os.environ["HOME"]+"/.pinballrt/data/gas")
                    urllib.request.urlretrieve(web_data_location, 
                            os.environ["HOME"]+"/.pinballrt/data/gas/"+filename)
                    filename = os.environ["HOME"]+"/.pinballrt/data/gas/"+filename
                else:
                    print(web_data_location+' does not exist')
                    return   

        f = open(filename)

        for i in range(3):
            f.readline()

        self.mass = float(f.readline())

        f.readline()
        nlev = int(f.readline())
        f.readline()

        self.J = numpy.empty(nlev, dtype="<U6")
        self.E = numpy.empty(nlev, dtype=float)
        self.g = numpy.empty(nlev, dtype=float)
        for i in range(nlev):
            temp, self.E[i], self.g[i], self.J[i] = tuple(f.readline().split())
        self.E = self.E / u.cm

        f.readline()
        ntrans = int(f.readline())
        f.readline()

        self.J_u = numpy.empty(ntrans, dtype=int)
        self.J_l = numpy.empty(ntrans, dtype=int)
        self.A_ul = numpy.empty(ntrans, dtype=float)
        self.nu = numpy.empty(ntrans, dtype=float)
        self.E_u = numpy.empty(ntrans, dtype=float)
        for i in range(ntrans):
            temp, self.J_u[i], self.J_l[i], self.A_ul[i], self.nu[i], \
                    self.E_u[i] = tuple(f.readline().split())
        self.A_ul = self.A_ul * 1.0 / u.second
        self.nu = self.nu * u.GHz
        self.E_u = self.E_u * u.K

        self.B_ul = const.c**2 * self.A_ul / (2*const.h*self.nu**3)

        f.readline()
        npartners = int(f.readline())

        self.partners = []
        self.temp = []
        self.J_u_coll = []
        self.J_l_coll = []
        self.gamma = []
        for i in range(npartners):
            f.readline()
            self.partners.append(f.readline())
            f.readline()
            ncolltrans = int(f.readline())
            f.readline()
            ncolltemps = int(f.readline())
            f.readline()
            self.temp.append(numpy.array(f.readline().split(), dtype=float))
            f.readline()

            self.J_u_coll.append(numpy.empty(ncolltrans, dtype=int))
            self.J_l_coll.append(numpy.empty(ncolltrans, dtype=int))
            self.gamma.append(numpy.empty((ncolltrans,ncolltemps), \
                    dtype=float))

            for j in range(ncolltrans):
                temp, self.J_u_coll[i][j], self.J_l_coll[i][j], temp2 = \
                        tuple(f.readline().split(None,3))
                self.gamma[i][j,:] = numpy.array(temp2.split())

        f.close()

    def partition_function(self, T):
        if not isinstance(T, u.Quantity):
            T = T * u.K
            
        Q = 0.0
        for i in range(len(self.E)):
            Q += self.g[i] * numpy.exp(-const.h * const.c * self.E[i]/(const.k_B*T))
        return Q