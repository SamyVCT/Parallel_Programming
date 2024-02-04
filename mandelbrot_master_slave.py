# Calcul de l'ensemble de Mandelbrot en python
import glob
from threading import local
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


globCom = MPI.COMM_WORLD.Dup()
nbp     = globCom.size
rank    = globCom.rank
name    = MPI.Get_processor_name()

print("nbp = ",nbp)
# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024
scaleX = 3./width
scaleY = 2.25/height

# Master process
if rank == 0:
    # Divide the work among processes
    local_height = height // (nbp - 1)
    for i in range(1, nbp):
        istart = (i - 1) * local_height
        iend = i * local_height if i != nbp - 1 else height
        globCom.send((istart, iend), dest=i)

    # Gather the results from the worker processes
    data = []
    for i in range(1, nbp):
        data.append(globCom.recv(source=i))

    # Combine the results
    result = np.hstack(data)

    # Constitution de l'image résultante :
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(result.T)*255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin-deb}")
    image.show()

# Worker processes
else:
    # Receive the work from the master process
    istart, iend = globCom.recv(source=0)

    # Calculate the Mandelbrot set
    convergence = np.empty((width, iend - istart), dtype=np.double)
    deb = time()
    for y in range(istart, iend):
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
            convergence[x, y - istart] = mandelbrot_set.convergence(c, smooth=True)
    fin = time()
    print(f"Temps du calcul d'un quart de Mandelbrot par le processus {rank} : {fin-deb}")
    # Send the results to the master process
    globCom.send(convergence, dest=0) 



"""
Temps de calcul 

pour nbp = 2 :  2s
pour nbp = 3 :  1.1s par processus
pour nbp = 5 :  0.6s ou 0.7s par processus (il a 2 processus plus rapides parce qu'ils ont la partie la plus simple du calcul : le haut et le bas de l'image
pour nbp = 9 :  entre 0.38s et 0.50s par processus


speedup(3) = 2/1.1 = 1.81
speedup(5) = 2/0.65 = 3.07
speedup(9) = 2/0.44 = 4.54

On observe des temps de calculs très proches de ceux obtenus avec la version précédente où on partitionnait l'image en bandes horizontales.
C'est seulement avec 9 processus que l'on observe un temps de calcul plus élevé que celui obtenu avec l'ancienne méthode.
"""