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

image     = np.zeros((width, height),dtype=np.double)
image_loc = np.zeros((width, height),dtype=np.double)

# Master process
if rank == 0:
    # Divide the work among processes
    next_line = 0
    for i in range(1, nbp):
        globCom.send(next_line, dest=i)
        next_line += 1

    stat : MPI.Status = MPI.Status() 
    while next_line < height:
        done = globCom.recv(status=stat)# On reçoit du premier process à envoyer un message
        slaveRk = stat.source
        globCom.send(next_line, dest=slaveRk)
        next_line +=1
    next_line = -1
    for i in range(1, nbp):
        status = MPI.Status()
        done = globCom.recv(status=status)# On reçoit du premier process à envoyer un message
        slaveRk : int = status.source
        globCom.send(next_line, dest=slaveRk)
    globCom.Reduce([image_loc,MPI.INT64_T], [image,MPI.INT64_T], op=MPI.SUM, root=0)

# Worker processes
else:
    status : MPI.Status = MPI.Status()
    next_line : int
    res   : int = 1

    next_line = globCom.recv(source=0)
    while next_line != -1:
        # Calculate the Mandelbrot set for the line
        image_loc = np.empty((width, height), dtype=np.double)
        # deb = time()
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * next_line)
            image_loc[x,next_line] = mandelbrot_set.convergence(c, smooth=True)
        # fin = time()
        # print(f"Temps du calcul de la ligne {next_line} par le processus {rank} : {fin-deb}")
        req : MPI.Request = globCom.isend(res,0)
        image += image_loc
        next_line = globCom.recv(source=0)
    globCom.Reduce([image,MPI.INT64_T], None, op=MPI.SUM, root=0)

if rank == 0:
    # Constitution de l'image résultante :
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(image.T)*255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin-deb}")
    image.show()

