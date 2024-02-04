# Produit matrice-vecteur v = A.u
import numpy as np
from mpi4py import MPI
from time import time

globCom = MPI.COMM_WORLD.Dup()
nbp     = globCom.size
rank    = globCom.rank
name    = MPI.Get_processor_name()

print("nbp = ",nbp)

# Dimension du problème (peut-être changé)
dim = 6000
# Initialisation de la matrice
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
# print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
# print(f"u = {u}")

Nloc = dim//nbp
print("Nloc",Nloc)
jstart = rank*Nloc
jend = (rank+1)*Nloc
res = np.zeros((dim,1))

deb = time()
res = np.dot(A[:,jstart:jend], u[jstart:jend])
fin = time()

print(f"Temps du calcul par le processus {rank} : {fin-deb}")

data = globCom.gather(res, root=0)

if rank == 0:
    result = np.zeros((dim,1))
    for i in range(len(data)):
     result = np.add(result, data[i]) 

    deb = time()
    v = A.dot(u)
    fin = time()
    print(f"Temps de calcul du produit matrice-vecteur : {fin-deb}")

    # print("calcul correct = ", (result)==v)


"""
Test avec dim = 6000
nbp =  10

Temps du calcul par le processus 9 : 0.0382232666015625
  
Temps du calcul par le processus 1 : 0.030105113983154297
  
Temps du calcul par le processus 5 : 0.01963973045349121
  
Temps du calcul par le processus 3 : 0.025314807891845703
  
Temps du calcul par le processus 8 : 0.019771099090576172
  
Temps du calcul par le processus 7 : 0.018399715423583984
  
Temps du calcul par le processus 6 : 0.022299766540527344
  
Temps du calcul par le processus 4 : 0.0227813720703125
  
Temps du calcul par le processus 2 : 0.015140056610107422
  
Temps du calcul par le processus 0 : 0.02697300910949707

Temps de calcul du produit matrice-vecteur sans parallélisation : 0.052983760833740234
"""