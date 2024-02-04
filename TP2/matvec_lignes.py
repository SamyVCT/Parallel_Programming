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
dim = 5000
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
for j in range(dim):
    for i in range(jstart,jend):
        res[i] += A[i][j] * u[j]
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

    # print("calcul correct = ", np.transpose(result)==v)


"""
Test avec dim = 5000
nbp =  10
entre 7.3s et 8.2s par processus
A.dot(u) en  0.0103s
"""