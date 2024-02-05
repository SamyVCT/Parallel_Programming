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
u = np.array([i+1. for i in range(dim)]).reshape(-1, 1)
# print(f"u = {u}")

Nloc = dim//nbp
print("Nloc",Nloc)
jstart = rank*Nloc
jend = (rank+1)*Nloc
res = np.zeros((dim,1))

deb = time()
res[jstart:jend,:] = np.dot(A[jstart:jend,:], u)
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

    # print("calcul correct = ", result==v)


"""
Test avec dim = 6000
nbp =  10

Temps du calcul par le processus 5 : 0.0
 
Temps du calcul par le processus 1 : 0.0156707763671875
 
Temps du calcul par le processus 3 : 0.01687026023864746
 
Temps du calcul par le processus 2 : 0.01563572883605957
 
Temps du calcul par le processus 8 : 0.017382144927978516
 
Temps du calcul par le processus 9 : 0.0204927921295166
 
Temps du calcul par le processus 7 : 0.015586376190185547
 
Temps du calcul par le processus 6 : 0.0
 
Temps du calcul par le processus 4 : 0.0
 
Temps du calcul par le processus 0 : 0.0

Temps de calcul du produit matrice-vecteur sans parallélisation : 0.01579904556274414

"""