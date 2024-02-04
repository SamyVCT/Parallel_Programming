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
res[jstart:jend,:] = np.dot(A[jstart:jend,:], u).reshape(-1, 1)
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
Test avec dim = 6000
nbp =  10

Temps du calcul par le processus 6 : 0.019979000091552734
   
Temps du calcul par le processus 7 : 0.020400524139404297
   
Temps du calcul par le processus 3 : 0.009933233261108398
   
Temps du calcul par le processus 2 : 0.020172834396362305
   
Temps du calcul par le processus 9 : 0.005138874053955078
   
Temps du calcul par le processus 8 : 0.003542184829711914
   
Temps du calcul par le processus 5 : 0.0
   
Temps du calcul par le processus 4 : 0.019861221313476562
   
Temps du calcul par le processus 1 : 0.0058672428131103516
   
Temps du calcul par le processus 0 : 0.0040285587310791016

Temps de calcul du produit matrice-vecteur sans parallélisation : 0.0060787200927734375
"""