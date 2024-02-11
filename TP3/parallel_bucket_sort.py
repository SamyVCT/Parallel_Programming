from mpi4py import MPI
import numpy as np
import time


def parallel_bucket_sort():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    buckets = [[] for _ in range(size)]
    if rank == 0:
        # Générer des nombres aléatoires
        max = 1000000000
        min = 0
        arr = np.random.randint(min, max, size=1000000)
        # print("numbers random",arr)
        arr2 = arr.copy()
        deb = time.time()
        arr2.sort()
        fin = time.time()
        print(f"Temps de tri initial pour le processus {rank} avec {len(arr2)} valeurs : {fin-deb}")
        # print("numbers trie",arr2)
        # Distribuer les nombres à tous les processus
        # min_value, max_value = np.min(arr), np.max(arr)
        for num in arr:
            i = int((num - min) / (max - min + 1) * size)
            buckets[i].append(num)
    
    # Chaque processus reçoit ses nombres
    data = comm.scatter(buckets, root=0)
    deb = time.time()
    # Chaque processus trie son tableau
    data.sort()
    fin = time.time()
    print(f"Temps de tri pour le processus {rank} avec {len(data)} valeurs : {fin-deb}")
    # print("bucket sorted",data)
    # Le processus maître recueille les tableaux triés
    sorted_data = comm.gather(data, root=0)

    if rank == 0:
        # Le processus maître fusionne les tableaux triés
        sorted_data = np.concatenate(sorted_data)
        # print("parallel sorted", sorted_data, len(sorted_data))


if __name__ == "__main__":
    parallel_bucket_sort()


"""
Résultats :
Avec 500000 valeurs et 8 processus, on gagne 1.5 centième de seconde
Temps de tri pour le processus 1 avec 62619 valeurs : 0.016013383865356445
Temps de tri pour le processus 2 avec 62232 valeurs : 0.026132822036743164
Temps de tri pour le processus 3 avec 62450 valeurs : 0.026132822036743164
Temps de tri pour le processus 4 avec 62551 valeurs : 0.010507345199584961
Temps de tri pour le processus 5 avec 62795 valeurs : 0.016013383865356445
Temps de tri pour le processus 7 avec 62269 valeurs : 0.005506038665771484
Temps de tri pour le processus 6 avec 62147 valeurs : 0.016013383865356445
Temps de tri initial pour le processus 0 avec 500000 valeurs : 0.031701087951660156
Temps de tri pour le processus 0 avec 62937 valeurs : 0.016013383865356445

Avec 1000000 valeurs et 10 processus, on gagne 1.5 centième de seconde
Temps de tri pour le processus 1 avec 99742 valeurs : 0.03114151954650879
Temps de tri pour le processus 2 avec 99876 valeurs : 0.03114151954650879
Temps de tri pour le processus 3 avec 100001 valeurs : 0.031258583068847656
Temps de tri pour le processus 4 avec 99944 valeurs : 0.03114151954650879
Temps de tri pour le processus 6 avec 99746 valeurs : 0.015633583068847656
Temps de tri pour le processus 5 avec 99861 valeurs : 0.031258583068847656
Temps de tri pour le processus 8 avec 100587 valeurs : 0.031258583068847656
Temps de tri pour le processus 7 avec 99981 valeurs : 0.03114151954650879
Temps de tri pour le processus 9 avec 100124 valeurs : 0.031258583068847656
Temps de tri initial pour le processus 0 avec 1000000 valeurs : 0.04531121253967285
Temps de tri pour le processus 0 avec 100138 valeurs : 0.03114151954650879
"""