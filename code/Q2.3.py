import numpy as np
import random


# Nombres d'arrêtes entre n'importe quelle paire de communauté
def nb_edges(vector, G, i, j):
    nb = 0

    vector_i = np.array([])
    vector_j = np.array([])
    for l in range(0, vector.size):
        if vector[l] == i:
            vector_i = np.append(vector_i, l)
        if vector[l] == j:
            vector_j = np.append(vector_j, l)

    for l in list(vector_i):
        for k in list(vector_j):
            if G[int(l)][int(k)] == 1:
                nb += 1
    if (i == j):
        nb = int(nb / 2)

    return nb


# Nombres de non-arrêtes entre n'importe quelle paire de communauté
def nb_non_edges(vector, i, j, NB_edges_vector, NB_commu):
    nb = 0

    if (i == j):
        nb = ((abs(NB_commu[i - 1]) *
               (abs(NB_commu[i - 1]) - 1)) / 2) - NB_edges_vector[i - 1][j - 1]
    else:
        nb = (abs(NB_commu[i - 1]) *
              abs(NB_commu[j - 1])) - NB_edges_vector[i - 1][j - 1]

    return int(nb)


# P(X=x) ou P(X=y)
def P_vector(vector, p, K, NB_commu):
    result = 0.0

    for i in range(1, K + 1):
        result += float(np.log(p[i - 1])) * abs(NB_commu[i - 1])

    return result


# P(G|x) ou P(G|y)
def P_G_vector(vector, K, A, B, NB_edges_vector, NB_commu):
    result = 0.0

    for i in range(1, K + 1):  # valeur des communautés ∈ {1,..., K}
        for j in range(i, K + 1):
            if (i == j):
                result += float(
                    (np.log(A)) * NB_edges_vector[i - 1][j - 1]) + float(
                        np.log(1 - A)) * nb_non_edges(
                            vector, i, j, NB_edges_vector, NB_commu)
            else:
                result += float(
                    (np.log(B)) * NB_edges_vector[i - 1][j - 1]) + float(
                        np.log(1 - B)) * nb_non_edges(
                            vector, i, j, NB_edges_vector, NB_commu)

    return result


# Distribution de proposition simple, q_s, consistant simplement à choisir aléatoirement un noeud du graphe et à en changer la communauté aléatoirement
def q_s(vector, G, K, NB_edges_vector, NB_commu):
    rand_position = random.randint(0, vector.size - 1)
    old_k = vector[rand_position]
    numbers = list(range(1, K + 1))
    numbers.remove(old_k)
    rand_k = random.choice(numbers)

    if (old_k != rand_k):

        # Met à jour le nombre de noeuds de chaque communauté :
        NB_commu[old_k - 1] -= 1
        NB_commu[rand_k - 1] += 1

        vector[rand_position] = rand_k

        # Met à jour le NB_edges :
        for l in range(0, G.shape[0]):
            if (G[rand_position][l] == 1):

                NB_edges_vector[rand_k - 1][vector[l] - 1] += 1
                NB_edges_vector[old_k - 1][vector[l] - 1] -= 1

                if (rand_k != vector[l]):
                    NB_edges_vector[vector[l] - 1][rand_k - 1] += 1
                if (old_k != vector[l]):
                    NB_edges_vector[vector[l] - 1][old_k - 1] -= 1
                if (NB_edges_vector[old_k - 1][vector[l] - 1] < 0):
                    print("erreur taille communaute")
    else:
        print("erreur old_k != rand_k")

    return vector, NB_edges_vector, NB_commu


# Fonction permettant d'appeller uniquement 1x la fonction nb_edges() et de stocker sa réponse dans un tableau afin d'éviter de reparcourir plusieur fois le graphe (ce qui prendrait énormément de temps)
def NB_edges(vector, G, K):
    NB = np.zeros((K, K), dtype=int)

    for i in range(1, K + 1):
        for j in range(i, K + 1):
            nb = nb_edges(vector, G, i, j)

            NB[i - 1][j - 1] = nb
            NB[j - 1][i - 1] = nb

    return NB


# Algorithme de Metropolis-Hastings
def MH(x_before, G, p, K, A, B, NB_edges_x, t_max):
    t = 1
    max_P_G_x_P_x = float('-inf')
    tab_max_P_G_x_P_x = list()
    tab_x = list()

    # Stocke le nombre de noeuds de chaque communauté, afin d'éviter de reparccourir constamment le vecteur x :
    NB_commu_x = np.zeros((K, ), dtype=int)
    for node in x_before:
        NB_commu_x[node - 1] += 1

    while (t < t_max):

        alpha_denom = P_G_vector(x_before, K, A, B, NB_edges_x,
                                 NB_commu_x) + P_vector(
                                     x_before, p, K, NB_commu_x)

        NB_edges_y = np.copy(NB_edges_x)
        y = np.copy(x_before)
        NB_commu_y = np.copy(NB_commu_x)
        y, NB_edges_y, NB_commu_y = q_s(y, G, K, NB_edges_y, NB_commu_y)

        alpha_num = P_G_vector(y, K, A, B, NB_edges_y, NB_commu_y) + P_vector(
            y, p, K, NB_commu_y)

        alpha = min(1.0, np.exp(alpha_num - alpha_denom))

        u = random.uniform(0, 1)
        if u < alpha:

            x_before = np.copy(y)
            NB_edges_x = np.copy(NB_edges_y)
            NB_commu_x = np.copy(NB_commu_y)

            if (alpha_num > max_P_G_x_P_x):
                max_P_G_x_P_x = alpha_num
                tab_max_P_G_x_P_x.append(max_P_G_x_P_x)
                tab_x.append(np.copy(x_before))

        t = t + 1

    return tab_max_P_G_x_P_x, tab_x


# Fonction dédiée à la compétition ayant pour but de détecter le plus précisément possible les communautés du graphe G fourni
# Renvoie un vecteur x représentant une estimation de la distribution de communautés parmis les noeuds du graphe
def competition(p, K, t_max, nb_MH, A, B, N):
    G = np.load('G.npy')

    MH_max = float('-inf')
    MH_x = np.array([])

    x_0 = np.ones((N, ), dtype=int)
    for j in range(0, N):
        u = random.uniform(0, 1)
        if u < 0.5:
            x_0[j] = 2

    NB_edges_x = NB_edges(x_0, G, K)

    for i in range(0, nb_MH):

        # pour avoir une idée de l'avancement du code
        if (i != 0):
            print(str(int((float(i) / nb_MH) * 100)) + " %")

        tab_max, tab_x = MH(np.copy(x_0), G, p, K, A, B, np.copy(NB_edges_x),
                            t_max)

        max_tab_max = float('-inf')
        index_x = 0

        for k in range(0, len(tab_max)):
            if (tab_max[k] > max_tab_max):
                max_tab_max = tab_max[k]
                index_x = k

        if (max_tab_max > MH_max):
            MH_max = max_tab_max
            MH_x = tab_x[index_x]

    return MH_x, MH_max


if __name__ == '__main__':
    print("\n\n***Start***")
    MH_x, MH_max = competition(p=np.array([0.5, 0.5]),
                               K=2,
                               t_max=200000,
                               nb_MH=1,
                               A=39.76 / 16572,
                               B=3.29 / 16572,
                               N=16572)

    print("\n\n>>>>> MH_max = " + str(MH_max) + " <<<<<")

    np.savetxt('x.csv', MH_x, delimiter=',', fmt='%i')
    print("***End***\n\n")
