import numpy as np
import random
from matplotlib import pyplot as plt


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


# Fonction permettant d'appeller uniquement 1x la fonction nb_edges() et de stocker sa réponse dans un tableau afin d'éviter de reparcourir plusieurs fois le graphe (ce qui prendrait énormément de temps)
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


# Pour mesurer la qualité d’une estimation, on utilisera la concordance entre xstar le vecteur de communautés ayant servi à sa génération, et x une estimation de ce vecteur
def concordance(xstar, x, N, K):
    x_reverse = np.ones((N, ), dtype=int)
    for i in range(0, N):
        if (x[i] == x_reverse[i] == 1):
            x_reverse[i] = 2
    Sk = list()
    Sk.append(x)
    Sk.append(x_reverse)
    max = 0
    for pi in Sk:
        sum_current_pi = 0.0
        for i in range(0, N):
            if (xstar[i] == pi[i]):  # pi(x[i])
                sum_current_pi += float(1) / N
        if (sum_current_pi > max):
            max = sum_current_pi
    return max


# Génération aléatoire de graphe à partir d’un modèle probabiliste particulier, appelé modèle à blocs stochastiques (SBM)
def sbm_graph(N, A, B):

    vector = np.ones((N, ), dtype=int)
    for i in range(0, N):
        u = random.uniform(0, 1)
        if u < 0.5:
            vector[i] = 2

    graph = np.zeros((N, N), dtype=int)

    for i in range(0, N):
        for j in range(i + 1, N):
            u = random.uniform(0, 1)

            if (vector[i] == vector[j]):
                if u < A:
                    graph[i][j] += 1
                    graph[j][i] += 1
            else:
                if u < B:
                    graph[i][j] += 1
                    graph[j][i] += 1

    return graph, vector


# Calcul des concordances moyennes de plusieurs vecteurs x par rapport à plusieurs graphes G et vecteurs xstar associés, tous générés au préalable
def mean_concordance(p, K, t_max, nb_MH, nb_G, N, a, b):

    A = float(a) / N
    B = float(b) / N

    # vecteur de départ généré aléatoirement
    x_0 = np.ones((N, ), dtype=int)
    for i in range(0, N):
        u = random.uniform(0, 1)
        if u < 0.5:
            x_0[i] = 2

    tab_concordance_random = np.array([])
    tab_concordance_G = np.array([])
    for i in range(0, nb_G):

        # génération graphe et vecteur associé
        G, xstar = sbm_graph(N, A, B)

        # concordance avec les communautés choisies de manière purement aléatoire
        concordance_xstar_x_random = concordance(xstar, x_0, N, K)
        tab_concordance_random = np.append(tab_concordance_random,
                                           concordance_xstar_x_random)

        # concordance avec l'algorithme de Metropolis-Hastings
        NB_edges_x = NB_edges(np.copy(x_0), np.copy(G), K)

        tab_concordance_MH = np.array([])
        for j in range(0, nb_MH):

            tab_max, tab_x = MH(np.copy(x_0), G, p, K, A, B,
                                np.copy(NB_edges_x), t_max)

            max_tab_max = float('-inf')
            best_tab_x = np.array([])
            index_x = 0

            if (len(tab_max) > 0):

                for k in range(0, len(tab_max)):
                    if (tab_max[k] > max_tab_max):
                        max_tab_max = tab_max[k]
                        index_x = k

                best_tab_x = np.copy(tab_x[index_x])

                concordance_xstar_x = concordance(xstar, best_tab_x, N, K)
                tab_concordance_MH = np.append(tab_concordance_MH,
                                               concordance_xstar_x)

        if (len(tab_concordance_MH) > 0):
            mean_MH = np.mean(tab_concordance_MH)
            tab_concordance_G = np.append(tab_concordance_G, mean_MH)

    mean_G = -1
    mean_G_random = -1
    if (len(tab_concordance_G) > 0 and len(tab_concordance_random) > 0):
        mean_G = np.mean(tab_concordance_G)
        mean_G_random = np.mean(tab_concordance_random)

    return mean_G, mean_G_random


# Représentation de l'évolution de la concordance moyenne calculée au préalable
def evol_mean_concordance(deg_moy, p, K, t_max, nb_MH, nb_G, N):

    # conditions de bon fonctionnement du code
    if (deg_moy >= N):
        print("\nerror : deg_moy >= N")
        return
    elif (deg_moy < 3):
        print("\nerror : deg_moy < 3")
        return
    elif (K > 2):
        print("\nerror : K > 2")
        return

    authorised_a = list(range(int(((2 * deg_moy) / 2) + 1), (2 * deg_moy)))
    authorised_b = list(range(1, int((2 * deg_moy) / 2)))

    tab_mean = list()
    tab_mean_random = list()
    tab_deg_moy = list()

    for i in range(0, len(authorised_a)):

        # pour avoir une idée de l'avancement du code
        if (i != 0):
            print(str(int((float(i) / len(authorised_a)) * 100)) + " %")

        j = len(authorised_a) - i - 1

        mean, mean_random = mean_concordance(p,
                                             K,
                                             t_max,
                                             nb_MH,
                                             nb_G,
                                             N,
                                             a=authorised_a[i],
                                             b=authorised_b[j])

        if (mean != -1 and mean_random != -1):
            tab_mean.append(mean)
            tab_mean_random.append(mean_random)
            tab_deg_moy.append(float(authorised_b[j]) / authorised_a[i])

    if (len(tab_mean) == 0 and len(tab_mean_random) == 0):
        print("\nerror : no mean could be calculated")
        return
    else:
        # plot tab_mean
        plt.plot(tab_deg_moy, tab_mean, marker='o')
        plt.xlabel('Rapport b/a')
        plt.ylabel('Concordance moyenne')
        #plt.title('Evolution de la concordance moyenne en fonction du rapport b/a dans [0,1] avec le degré moyen = '+str(deg_moy))
        plt.savefig('evol_mean_concordance.png', bbox_inches='tight')
        plt.clf()

        # plot tab_mean_random
        plt.plot(tab_deg_moy, tab_mean_random, marker='o')
        plt.xlabel('Rapport b/a')
        plt.ylabel('Concordance moyenne')
        #plt.title('Evolution de la concordance moyenne en fonction du rapport b/a dans [0,1] avec le degré moyen = '+str(deg_moy))
        plt.savefig('evol_random_mean_concordance.png', bbox_inches='tight')
        plt.clf()

        # pour avoir une idée de l'avancement du code
        print("100 %")


if __name__ == '__main__':
    print("\n\n***Start***")
    evol_mean_concordance(deg_moy=20,
                          p=np.array([0.5, 0.5]),
                          K=2,
                          t_max=100000,
                          nb_MH=1,
                          nb_G=10,
                          N=1000)
    print("***End***\n\n")
