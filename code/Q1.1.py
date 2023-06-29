import numpy as np
from matplotlib import pyplot as plt


# Initialisation de Q, la matrice de transition et x, le domaine des états
def chaine_markov():
    Q = np.array([[0, 0.1, 0.1, 0.8], [1, 0, 0, 0], [0.6, 0, 0.1, 0.3],
                  [0.4, 0.1, 0.5, 0]])
    x = np.array([1, 2, 3, 4])
    return Q, x


# P(Xt = x), où x = 1,2,3,4, en supposant que le premier état est choisi au hasard
def P_x_random(Q, x, n):
    P_Xn = np.array([1 / len(x) for i in range(len(x))
                     ])  # vecteur de base [1/4, 1/4, 1/4, 1/4]

    if n == 0:
        return P_Xn
    else:
        P_Xn = np.matmul(P_Xn, Qt(Q, n))
        return P_Xn  # vecteur où P_Xn[i-1] = P(X_t = i)


# P(Xt = x), où x = 1,2,3,4, en supposant que l’état initial est toujours 3
def P_x_3(Q, x, n):
    P_Xn = np.array([0, 0, 1, 0])  #vecteur de base

    if n == 0:
        return P_Xn
    else:
        P_Xn = np.matmul(P_Xn, Qt(Q, n))
        return P_Xn  # vecteur où P_Xn[i-1] = P(X_t = i)


# t-ième puissance de la matrice de transition, avec t>0
def Qt(Q, t):
    Result = Q

    for i in range(1, t):  # si t=1 on ne rentre pas dans la boucle --> Q^1 = Q
        Result = np.matmul(Result, Q)

    return Result


# Représentation de l’évolution des deux premières grandeurs (P_x_random et P_x_3) sur un graphe avec une courbe par état
def representation(Q, x, n):
    evol_unif_1 = np.array([])
    evol_unif_2 = np.array([])
    evol_unif_3 = np.array([])
    evol_unif_4 = np.array([])
    evol_trois_1 = np.array([])
    evol_trois_2 = np.array([])
    evol_trois_3 = np.array([])
    evol_trois_4 = np.array([])
    time = list(range(1, n + 1))

    for i in range(0, n):
        Pr = P_x_random(Q, x, i)
        evol_unif_1 = np.append(evol_unif_1, Pr[0])
        evol_unif_2 = np.append(evol_unif_2, Pr[1])
        evol_unif_3 = np.append(evol_unif_3, Pr[2])
        evol_unif_4 = np.append(evol_unif_4, Pr[3])
        P3 = P_x_3(Q, x, i)
        evol_trois_1 = np.append(evol_trois_1, P3[0])
        evol_trois_2 = np.append(evol_trois_2, P3[1])
        evol_trois_3 = np.append(evol_trois_3, P3[2])
        evol_trois_4 = np.append(evol_trois_4, P3[3])

    plt.plot(time, evol_unif_1, marker='o', label="x=1")
    plt.plot(time, evol_unif_2, marker='o', label="x=2")
    plt.plot(time, evol_unif_3, marker='o', label="x=3")
    plt.plot(time, evol_unif_4, marker='o', label="x=4")
    plt.xlabel('Incréments de temps')
    plt.ylabel('P(Xt = x)')
    #plt.title('Evolution de la première grandeur')
    plt.legend()
    plt.savefig('evol_unif.png', bbox_inches='tight')
    plt.clf()

    plt.plot(time, evol_trois_1, marker='o', label="x=1")
    plt.plot(time, evol_trois_2, marker='o', label="x=2")
    plt.plot(time, evol_trois_3, marker='o', label="x=3")
    plt.plot(time, evol_trois_4, marker='o', label="x=4")
    plt.xlabel('Incréments de temps')
    plt.ylabel('P(Xt = x)')
    #plt.title('Evolution de la deuxième grandeur')
    plt.legend()
    plt.savefig('evol_trois.png', bbox_inches='tight')
    plt.clf()


# Distribution stationnaire pi_inf de la chaîne de Markov définie par [pi_inf]j = lim t→inf P(Xt = j)
def distrib_pi_inf(Q, x):
    Pi_t = np.array([])
    for i in x:
        Pi_t = np.append(Pi_t, P_x_random(Q, x, 0)[i - 1])  # initialisé à Pi_0
    Pi_inf = np.matmul(Pi_t, Q)
    while (Pi_inf != Pi_t).all():
        Pi_t = Pi_inf
        Pi_inf = np.matmul(Pi_t, Q)

    # plot
    x_axis = list(range(1, 5))
    x = ['x=1', 'x=2', 'x=3', 'x=4']
    plt.bar(x_axis, Pi_inf)
    plt.xlabel('Etats')
    plt.ylabel('π∞(i)')
    #plt.title('Distribution stationnaire π∞ de la chaîne de Markov')
    plt.xticks(x_axis, x)
    plt.savefig('pi_inf.png', bbox_inches='tight')
    plt.clf()

    return Pi_inf


# Distribution stationnaire pour un temps n donné
def distrib_pi_inf_bis(Q, x, value, n):
    Pi_t = np.array([])
    for i in x:
        Pi_t = np.append(Pi_t, P_x_random(Q, x, 0)[i - 1])  # initialisé à Pi_0
    Pi_t = np.matmul(Pi_t, Qt(Q, n))

    ind = np.where(x == value)

    return Pi_t[ind[0][0]]


# Calcul du noeud suivant de la chaîne de Markov
def realisation_aleatoire_chaine_markov_next(Q, x, ind_now):
    Prob_next = np.array([])
    for i in range(0, x.size):
        Prob_next = np.append(Prob_next, Q[ind_now][i])
    rand = np.array(np.random.multinomial(1, Prob_next))
    ind_next = np.where(rand == 1)
    return ind_next[0][0]


# Réalisation aléatoire de longueur T de la chaîne de Markov
def realisation_aleatoire_chaine_markov(Q, x, T):
    ind = np.random.randint(x.size - 1)
    chaine = np.array([x[ind]])
    for i in range(0, T):
        ind = realisation_aleatoire_chaine_markov_next(Q, x, ind)
        chaine = np.append(chaine, x[ind])
    return chaine


# Calcul, pour chaque état, du nombre de fois qu’il apparaît dans la réalisation divisé par la longueur de la réalisation (taux d'occurrence)
def realisation_aleatoire_chaine_markov_prop_etats(chaine):
    x1, x2, x3, x4 = 0, 0, 0, 0
    T = len(chaine)
    for i in range(0, T):
        if chaine[i] == 1:
            x1 += 1 / T
        elif chaine[i] == 2:
            x2 += 1 / T
        elif chaine[i] == 3:
            x3 += 1 / T
        else:
            x4 += 1 / T

    return x1, x2, x3, x4


# Représentation de l’évolution de la proportion de présence de chaque état dans la chaine de markov lorsque T croît
def evol_states(Q, x, T_max):
    evol_x1 = np.array([])
    evol_x2 = np.array([])
    evol_x3 = np.array([])
    evol_x4 = np.array([])
    for i in range(0, T_max):
        chaine = realisation_aleatoire_chaine_markov(Q, x, T_max)
        x1, x2, x3, x4 = realisation_aleatoire_chaine_markov_prop_etats(chaine)
        evol_x1 = np.append(evol_x1, x1)
        evol_x2 = np.append(evol_x2, x2)
        evol_x3 = np.append(evol_x3, x3)
        evol_x4 = np.append(evol_x4, x4)

    # proportions d'états
    evol_T = list(range(0, T_max))
    plt.plot(evol_T, evol_x1, label="x=1")
    plt.plot(evol_T, evol_x2, label="x=2")
    plt.plot(evol_T, evol_x3, label="x=3")
    plt.plot(evol_T, evol_x4, label="x=4")
    plt.xlabel('Longueur de la réalisation')
    plt.ylabel('Proportions des états dans la réalisation')
    #plt.title('Evolution de la proportions des états dans la réalisation')
    plt.legend()
    plt.savefig('evol_states.png', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    Q, x = chaine_markov()

    distrib_pi_inf(Q, x)
    representation(Q, x, 10)
    evol_states(Q, x, 300)
