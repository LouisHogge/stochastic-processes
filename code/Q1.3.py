import numpy as np
import random
from scipy.special import binom
from scipy.stats import binom as binom_plot
from matplotlib import pyplot as plt
import pandas as pd


# Distribution binomiale
def P_bin(k, p, K):
    C = binom(K, k)
    return C * p**k * (1 - p)**(K - k)


# Distribution de proposition donnée
def q_y_x(r, K, y, x):
    if (x == 0 and y == 0) or (0 < x <= K and y == x - 1):
        return r
    elif (x == K and y == K) or (0 <= x < K and y == x + 1):
        return 1 - r
    else:
        return 0


# Calcul y en fonction de la distribution de proposition donnée q_y_x
def y_next(x, r, K):
    rand = random.uniform(0, 1)
    if (x == 0) or (0 < x <= K):
        if rand < r:
            if x == 0:
                y = x
            else:
                y = x - 1
        else:
            if x == K:
                y = x
            else:
                y = x + 1

    elif (x == K) or (0 <= x < K):
        if rand >= r:
            if x == K:
                y = x
            else:
                y = x + 1
        else:
            if x == 0:
                y = x
            else:
                y = x - 1
    return y


# Algorithme de Metropolis-Hastings
def MH(x_0, r, p, K, t_max):
    t = 1
    x = np.array([x_0])
    y = np.array([x_0])
    moyenne = np.array([x_0])
    variance = np.array([0])

    while (t < t_max):
        y = np.append(y, y_next(x[t - 1], r, K))
        alpha = min(
            1,
            P_bin(y[t], p, K) / P_bin(x[t - 1], p, K) *
            q_y_x(r, K, x[t - 1], y[t]) / q_y_x(r, K, y[t], x[t - 1]))

        u = random.uniform(0, 1)

        if u < alpha:
            x = np.append(x, y[t])
        else:
            x = np.append(x, x[t - 1])

        t = t + 1

        moyenne = np.append(moyenne, x.mean())
        variance = np.append(variance, x.var())

    return x, moyenne, variance


# Génération d'une réalisation suffisamment longue de la chaine avec r=0.1 et r=0.5. Ensuite, représentations de la convergence des moyennes et de la convergence des variances des valeurs ainsi générées. Enfin, création d'histogrammes des fréquences d’apparition de chaque valeur dans la réalisation avec r=0.1 et r=0.5.
def etude_convergenc(x_0, p, K, t_max):
    x_05, moyenne_05, variance_05 = MH(0, 0.5, p, K, t_max)
    x_01, moyenne_01, variance_01 = MH(0, 0.1, p, K, t_max)

    time = list(range(0, t_max))

    # plot mean
    plt.figure()
    plt.plot(time, moyenne_01, label="r=0.1")
    plt.plot(time, moyenne_05, label="r=0.5")
    plt.xlabel('Longueur de la réalisation')
    plt.ylabel('Moyennes')
    #plt.title('Evolution de la moyenne des valeurs générées en fonction de la longueur de la réalisation')
    plt.legend()
    plt.savefig('mean.png', bbox_inches='tight')
    plt.clf

    # plot variance
    plt.figure()
    plt.plot(time, variance_01, label="r=0.1")
    plt.plot(time, variance_05, label="r=0.5")
    plt.xlabel('Longueur de la réalisation')
    plt.ylabel('Variances')
    #plt.title('Evolution de la variance des valeurs générées en fonction de la longueur de la réalisation')
    plt.legend()
    plt.savefig('var.png', bbox_inches='tight')
    plt.clf

    # plot histogramme
    increments = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    localisation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    k, p = 10, 0.3
    etats = np.arange(binom_plot.ppf(0.01, k, p), binom_plot.ppf(0.99, k, p))

    # r=0.1
    pd.DataFrame(x_01).hist(grid=False,
                            ec='black',
                            density=True,
                            bins=np.arange(-0.5, 10.5, 1.0))

    plt.plot(etats, binom_plot.pmf(etats, k, p), 'ro', ms=8)
    plt.vlines(etats,
               0,
               binom_plot.pmf(etats, k, p),
               colors='r',
               lw=5,
               alpha=0.5,
               label='_nolegend_')
    #plt.title('Histogramme des fréquences d’apparition de chaque valeur dans la réalisation avec r=0.1')
    plt.title('')
    plt.xlabel('Etats')
    plt.ylabel('Densité')
    plt.xticks(ticks=localisation, labels=increments)
    plt.xlim([-1.0, 11.0])
    plt.gca().legend(('Distribution théorique binomiale',
                      'Fréquences d\'apparitions des etats'))
    plt.savefig('hist_r_01.png', bbox_inches='tight')
    plt.clf()

    # r=0.5
    pd.DataFrame(x_05).hist(grid=False,
                            ec='black',
                            density=True,
                            bins=np.arange(-0.5, 10.5, 1.0))

    plt.plot(etats, binom_plot.pmf(etats, k, p), 'ro', ms=8)
    plt.vlines(etats,
               0,
               binom_plot.pmf(etats, k, p),
               colors='r',
               lw=5,
               alpha=0.5,
               label='_nolegend_')
    #plt.title('Histogramme des fréquences d’apparition de chaque valeur dans la réalisation avec r=0.5')
    plt.title('')
    plt.xlabel('Etats')
    plt.ylabel('Densité')
    plt.xticks(ticks=localisation, labels=increments)
    plt.xlim([-1.0, 11.0])
    plt.gca().legend(('Distribution théorique binomiale',
                      'Fréquences d\'apparitions des etats'))
    plt.savefig('hist_r_05.png', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    etude_convergenc(x_0=0, p=0.3, K=10, t_max=15000)
