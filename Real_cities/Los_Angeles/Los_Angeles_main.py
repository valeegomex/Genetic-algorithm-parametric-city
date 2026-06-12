
from sidermit.publictransportsystem import TransportMode, passenger
from AlgoritmoGenetico.Operadores.divisor import Divisor_umbral, Divisor_formula, Divisor_intervalo, \
    Divisor_sin_division
from AlgoritmoGenetico.Operadores.evaluador import Evaluador
from AlgoritmoGenetico.Operadores.iterador import Iterador
from AlgoritmoGenetico.Poblacion.poblacion import Poblacion
from AlgoritmoGenetico.algoritmo import Algoritmo_genetico


import logging

if __name__ ==  '__main__':
    # Desactivar logging de sidermit
    logging.getLogger("sidermit").setLevel(logging.WARNING) # WARNING

    # Cantidad de zonas ciudad
    n_zonas= 7
    # Parámetros de la población
    size_poblacion = 200
    densidad_max_edl = 12
    # Parámetros del evaluador
    L, g, P = 11.65, 0.79, 1
    Y, a, alpha, beta = int(4500000/1000), 0.91, 0.0033, 0.287
    G_inner, G_outer = 1, 5.35
    Gi = [G_inner, G_outer, G_inner, G_outer, G_inner, G_inner, G_outer]
    Hi = Gi
    # Parámetros del iterador
    p_elitismo, prob_mutacion, p_crossover = 0.2, 0.1, 0.5
    # Parámetros del divisor
    d1, d2, umbral = 0.018, 0.250, 0.972
    # Parámetros del algoritmo genético
    gen_max = 14

    # Pasajeros y modo de transporte
    pasajero=  passenger.Passenger(va=4, pv=2.74, pw=5.48, pa=0, pt=16, spv=2.74, spw=5.48, spa=0, spt=16)
    tmode=  TransportMode(name='bus', bya=0, co=8.61 , c1=0.15, c2=0, v=20, t=2.5, fmax=80, kmax=160, theta=0.5, tat=0, d=1, fini=10)

    ### ------- Original --------------------

    # # Parámetros de la población
    # poblacion = Poblacion(size=size_poblacion, max_densitiy=densidad_max_edl)
    #
    # # Parámetros del evaluador
    # evaluador = Evaluador(passenger_obj=pasajero, custom_tmode=tmode, L=L, g=g, P=P, Y=Y, a=a, alpha=alpha, beta=beta,
    #                       n_zonas=n_zonas, Gi=Gi, Hi=Hi)
    #
    # # Parámetros del iterador
    # iterador = Iterador(p_elitismo=p_elitismo, prob_mutacion=prob_mutacion, p_crossover=p_crossover)
    #
    # # Parámetros del divisor
    # divisor = Divisor_sin_division(d1=d1, d2=d2, L=L)
    #
    # # Ejecutar algoritmo
    # AG = Algoritmo_genetico(n_zonas=n_zonas, poblacion=poblacion, gen_max=gen_max,
    #                          evaluador=evaluador, iterador=iterador, divisor=divisor, id='103', n_procesos=12,
    #                         name='LosAngelesPoblacionInicial1')

    # --------- Umbral ---------------

    # Parámetros de la población
    poblacion = Poblacion(size=size_poblacion, max_densitiy=densidad_max_edl)

    # Parámetros del evaluador
    evaluador = Evaluador(passenger_obj=pasajero, custom_tmode=tmode, L=L, g=g, P=P, Y=Y, a=a, alpha=alpha, beta=beta,
                          n_zonas=n_zonas, Gi=Gi, Hi=Hi)

    # Parámetros del iterador
    iterador = Iterador(p_elitismo=p_elitismo, prob_mutacion=prob_mutacion, p_crossover=p_crossover)

    # Parámetros del divisor
    divisor = Divisor_umbral(d1=d1, d2=d2, L=L, umbral=umbral)

    # Ejecutar algoritmo
    AG = Algoritmo_genetico(n_zonas=n_zonas, poblacion=poblacion, gen_max=12,
                            evaluador=evaluador, iterador=iterador, divisor=divisor, id='108', n_procesos=12,
                            name='checkpoint')

    # --------- Formula ---------------

    # # Parámetros de la población
    # poblacion = Poblacion(size=size_poblacion, max_densitiy=densidad_max_edl)
    #
    # # Parámetros del evaluador
    # evaluador = Evaluador(passenger_obj=pasajero, custom_tmode=tmode, L=L, g=g, P=P, Y=Y, a=a, alpha=alpha, beta=beta,
    #                       n_zonas=n_zonas, Gi=Gi, Hi=Hi)
    #
    # # Parámetros del iterador
    # iterador = Iterador(p_elitismo=p_elitismo, prob_mutacion=prob_mutacion, p_crossover=p_crossover)
    #
    # # Parámetros del divisor
    # divisor = Divisor_formula(d1=d1, d2=d2, L=L)
    #
    # # Ejecutar algoritmo
    # AG = Algoritmo_genetico(n_zonas=n_zonas, poblacion=poblacion, gen_max=gen_max,
    #                         evaluador=evaluador, iterador=iterador, divisor=divisor, id='105', n_procesos=12,
    #                         name='LosAngelesPoblacionInicial1')

