
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
    logging.getLogger("sidermit").setLevel(logging.WARNING)

    # Cantidad de zonas ciudad
    n_zonas= 6

    # Pasajeros y modo de transporte
    pasajero=  passenger.Passenger(va=4, pv=2.74, pw=5.48, pa=0, pt=16, spv=2.74, spw=5.48, spa=0, spt=16)
    tmode=  TransportMode(name='bus', bya=0, co=8.61 , c1=0.15, c2=0, v=20, t=2.5, fmax=150, kmax=160, theta=0.5, tat=0, d=1, fini=15)

    # Parámetros de la población
    size_poblacion= 200
    densidad_max_edl= 12
    poblacion = Poblacion(size=size_poblacion, max_densitiy=densidad_max_edl)

    # Parámetros del evaluador
    L, g, P = 10, 1.8, 1
    Y, a, alpha, beta = 3750, 0.8, 0.08, 0.46
    evaluador = Evaluador(passenger_obj=pasajero, custom_tmode=tmode, L=L, g=g, P=P, Y=Y, a=a, alpha=alpha, beta=beta,
                          n_zonas=n_zonas)

    # Parámetros del iterador
    p_elitismo, prob_mutacion, p_crossover = 0.2, 0.1, 0.5
    iterador = Iterador(p_elitismo=p_elitismo, prob_mutacion=prob_mutacion, p_crossover=p_crossover)

    # Parámetros del divisor
    d1, d2, umbral = 0.026, 0.29, 0.809
    # d1, d2, umbral = 0.022, 0.244, 0.723
    # divisor = Divisor_formula(d1=d1, d2=d2, L=L)
    # divisor = Divisor_umbral(d1=d1, d2=d2, L=L, umbral=umbral)
    # divisor = Divisor_intervalo(d1=d1, d2=d2, L=L, umbral=umbral, tolerancia = 0.2)
    divisor = Divisor_sin_division(d1=d1, d2=d2, L=L)

    # Parámetros del algoritmo genético
    gen_max=14

    # Ejecutar algoritmo
    AG = Algoritmo_genetico(n_zonas= n_zonas, poblacion=poblacion, gen_max=gen_max,
                             evaluador=evaluador, iterador=iterador, divisor=divisor, id='10', n_procesos=8,
                            name='zonas6size200SoloFactible')

    # 'zonas6size200SoloFactible'
