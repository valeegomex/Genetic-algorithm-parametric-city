from sidermit.city import Graph
from sidermit.publictransportsystem import TransportMode, passenger
import pandas as pd

from AlgoritmoGenetico.Operadores.calibrador import Calibrador

n, L, g,  P = 4, 10, 1.8, 1
Y = n*1000
tmode = TransportMode(name='bus', bya=0, co=8.61 , c1=0.15, c2=0, v=20, t=2.5, fmax=150, kmax=160, theta=0.5, tat=0, d=1, fini=12)
passenger_obj = passenger.Passenger(va=4, pv=2.74, pw=5.48, pa=0, pt=16, spv=2.74, spw=5.48, spa=0, spt=16)
graph_obj = Graph.build_from_parameters(n=n, l=L, g=g, p=P)

calibrador = Calibrador(n=n, graph_obj=graph_obj, tmode=tmode, passenger_obj=passenger_obj, Y=Y, L=L, build=False)

# frontera = calibrador.obtener_frontera()
# error = calibrador.error_prediccion(d1=0.8, d2=0.1, umbral=0.54, frontera=frontera)

frontera = pd.read_csv('FronteraN4.csv')
edl_completa = pd.read_csv('EDL_completaN4.csv')
edl_dividida = pd.read_csv('EDL_divididaN4.csv')

calibrador.set_edl_compelta(edl_completa)
calibrador.set_edl_dividida(edl_dividida)

# calibrador.error_prediccion(0.8, 0.1, 0.56, frontera)

terminos_optimos = calibrador.calibrar_frontera(frontera)
print(terminos_optimos)

