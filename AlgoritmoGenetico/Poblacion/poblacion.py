import pickle
import numpy as np

from AlgoritmoGenetico.BaseDatos.BD import BD
from AlgoritmoGenetico.Poblacion.individuo import Individuo


class Poblacion:
    def __init__(self, size: int, max_densitiy: int):
        """
        :param size: Tamaño de la población.
        :param max_densitiy: Máxima cantidad de líneas por EDL.
        """
        self.size = size
        self.max_density = max_densitiy
        self.population = []
        self.best = []
        self.gen = 0

    def get_population(self) -> list[Individuo]:
        return self.population

    def set_population(self, new_population):
        self.population = new_population

    def get_gen(self):
        return self.gen

    def set_gen(self, new_gen: int):
        self.gen = new_gen

    def build_random(self, bd: BD):
        """
        Construye la población aleatoriamente, sin importar su factibilidad.
        :param bd: Base de datos de líneas.
        :return:
        """
        for i in range(self.size):
            density = np.random.randint(bd.get_n(), self.max_density)
            id_lineas = np.random.choice(bd.get_size(), size=density, replace=False).tolist()
            individuo = Individuo(id_lineas)
            self.population.append(individuo)
        pass

    def build_from_file(self, name: str):
        """
        Construye la población a partir de los id_listas que lee en el archivo.
        :param name:
        :return:
        """
        with open("{}.pickle".format(name), "rb") as f:
            obj = pickle.load(f)
        for id_lineas in obj:
            individuo = Individuo(id_lineas)
            self.population.append(individuo)
        pass

    def ind_nuevo_azar(self, bd: BD) -> Individuo:
        """
        Crea un nuevo individuo al azar.
        :param bd: Base de datos de líneas.
        :return:
        """
        density = np.random.randint(bd.get_n(), self.max_density)
        id_lineas = np.random.choice(bd.get_size(), size=density, replace=False).tolist()
        individuo = Individuo(id_lineas)

        return individuo

    def report(self, name: str) -> None:
        """
        Guarda un reporte en el archivo txt.
        :param name: str. Nombre del archivo.
        :return: None
        """
        f = open(f'{name}.txt', 'w')
        f.write(f'Generacion {self.get_gen()}:')
        f.write("\n")
        for ind in self.get_population():
            f.write(f'EDL {ind.get_id_lineas()} MVRC {ind.get_MVRC()}')
            f.write("\n")
        f.write("\n")
        f.close()
        pass

    def save_edl_population(self, name: str):
        """
        Guardar una lista con los id_lineas de los individuos de la población.
        :return:
        """

        obj = []
        for ind in self.get_population():
            obj.append(ind.get_id_lineas())
        with open("{}.pickle".format(name), "wb") as f:
            pickle.dump(obj, f)
        pass