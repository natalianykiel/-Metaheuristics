import numpy as np
import random
from utils import Utils, Data
from ant import Ant
from threading import Thread, Event
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import copy
import time
from multiprocessing import Process
import matplotlib.pyplot as plt
import os


class Multithreading:
    def __init__(self, utils: Utils, ants_num=10, beta=1, q0=0.1):
        super()
        self.utils = utils
        self.ants_num = ants_num
        self.max_load = utils.vehicle_capacity
        self.beta = beta
        self.q0 = q0
        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None

    def run(self, file_to_write_path=None):
        process_thread = Process(target=self._multiple_colony(file_to_write_path))
        process_thread.start()
        process_thread.join()
        return self.best_path_distance, self.best_vehicle_num

    def _multiple_colony(self, file_to_write_path=None):
        if file_to_write_path is not None:
            file_to_write = open(file_to_write_path, 'w')
        else:
            file_to_write = None

        start_time_total = time.time()

        global_path_to_acs_time = Queue()
        global_path_to_acs_vehicle = Queue()

        path_found_queue = Queue()

        # ALGORYTM NAJBLIŻSZEGO SĄSIADA – wstępne rozwiązanie
        self.best_path, self.best_path_distance, self.best_vehicle_num = self.utils.nearest_neighbor_heuristic()

        while True:
            print('[]: nowa iteracja')
            start_time = time.time()

            global_path_to_acs_vehicle.put(Data(self.best_path, self.best_path_distance))
            global_path_to_acs_time.put(Data(self.best_path, self.best_path_distance))

            stop_event = Event()

            # --- TWORZENIE WĄTKÓW ---
            utils_for_acs_vehicle = self.utils.copy(self.utils.init_pheromone_val)
            acs_vehicle_thread = Thread(target=Multithreading.thread_vehicle,
                                        args=(utils_for_acs_vehicle, self.best_vehicle_num-1, self.ants_num, self.q0,
                                              self.beta, global_path_to_acs_vehicle, path_found_queue, stop_event))

            utils_for_acs_time = self.utils.copy(self.utils.init_pheromone_val)
            acs_time_thread = Thread(target=Multithreading.thread_time,
                                     args=(utils_for_acs_time, self.best_vehicle_num, self.ants_num, self.q0, self.beta,
                                           global_path_to_acs_time, path_found_queue, stop_event))
            print('[]: uruchamianie acs_vehicle i acs_time')
            acs_vehicle_thread.start()
            acs_time_thread.start()

            # Lokalna kopia najlepszej liczby pojazdów (na starcie iteracji)
            best_vehicle_num = self.best_vehicle_num

            while acs_vehicle_thread.is_alive() and acs_time_thread.is_alive():

                given_time = 10  # WARUNEK STOPU (w sekundach)
                if time.time() - start_time > 60 * given_time:
                    stop_event.set()
                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    self.print_and_write_in_file(file_to_write, 'Czas upłynął: nie znaleziono lepszego rozwiązania '
                                                                'w określonym czasie (%d s)' % given_time)
                    self.print_and_write_in_file(file_to_write, 'Czas działania: %0.3f sekund'
                                                    % (time.time()-start_time_total))
                    self.print_and_write_in_file(file_to_write,
                                                 'Najlepsza znaleziona ścieżka to:')
                    self.print_and_write_in_file(file_to_write, self.best_path)
                    self.print_and_write_in_file(file_to_write,
                                                 'Najlepsza odległość to %f, liczba pojazdów: %d'
                                                 % (self.best_path_distance, self.best_vehicle_num))
                    self.print_and_write_in_file(file_to_write, '*' * 50)

                    self.plot_final_route()  

                    if file_to_write is not None:
                        file_to_write.flush()
                        file_to_write.close()
                    return

                # Jeśli brak nowej ścieżki w kolejce – kontynuuj
                if path_found_queue.empty():
                    continue

                # Pobieramy informację o znalezionej ścieżce (lub ścieżkach)
                path_info = path_found_queue.get()
                print('[]: odebrano informacje o ścieżce')
                found_path, found_path_distance, found_path_used_vehicle_num = path_info.get_path_info()

                # Sprawdzamy kolejne, jeśli jest ich więcej
                while not path_found_queue.empty():
                    path, distance, vehicle_num = path_found_queue.get().get_path_info()
                    # Porównujemy na dwóch kryteriach: odległość i liczba pojazdów
                    if distance < found_path_distance:
                        found_path, found_path_distance, found_path_used_vehicle_num = path, distance, vehicle_num
                    if vehicle_num < found_path_used_vehicle_num:
                        found_path, found_path_distance, found_path_used_vehicle_num = path, distance, vehicle_num

                # --- Aktualizacja najlepszego rozwiązania po kącie odległości ---
                if found_path_distance < self.best_path_distance:
                    start_time = time.time()

                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    self.print_and_write_in_file(
                        file_to_write,
                        '[macs]: odległość znalezionej ścieżki (%f) lepsza niż obecnej najlepszej (%f)'
                        % (found_path_distance, self.best_path_distance)
                    )
                    self.print_and_write_in_file(
                        file_to_write,
                        'Czas działania: %0.3f sekund' % (time.time()-start_time_total)
                    )
                    self.print_and_write_in_file(file_to_write, '*' * 50)

                    if file_to_write is not None:
                        file_to_write.flush()

                    self.best_path = found_path
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_distance = found_path_distance

                    # Powiadamiamy o nowym globalnym optimum
                    global_path_to_acs_vehicle.put(Data(self.best_path, self.best_path_distance))
                    global_path_to_acs_time.put(Data(self.best_path, self.best_path_distance))

                # --- Aktualizacja najlepszego rozwiązania po kącie liczby pojazdów ---
                if found_path_used_vehicle_num < best_vehicle_num:
                    start_time = time.time()
                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    self.print_and_write_in_file(
                        file_to_write,
                        '[macs]: liczba pojazdów w znalezionej ścieżce (%d) lepsza od obecnej najlepszej (%d), '
                        'odległość znalezionej ścieżki to %f'
                        % (found_path_used_vehicle_num, best_vehicle_num, found_path_distance)
                    )
                    self.print_and_write_in_file(
                        file_to_write,
                        'Czas działania: %0.3f sekund' % (time.time() - start_time_total)
                    )
                    self.print_and_write_in_file(file_to_write, '*' * 50)

                    if file_to_write is not None:
                        file_to_write.flush()

                    self.best_path = found_path
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_distance = found_path_distance

                    # Zatrzymujemy wątki, bo mamy globalnego zwycięzcę
                    print('[macs]: wysyłanie informacji o zatrzymaniu do acs_time i acs_vehicle')
                    stop_event.set()
                   

    # =========================
    #        WĄTEK: TIME
    # =========================
    @staticmethod
    def thread_time(new_utils: Utils, vehicle_num: int, ants_num: int, q0: float, beta: int,
                    global_path_queue: Queue, path_found_queue: Queue, stop_event: Event):

        print('[WĄTEK:asc_time]: start, vehicle_num %d' % vehicle_num)

        global_best_path = None
        global_best_distance = None

        # Tworzy pulę wątków mrówek
        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []

        while True:
            print('[WĄTEK:asc_time]: nowa iteracja')

            if stop_event.is_set():
                print('[WĄTEK:asc_time]: zatrzymano')
                return

            # Uruchamiamy mrówki w wątkach
            for k in range(ants_num):
                ant = Ant(new_utils, 0)
                thread = ants_pool.submit(Multithreading.thread_ant,
                                          ant, vehicle_num, True, np.zeros(new_utils.node_num),
                                          q0, beta, stop_event)
                ants_thread.append(thread)
                ants.append(ant)

            for thread in ants_thread:
                thread.result()

            ant_best_travel_distance = None
            ant_best_path = None

            for ant in ants:
                if stop_event.is_set():
                    print('[acs_time]: zatrzymano')
                    return

                # Sprawdzamy, czy otrzymaliśmy globalne info o ścieżce
                if not global_path_queue.empty():
                    info = global_path_queue.get()
                    while not global_path_queue.empty():
                        info = global_path_queue.get()
                    print('[acs_time]: otrzymano globalne informacje o ścieżce')

                    global_best_path, global_best_distance, global_used_vehicle_num = info.get_path_info()

                # Jeżeli mrówka odwiedziła wszystkie węzły i jest lepsza lokalnie
                if ant.index_to_visit_empty() and \
                        (ant_best_travel_distance is None or ant.total_travel_distance < ant_best_travel_distance):
                    ant_best_travel_distance = ant.total_travel_distance
                    ant_best_path = ant.travel_path

            # Aktualizacja feromonów globalna (na podstawie globalnie najlepszej ścieżki)
            if global_best_path is not None and global_best_distance is not None:
                new_utils.global_update_pheromone(global_best_path, global_best_distance)

            # Jeśli lokalna ścieżka (ant_best) jest lepsza od globalnie najlepszej -> zgłaszamy do MACS
            if (ant_best_travel_distance is not None and global_best_distance is not None and
                    ant_best_travel_distance < global_best_distance):
                print('[acs_time]: lokalne wyszukiwanie mrówek znalazło ulepszoną ścieżkę')
                path_found_queue.put(Data(ant_best_path, ant_best_travel_distance))

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    # =========================
    #      WĄTEK: VEHICLE
    # =========================
    @staticmethod
    def thread_vehicle(new_utils: Utils, vehicle_num: int, ants_num: int, q0: float, beta: int,
                       global_path_queue: Queue, path_found_queue: Queue, stop_event: Event):
        print('[WĄTEK:acs_vehicle]: start, vehicle_num %d' % vehicle_num)
        global_best_path = None
        global_best_distance = None

        # Heurystyka najbliższego sąsiada (dla ograniczenia liczby pojazdów)
        current_path, current_path_distance, _ = new_utils.nearest_neighbor_heuristic(max_vehicle_num=vehicle_num)

        current_index_to_visit = list(range(new_utils.node_num))
        for ind in set(current_path):
            if ind in current_index_to_visit:
                current_index_to_visit.remove(ind)

        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        IN = np.zeros(new_utils.node_num)

        while True:
            print('[acs_vehicle]: nowa iteracja')

            if stop_event.is_set():
                print('[acs_vehicle]: zatrzymano')
                return

            # Uruchamiamy mrówki
            for k in range(ants_num):
                ant = Ant(new_utils, 0)
                thread = ants_pool.submit(Multithreading.thread_ant,
                                          ant, vehicle_num, False, IN,
                                          q0, beta, stop_event)
                ants_thread.append(thread)
                ants.append(ant)

            for thread in ants_thread:
                thread.result()

            for ant in ants:
                if stop_event.is_set():
                    print('[acs_vehicle]: zatrzymano')
                    return

                # Zliczamy ile razy dany węzeł został jeszcze nieodwiedzony
                IN[ant.index_to_visit] = IN[ant.index_to_visit] + 1

                # Sprawdzamy, czy mrówka odwiedziła więcej węzłów niż obecna ścieżka "current_path"
                if len(ant.index_to_visit) < len(current_index_to_visit):
                    current_path = copy.deepcopy(ant.travel_path)
                    current_index_to_visit = copy.deepcopy(ant.index_to_visit)
                    current_path_distance = ant.total_travel_distance
                    IN = np.zeros(new_utils.node_num)

                    # Jeśli mrówka rozwiązała całkowicie (wszystkie węzły)
                    if ant.index_to_visit_empty():
                        print('[acs_vehicle]: znaleziono wykonalną ścieżkę, wysyłanie do MACS')
                        path_found_queue.put(Data(ant.travel_path, ant.total_travel_distance))

            # Aktualizacja feromonów globalnie
            new_utils.global_update_pheromone(current_path, current_path_distance)

            # Pobieramy globalną ścieżkę
            if not global_path_queue.empty():
                info = global_path_queue.get()
                while not global_path_queue.empty():
                    info = global_path_queue.get()
                print('[acs_vehicle]: otrzymano globalne informacje o ścieżce')
                global_best_path, global_best_distance, global_used_vehicle_num = info.get_path_info()

            if global_best_path is not None and global_best_distance is not None:
                new_utils.global_update_pheromone(global_best_path, global_best_distance)

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    # =========================
    #   METODY POMOCNICZE
    # =========================
    @staticmethod
    def probability(index_to_visit, transition_prob):
        # Losowanie węzła z prawdopodobieństwem
        N = len(index_to_visit)
        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob / sum_tran_prob

        while True:
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]

    @staticmethod
    def thread_ant(ant: Ant, vehicle_num: int, local_search: bool, IN: np.ndarray,
                   q0: float, beta: int, stop_event: Event):

        remaining_depots = vehicle_num

        while not ant.index_to_visit_empty() and remaining_depots > 0:
            if stop_event.is_set():
                return

            feasible_next_nodes = ant.cal_next_index_meet_constrains()
            if len(feasible_next_nodes) == 0:
                # Brak dostępnych węzłów – jedziemy do depotu
                ant.move_to_next_index(0)
                remaining_depots -= 1
                continue

            # Obliczanie prawdopodobieństw przejścia
            num_options = len(feasible_next_nodes)
            ready_times = np.zeros(num_options)
            due_times = np.zeros(num_options)

            for idx in range(num_options):
                ready_times[idx] = ant.utils.nodes[feasible_next_nodes[idx]].ready_time
                due_times[idx] = ant.utils.nodes[feasible_next_nodes[idx]].due_time

            delivery_estimates = np.maximum(
                ant.vehicle_travel_time + ant.utils.node_dist_mat[ant.current_index][feasible_next_nodes],
                ready_times
            )
            travel_deltas = delivery_estimates - ant.vehicle_travel_time
            weights = travel_deltas * (due_times - ant.vehicle_travel_time)

            weights = np.maximum(1.0, weights - IN[feasible_next_nodes])
            inverse_weights = 1 / weights

            probabilities = ant.utils.pheromone_mat[ant.current_index][feasible_next_nodes] * \
                            np.power(inverse_weights, beta)
            probabilities /= np.sum(probabilities)

            # Q0 – wybór deterministyczny lub probabilistyczny
            if np.random.rand() < q0:
                selected_index = np.argmax(probabilities)
                next_node = feasible_next_nodes[selected_index]
            else:
                next_node = Multithreading.probability(feasible_next_nodes, probabilities)

            # Lokalna aktualizacja feromonu
            ant.utils.local_update_pheromone(ant.current_index, next_node)
            ant.move_to_next_index(next_node)

        # Jeśli wszystkie węzły odwiedzone, wracamy do depotu
        if ant.index_to_visit_empty():
            ant.utils.local_update_pheromone(ant.current_index, 0)
            ant.move_to_next_index(0)

        # Wstawiamy ewentualne nieodwiedzone węzły
        ant.insertion_procedure(stop_event)

        # Lokalny LS jeżeli jest to wątek "time" (local_search=True)
        if local_search and ant.index_to_visit_empty():
            ant.local_search_procedure(stop_event)

    @staticmethod
    def print_and_write_in_file(file_to_write=None, message='default message'):
        if file_to_write is None:
            print(message)
        else:
            print(message)
            file_to_write.write(str(message)+'\n')

    # =========================================
    #      FUNKCJA RYSUJĄCA FINALNĄ TRASĘ
    # =========================================
    def plot_final_route(self):
        if self.best_path is None:
            print("Brak trasy do narysowania.")
            return

        # Dzielenie best_path na osobne trasy: każda zaczyna się i kończy 0
        segments = []
        current_segment = [self.best_path[0]]
        for i in range(1, len(self.best_path)):
            current_segment.append(self.best_path[i])
            if self.best_path[i] == 0 and i < len(self.best_path)-1:
                # zamykamy aktualny segment, rozpoczynamy nowy
                segments.append(current_segment)
                current_segment = [0]
        # Ostatni segment
        if current_segment:
            segments.append(current_segment)

        plt.figure(figsize=(10, 6))
        colors = plt.cm.get_cmap("tab10", len(segments))  # paleta kolorów

        for idx, seg in enumerate(segments):
            x_coords = [self.utils.nodes[n].x for n in seg]
            y_coords = [self.utils.nodes[n].y for n in seg]
            plt.plot(x_coords, y_coords, marker='o', color=colors(idx), label=f'Pojazd {idx+1}')

            # Etykiety punktów (opcjonalnie)
            for (x, y, node_id) in zip(x_coords, y_coords, seg):
                plt.text(x, y, str(node_id), fontsize=9, color='black')

        plt.title("Finalna trasa MACS")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend()
        plt.close()
