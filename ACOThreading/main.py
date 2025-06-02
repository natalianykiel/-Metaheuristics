from utils import Utils
from multithreading import Multithreading
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = './solomon-100/r101.txt'
    ants_num = 10
    beta = 2
    q0 = 0.1
    show_figure = True

    utils = Utils(file_path)
    macs = Multithreading(utils, ants_num=ants_num, beta=beta, q0=q0)
    macs.run()

    # best_distances = []
    # best_vehicles = []

    # # Uruchamiamy 5 razy
    # for i in range(5):
    #     distance, vehicles = macs.run()
    #     best_distances.append(distance)
    #     best_vehicles.append(vehicles)

    # # Tworzymy dwa podwykresy
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

    # # Wykres 1 - Best Path Distance
    # ax1.plot(range(1, 6), best_distances, marker='o', color='blue')
    # ax1.set_title('Best Path Distance.')
    # ax1.set_xlabel('Numer uruchomienia')
    # ax1.set_ylabel('Wartość')

    # # Wykres 2 - Best Vehicle Num
    # ax2.plot(range(1, 6), best_vehicles, marker='o', color='orange')
    # ax2.set_title('Best Vehicle Num')
    # ax2.set_xlabel('Numer uruchomienia')
    # ax2.set_ylabel('Liczba pojazdów')

    # plt.tight_layout()
    # plt.show()

# import matplotlib.pyplot as plt
# from utils import Utils
# from multithreading import Multithreading

# import os
# import matplotlib.pyplot as plt
# from utils import Utils
# from multithreading import Multithreading

# if __name__ == '__main__':
#     # Folder, w którym zapiszemy wykresy
#     folder_name = 'plots'
#     os.makedirs(folder_name, exist_ok=True)  # Utwórz folder, jeśli nie istnieje

#     file_path = './solomon-100/r101.txt'
#     utils_global = Utils(file_path)

#     # =========================
#     #   EKSPERYMENT 1: ants_num
#     # =========================
#     ants_num_values = [30, 100]
#     beta_fixed = 2
#     q0_fixed = 0.1

#     print("\n=== EKSPERYMENT 1: Zmiana ants_num ===")
#     for ants in ants_num_values:
#         print(f"Parametry: ants_num={ants}, beta={beta_fixed}, q0={q0_fixed}")
#         macs = Multithreading(utils_global, ants_num=ants, beta=beta_fixed, q0=q0_fixed)

#         best_distances = []
#         best_vehicles = []

#         # Uruchamiamy 5 razy
#         for _ in range(5):
#             distance, vehicles = macs.run()
#             best_distances.append(distance)
#             best_vehicles.append(vehicles)

#         # Rysujemy wykres
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

#         ax1.plot(range(1, 6), best_distances, marker='o', color='blue')
#         ax1.set_title('Best Path Distance')
#         ax1.set_xlabel('Numer uruchomienia')
#         ax1.set_ylabel('Wartość')
#         ax1.set_ylim(1400, 3000)  # przykładowy zakres

#         ax2.plot(range(1, 6), best_vehicles, marker='o', color='orange')
#         ax2.set_title('Best Vehicle Num')
#         ax2.set_xlabel('Numer uruchomienia')
#         ax2.set_ylabel('Liczba pojazdów')

#         plt.suptitle(f'(Eksperyment 1) ants_num={ants}, beta={beta_fixed}, q0={q0_fixed}')
#         plt.tight_layout()

#         # Zapis do pliku
#         save_path = os.path.join(folder_name, f"exp1_ants{ants}.png")
#         plt.savefig(save_path, dpi=300)
#         plt.close()
#         # Jeśli chcesz dodatkowo wyświetlić wykres:
#         # plt.show()

#     # =========================
#     #   EKSPERYMENT 2: beta
#     # =========================
#     beta_values = [0.001, 2, 5, 8, 10]
#     ants_num_fixed = 10
#     q0_fixed = 0.1

#     print("\n=== EKSPERYMENT 2: Zmiana beta ===")
#     for b in beta_values:
#         print(f"Parametry: ants_num={ants_num_fixed}, beta={b}, q0={q0_fixed}")
#         macs = Multithreading(utils_global, ants_num=ants_num_fixed, beta=b, q0=q0_fixed)

#         best_distances = []
#         best_vehicles = []

#         for _ in range(5):
#             distance, vehicles = macs.run()
#             best_distances.append(distance)
#             best_vehicles.append(vehicles)

#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

#         ax1.plot(range(1, 6), best_distances, marker='o', color='blue')
#         ax1.set_title('Best Path Distance')
#         ax1.set_xlabel('Numer uruchomienia')
#         ax1.set_ylabel('Wartość')
#         ax1.set_ylim(1200, 3000)  # inny zakres przykładowy

#         ax2.plot(range(1, 6), best_vehicles, marker='o', color='orange')
#         ax2.set_title('Best Vehicle Num')
#         ax2.set_xlabel('Numer uruchomienia')
#         ax2.set_ylabel('Liczba pojazdów')

#         plt.suptitle(f'(Eksperyment 2) ants_num={ants_num_fixed}, beta={b}, q0={q0_fixed}')
#         plt.tight_layout()

#         # Zapis do pliku
#         save_path = os.path.join(folder_name, f"exp2_beta{b}.png")
#         plt.savefig(save_path, dpi=300)
#         plt.close()
#         # plt.show()

#     # =========================
#     #   EKSPERYMENT 3: q0
#     # =========================
#     q0_values = [0.05, 0.1, 0.15, 0.2, 1]
#     ants_num_fixed = 10
#     beta_fixed = 2

#     print("\n=== EKSPERYMENT 3: Zmiana q0 ===")
#     for q in q0_values:
#         print(f"Parametry: ants_num={ants_num_fixed}, beta={beta_fixed}, q0={q}")
#         macs = Multithreading(utils_global, ants_num=ants_num_fixed, beta=beta_fixed, q0=q)

#         best_distances = []
#         best_vehicles = []

#         for _ in range(5):
#             distance, vehicles = macs.run()
#             best_distances.append(distance)
#             best_vehicles.append(vehicles)

#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

#         ax1.plot(range(1, 6), best_distances, marker='o', color='blue')
#         ax1.set_title('Best Path Distance')
#         ax1.set_xlabel('Numer uruchomienia')
#         ax1.set_ylabel('Wartość')
#         ax1.set_ylim(1200, 3000)

#         ax2.plot(range(1, 6), best_vehicles, marker='o', color='orange')
#         ax2.set_title('Best Vehicle Num')
#         ax2.set_xlabel('Numer uruchomienia')
#         ax2.set_ylabel('Liczba pojazdów')

#         plt.suptitle(f'(Eksperyment 3) ants_num={ants_num_fixed}, beta={beta_fixed}, q0={q}')
#         plt.tight_layout()

#         # Zapis do pliku
#         save_path = os.path.join(folder_name, f"exp3_q0_{q}.png")
#         plt.savefig(save_path, dpi=300)
#         plt.close()
#         # plt.show()
