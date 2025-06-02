import random
import math
import matplotlib.pyplot as plt
import numpy as np

def f(x, wybor):
    """
    Funkcja celu do optymalizacji.

    Parametry:
    - x: punkt, w którym obliczana jest wartość funkcji.
    - wybor: wybór funkcji ('A' lub 'B').

    Zwraca:
    - y: wartość funkcji w punkcie x.
    """
    if wybor == "A":  # Funkcja z rozdziału 3, przykład 1
        if -105 <= x <= -95:
            y = 10 - 2 * abs(x + 100)
        elif 95 <= x <= 105:
            y = 11 - 2.2 * abs(x - 100)
        else:
            y = 0
        return y
    elif wybor == "B":  # Funkcja z rozdziału 4, przykład 4
        y = x * math.sin(10 * math.pi * x) + 1
        return y
    else:
        print("Podano nieprawidłową wartość.")
        return None



def simulatedAnnealing(T, M, k, alfT, x1, x2, wybor):
    """
    Implementacja algorytmu symulowanego wyżarzania.

    Parametry:
    - T: początkowa temperatura.
    - M: liczba iteracji.
    - k: stała Boltzmanna.
    - alfT: współczynnik zmniejszania temperatury.
    - x1, x2: zakres poszukiwań.
    - wybor: wybór funkcji ('A' lub 'B').

    Zwraca:
    - s_best: najlepsze znalezione rozwiązanie.
    - f(s_best): wartość funkcji w najlepszym punkcie.
    - steps: lista kroków (punktów odwiedzonych przez algorytm).
    """
    s = random.uniform(x1, x2)
    s_best = s  
    steps = [(s, f(s, wybor))]
    
    for i in range(M):
        fs = f(s, wybor)

        sprim = s + random.uniform(-2*T, 2*T) 
        sprim = max(min(sprim, x2), x1)
        
        fsprim = f(sprim, wybor)
        delta = fsprim - fs
        if delta > 0:
            s = sprim
        else:
            z = random.uniform(0, 1)
            if z < math.exp(delta / (T * k)):                          
                s = sprim
        if f(s, wybor) > f(s_best, wybor):
            s_best = s

        steps.append((s, f(s, wybor)))  

        T *= alfT
    return s_best, f(s_best, wybor), steps



def plot_results(x_vals, y_vals, steps, title):
    """
    Funkcja do tworzenia wykresu funkcji i etapów poszukiwania.

    Parametry:
    - x_vals, y_vals: punkty do wykresu funkcji.
    - steps: lista kroków (punktów odwiedzonych przez algorytm).
    - title: tytuł wykresu.
    """
    # Tworzenie wykresu funkcji
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(x)', color='blue')

    # Konwertowanie listy kroków na tablice NumPy
    steps = np.array(steps)
    steps_x = steps[:, 0]
    steps_y = steps[:, 1]

    # Rysowanie etapów poszukiwania z cieńszą linią
    plt.plot(steps_x, steps_y, 'o-', color='red', label='Etapy poszukiwania', markersize=3, linewidth=0.5)

    # Podkreślenie najlepszego znalezionego rozwiązania
    best_idx = np.argmax(steps_y)
    plt.plot(steps_x[best_idx], steps_y[best_idx], 'go', label='Najlepsze rozwiązanie', markersize=8)

    # Ustawienia wykresu z większymi etykietami
    plt.title(title, fontsize=20)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('f(x)', fontsize=20)
    plt.legend(fontsize=17)
    plt.grid(True)

    # Zwiększenie rozmiaru oznaczeń osi (tick labels)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.show()


# Parametry dla funkcji z rozdziału 3, przykład 1
T_3 = 500
alfT_3 = 0.999
k_3 = 0.1
M_3 = 3000
x1_3 = -150
x2_3 = 150

for i in range(1, 6):
    s_best, f_s_best, steps = simulatedAnnealing(T_3, M_3, k_3, alfT_3, x1_3, x2_3, "A")
    print(f"Wynik dla funkcji z rozdziału 3, próba {i}: x = {s_best:.4f}, f(x) = {f_s_best:.4f}")

    x_vals = np.linspace(x1_3, x2_3, 1000)
    y_vals = [f(x, "A") for x in x_vals]

    plot_results(x_vals, y_vals, steps, f'Funkcja z rozdziału 3, próba {i}')


# Parametry dla funkcji z rozdziału 4, przykład 4
T_4 = 5
alfT_4 = 0.997
k_4 = 0.1
M_4 = 1200
x1_4 = -1
x2_4 = 2

for i in range(1, 6):
    s_best, f_s_best, steps = simulatedAnnealing(T_4, M_4, k_4, alfT_4, x1_4, x2_4, "B")
    print(f"Wynik dla funkcji z rozdziału 4, próba {i}: x = {s_best:.4f}, f(x) = {f_s_best:.4f}")

    x_vals = np.linspace(x1_4, x2_4, 1000)
    y_vals = [f(x, "B") for x in x_vals]

    plot_results(x_vals, y_vals, steps, f'Funkcja z rozdziału 4, próba {i}')
