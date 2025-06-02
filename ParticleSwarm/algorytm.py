import os
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.lines as mlines


def ackley(x, y, a=20, b=0.2, c=2 * np.pi):
    part1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    part2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    return part1 + part2 + a + np.exp(1)

def rastrigin(x, y):
    return 10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

class Czastka:
    def __init__(self, x, y, inercja, stala_poznawcza, stala_spoleczna, maksymalizacja=False):
        self.x = x
        self.y = y
        self.inercja = inercja
        self.stala_poznawcza = stala_poznawcza
        self.stala_spoleczna = stala_spoleczna
        self.predkosc_x = 0.0
        self.predkosc_y = 0.0
        
        if maksymalizacja:
            self.przystosowanie = -float('inf')
            self.najlepsze_przystosowanie = -float('inf')
        else:
            self.przystosowanie = float('inf')
            self.najlepsze_przystosowanie = float('inf')

        self.najlepsze_x = x
        self.najlepsze_y = y
        self.maksymalizacja = maksymalizacja

    def zaktualizuj_przystosowanie(self, funkcja_przystosowania):
        aktualna = funkcja_przystosowania(self.x, self.y)
        self.przystosowanie = aktualna

        if self.maksymalizacja:
            if aktualna > self.najlepsze_przystosowanie:
                self.najlepsze_przystosowanie = aktualna
                self.najlepsze_x = self.x
                self.najlepsze_y = self.y
        else:
            if aktualna < self.najlepsze_przystosowanie:
                self.najlepsze_przystosowanie = aktualna
                self.najlepsze_x = self.x
                self.najlepsze_y = self.y

    def zaktualizuj_predkosc(self, najlepsze_globalne_x, najlepsze_globalne_y):
        MAX_V = 0.5
        r1 = random.random()
        r2 = random.random()

        komponent_poznawczy_x = self.stala_poznawcza * r1 * (self.najlepsze_x - self.x)
        komponent_spoleczny_x = self.stala_spoleczna * r2 * (najlepsze_globalne_x - self.x)

        komponent_poznawczy_y = self.stala_poznawcza * r1 * (self.najlepsze_y - self.y)
        komponent_spoleczny_y = self.stala_spoleczna * r2 * (najlepsze_globalne_y - self.y)

        self.predkosc_x = self.inercja * self.predkosc_x + komponent_poznawczy_x + komponent_spoleczny_x
        self.predkosc_y = self.inercja * self.predkosc_y + komponent_poznawczy_y + komponent_spoleczny_y

        self.predkosc_x = np.clip(self.predkosc_x, -MAX_V, MAX_V)
        self.predkosc_y = np.clip(self.predkosc_y, -MAX_V, MAX_V)

    def zaktualizuj_pozycje(self):
        self.x += self.predkosc_x
        self.y += self.predkosc_y

        self.x = np.clip(self.x, -5, 5)
        self.y = np.clip(self.y, -5, 5)


def wygeneruj_roj(liczba_czastek, inercja, stala_poznawcza, stala_spoleczna, maksymalizacja):
    czastki = []
    for _ in range(liczba_czastek):
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        czastki.append(
            Czastka(x, y, inercja, stala_poznawcza, stala_spoleczna, maksymalizacja)
        )
    return czastki

def pobierz_najlepsze(roj, funkcja_przystosowania, maksymalizacja=False):
    if maksymalizacja:
        najlepsze_global = -float('inf')
    else:
        najlepsze_global = float('inf')

    najlepsza_czastka = None

    for czastka in roj:
        czastka.zaktualizuj_przystosowanie(funkcja_przystosowania)
        if maksymalizacja:
            if czastka.przystosowanie > najlepsze_global:
                najlepsze_global = czastka.przystosowanie
                najlepsza_czastka = czastka
        else:
            if czastka.przystosowanie < najlepsze_global:
                najlepsze_global = czastka.przystosowanie
                najlepsza_czastka = czastka

    return najlepsza_czastka


def algorytm_roju(funkcja_przystosowania,
                  liczba_czastek,
                  inercja,
                  stala_poznawcza,
                  stala_spoleczna,
                  maks_iteracje,
                  maksymalizacja=False):
    roj = wygeneruj_roj(
        liczba_czastek,
        inercja,
        stala_poznawcza,
        stala_spoleczna,
        maksymalizacja
    )

    najlepsza_czastka_globalna = pobierz_najlepsze(roj, funkcja_przystosowania, maksymalizacja)


    for iteracja in range(maks_iteracje):
        for czastka in roj:
            czastka.zaktualizuj_predkosc(
                najlepsza_czastka_globalna.najlepsze_x,
                najlepsza_czastka_globalna.najlepsze_y
            )
            czastka.zaktualizuj_pozycje()

        nowa_najlepsza = pobierz_najlepsze(roj, funkcja_przystosowania, maksymalizacja)

        if maksymalizacja:
            if nowa_najlepsza.przystosowanie > najlepsza_czastka_globalna.przystosowanie:
                najlepsza_czastka_globalna = nowa_najlepsza
        else:
            if nowa_najlepsza.przystosowanie < najlepsza_czastka_globalna.przystosowanie:
                najlepsza_czastka_globalna = nowa_najlepsza

        print(
            f"Iteracja {iteracja + 1}/{maks_iteracje}, "
            f"Najlepsze przystosowanie: {najlepsza_czastka_globalna.przystosowanie}"
        )



