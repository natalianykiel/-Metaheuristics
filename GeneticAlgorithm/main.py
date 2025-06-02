from random import random, randint, choices, sample
from typing import List
class Osobnik:
    def __init__(self, chromosom: List[int], wagi: List[int], wartosci: List[int], max_waga: int):
        self.chromosom = chromosom
        self.wagi = wagi
        self.wartosci = wartosci
        self.max_waga = max_waga
        self.waga = 0
        self.wartosc = 0
        self.ocenaPrzystosowania = self.funkcja_przystosowania()
    
    def funkcja_przystosowania(self):
        self.waga = sum(w * c for w, c in zip(self.wagi, self.chromosom))
        self.wartosc = sum(v * c for v, c in zip(self.wartosci, self.chromosom))
        
        if self.waga > self.max_waga:
            self.wartosc = 0
            self.waga = 0
        
        return self.waga

    def _wykonaj_mutacje(self):
        index = randint(0, len(self.chromosom) - 1)
        self.chromosom[index] = 1 - self.chromosom[index]  

    def mutacja(self):
        self._wykonaj_mutacje()
        self.ocenaPrzystosowania = self.funkcja_przystosowania()

    def krzyzowanie(self, osobnik2: 'Osobnik', metoda='jednopunktowa') -> 'Osobnik':
        if metoda == 'jednopunktowa':
            punkt = randint(1, len(self.chromosom) - 1)
            dziecko_chromosom = self.chromosom[:punkt] + osobnik2.chromosom[punkt:]
        elif metoda == 'rownomierna':
            dziecko_chromosom = [self.chromosom[i] if random() > 0.5 else osobnik2.chromosom[i]
                                 for i in range(len(self.chromosom))]
        return Osobnik(dziecko_chromosom, self.wagi, self.wartosci, self.max_waga)


class AlgorytmGenetyczny:
    def __init__(self, wielkosc_populacji: int, wagi: List[int], wartosci: List[int], max_waga: int,
                 prawdopodobienstwo_mutacji: float,prawdopodobienstwo_krzyzowania: float, liczba_generacji: int, metoda_selekcji: str):
        self.wielkosc_populacji = wielkosc_populacji
        self.wagi = wagi
        self.wartosci = wartosci
        self.max_waga = max_waga
        self.prawdopodobienstwo_mutacji = prawdopodobienstwo_mutacji
        self.prawdopodobienstwo_krzyzowania = prawdopodobienstwo_krzyzowania
        self.liczba_generacji = liczba_generacji
        self.metoda_selekcji = metoda_selekcji
        self.populacja = self._generuj_populacje()

    def _generuj_populacje(self):
        populacja = []
        while len(populacja) < self.wielkosc_populacji:
            chromosom = [randint(0, 1) for _ in range(len(self.wagi))]
            osobnik = Osobnik(chromosom, self.wagi, self.wartosci, self.max_waga)
            if osobnik.ocenaPrzystosowania > 0:  
                populacja.append(osobnik)
        return populacja

    def _selekcja_ruletkowa(self):
        suma_przystosowan = sum(osobnik.ocenaPrzystosowania for osobnik in self.populacja)
        if suma_przystosowan == 0:
            return [choices(self.populacja)[0] for _ in range(self.wielkosc_populacji)]
        prawdopodobienstwa = [osobnik.ocenaPrzystosowania / suma_przystosowan for osobnik in self.populacja]
        return choices(self.populacja, prawdopodobienstwa, k=self.wielkosc_populacji)


    def _selekcja_turniejowa(self, rozmiar_turnieju=3):
        wybrani = []
        for _ in range(self.wielkosc_populacji):
            turniej = sample(self.populacja, rozmiar_turnieju)
            najlepszy = max(turniej, key=lambda o: o.ocenaPrzystosowania)
            wybrani.append(najlepszy)
        return wybrani

    def _selekcja(self):
        if self.metoda_selekcji == 'ruletkowa':
            return self._selekcja_ruletkowa()
        elif self.metoda_selekcji == 'turniejowa':
            return self._selekcja_turniejowa()
        else:
            raise ValueError("Nieznana metoda selekcji")

    def run(self):
        najlepszy_osobnik = None
        for generacja in range(self.liczba_generacji):
            nowa_populacja = self._selekcja()
            potomstwo = []
            for i in range(0, len(nowa_populacja), 2):
                rodzic1, rodzic2 = nowa_populacja[i], nowa_populacja[min(i + 1, len(nowa_populacja) - 1)]

                if random() < self.prawdopodobienstwo_krzyzowania:
                    dziecko = rodzic1.krzyzowanie(rodzic2, metoda='jednopunktowa')
                else:
                    dziecko = rodzic1 if rodzic1.ocenaPrzystosowania > rodzic2.ocenaPrzystosowania else rodzic2
                
                if random() < self.prawdopodobienstwo_mutacji:
                    dziecko.mutacja()

                potomstwo.append(dziecko)
            self.populacja = potomstwo
            najlepszy_w_generacji = max(self.populacja, key=lambda o: o.wartosc)
            if not najlepszy_osobnik or najlepszy_w_generacji.wartosc > najlepszy_osobnik.wartosc:
                najlepszy_osobnik = najlepszy_w_generacji
            print(f"Generacja {generacja}: Najlepsza waga: {najlepszy_w_generacji.waga}, wartość: {najlepszy_w_generacji.wartosc}")
        print(f"Najlepsze znalezione rozwiązanie: {najlepszy_osobnik.chromosom}, waga: {najlepszy_osobnik.waga}, wartość: {najlepszy_osobnik.wartosc}")


# Dane wejściowe
dane = [
    (32252, 68674), (225790, 471010), (468164, 944620), (489494, 962094),
    (35384, 78344), (265590, 579152), (497911, 902698), (800493, 1686515),
    (823576, 1688691), (552202, 1056157), (323618, 677562), (382846, 833132),
    (44676, 99192), (169738, 376418), (610876, 1253986), (854190, 1853562),
    (671123, 1320297), (698180, 1301637), (446517, 859835), (909620, 1677534),
    (904818, 1910501), (730061, 1528646), (931932, 1827477), (952360, 2068204),
    (926023, 1746556), (978724, 2100851)
]
wagi = [waga for waga, _ in dane]
wartosci = [wartosc for _, wartosc in dane]
max_waga = 6404180

# Uruchomienie algorytmu
algorytm = AlgorytmGenetyczny(
    wielkosc_populacji=20,
    wagi=wagi,
    wartosci=wartosci,
    max_waga=max_waga,
    prawdopodobienstwo_mutacji=0.7,
    prawdopodobienstwo_krzyzowania=0.7,
    liczba_generacji=100,
    metoda_selekcji='ruletkowa'
)
algorytm.run()


