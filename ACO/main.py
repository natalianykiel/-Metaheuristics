import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import random
import pandas as pd

class AntColony:
    def __init__(self, coordinates, num_ants, alpha, beta, evaporation_rate, iterations, randomness):
        self.coordinates = coordinates
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.randomness = randomness 
        self.num_points = len(coordinates)
        self.distances = self._calculate_distances()
        self.pheromone = np.ones((self.num_points, self.num_points))
        self.best_path = None
        self.best_distance = float('inf')
    
    def _calculate_distances(self):
        return np.array([[euclidean(a, b) for b in self.coordinates] for a in self.coordinates])
    
    def _probability(self, pheromone, visibility):
        return (pheromone ** self.alpha) * (visibility ** self.beta)
    
    def _choose_next(self, current_point, unvisited):
        if random.random() < self.randomness:  
            return random.choice(unvisited) 
        
        probabilities = []
        for next_point in unvisited:
            distance = self.distances[current_point, next_point]
            if distance == 0:  
                probabilities.append(0)
            else:
                probabilities.append(self._probability(
                    self.pheromone[current_point, next_point],
                    1 / distance
                ))

        probabilities = np.array(probabilities)
        if probabilities.sum() <= 0: 
            print(f"Warning: Probabilities sum to zero for point {current_point}. Choosing randomly.")
            return random.choice(unvisited)
        probabilities /= probabilities.sum()  
        
        return np.random.choice(unvisited, p=probabilities)


    
    def _update_pheromones(self, paths, distances):
        self.pheromone *= (1 - self.evaporation_rate)
        for path, dist in zip(paths, distances):
            if dist == 0: 
                continue
            for i in range(len(path) - 1):
                self.pheromone[path[i], path[i+1]] += 1 / dist
                self.pheromone[path[i+1], path[i]] += 1 / dist
        for i in range(len(self.pheromone)):
            for j in range(len(self.pheromone[i])):
                if self.pheromone[i, j] < 0:
                    self.pheromone[i, j] = 0  

    
    def run(self):
        for i in range(self.iterations):
            paths = []
            distances = []
            for j in range(self.num_ants):
                path = [random.randint(0, self.num_points - 1)]
                while len(path) < self.num_points:
                    unvisited = [p for p in range(self.num_points) if p not in path]
                    path.append(self._choose_next(path[-1], unvisited))
                path.append(path[0])
                paths.append(path)
                distances.append(self._calculate_path_distance(path))
            self._update_pheromones(paths, distances)
            min_distance = min(distances)
            if min_distance < self.best_distance:
                self.best_distance = min_distance
                self.best_path = paths[distances.index(min_distance)]
    
    def _calculate_path_distance(self, path):
        return sum(self.distances[path[i], path[i+1]] for i in range(len(path) - 1))



    #FUNKCJA RYSUJĄCA WYKRES PRZEDSTAWIAJĄCY TRASĘ
    def plot(self, filename=None):
        plt.figure(figsize=(8, 8))
        for i, j in zip(self.best_path, self.best_path[1:]):
            plt.plot(
                [self.coordinates[i][0], self.coordinates[j][0]],
                [self.coordinates[i][1], self.coordinates[j][1]],
                'b-'
            )
        plt.scatter(*zip(*self.coordinates), c='red', marker='o', label='Atrakcje')
        start = self.best_path[0]
        plt.scatter(self.coordinates[start][0], self.coordinates[start][1], 
                    c='green', marker='o', s=100, label='Start (Pierwsza atrakcja)')
        plt.title(f'Najlepsza trasa, odległość: {self.best_distance:.2f}')
        plt.legend()
        if filename:
            plt.savefig(filename)
        plt.show()



#OBSŁUGA URUCHOMIENIA, WYKRESÓW I PLIKU CSV
def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        coordinates = []
        for line in lines:
            _, x, y = map(int, line.split())
            coordinates.append((x, y))
    return coordinates


coordinates = load_data("A-n32-k5.txt") 
results = []


for i in range(5):
    ant_colony = AntColony(coordinates, num_ants=40, alpha=2, beta=10, evaporation_rate=0.5, iterations=200,  randomness=0.1)
    ant_colony.run()
    results.append({
        "run": i + 1,
        "distance": ant_colony.best_distance,
        "path": ant_colony.best_path
    })
    ant_colony.plot(filename=f"run_{i+1}.png")


distances = [res["distance"] for res in results]
best_result = min(distances)
worst_result = max(distances)
average_result = sum(distances) / len(distances)


plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), distances, marker='o', label='Odległości')
plt.axhline(best_result, color='green', linestyle='--', label=f'Najlepszy wynik: {best_result:.2f}')
plt.axhline(worst_result, color='red', linestyle='--', label=f'Najgorszy wynik: {worst_result:.2f}')
plt.axhline(average_result, color='blue', linestyle='--', label=f'Średni wynik: {average_result:.2f}')
plt.xlabel('Numer uruchomienia', fontsize=20)  
plt.ylabel('Odległość', fontsize=20)  
plt.tick_params(axis='both', labelsize=13) 
plt.legend(fontsize=13)  
plt.savefig("summary_results.png")
plt.show()


df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
