from Pyro4 import expose
import random

class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        self.pixels = []
        self.width = 0
        self.height = 0
        self.num_clusters = 0
        self.max_iter = 100
        print("Inited")

    def read_input(self):
        with open(self.input_file_name, 'r') as f:
            self.num_clusters = int(f.readline().strip())
            self.width, self.height = map(int, f.readline().split())
            pixel_data = []
            for line in f:
                pixel_data.extend(map(int, line.strip().split()))
            
            if len(pixel_data) != 3 * self.width * self.height:
                raise ValueError("Invalid pixel data")
            
            self.pixels = []
            for i in range(0, len(pixel_data), 3):
                self.pixels.append(tuple(pixel_data[i:i+3]))

    def write_output(self, closest_centroids, centroids):
        with open(self.output_file_name, 'w') as f:
            f.write("P3\n{} {}\n255\n".format(self.width, self.height))
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    idx = y * self.width + x
                    cluster = closest_centroids[idx]
                    row.extend(map(str, centroids[cluster]))
                f.write(' '.join(row) + '\n')

    def _init_centroids(self):
        return random.sample(self.pixels, self.num_clusters)

    def get_closest_centroids_par(self, centroids):
        chunk_size = len(self.pixels) // len(self.workers)
        chunks = [self.pixels[i*chunk_size:(i+1)*chunk_size] for i in range(len(self.workers))]
        
        # Distribute remaining pixels
        remaining = len(self.pixels) % len(self.workers)
        for i in range(remaining):
            chunks[i].append(self.pixels[-(i+1)])
        
        mapped = []
        for i in range(len(self.workers)):
            chunk = chunks[i]
            mapped.append(self.workers[i].get_closest_centroids(chunk, centroids))
        
        return self.myreduce(mapped)

    @staticmethod
    @expose
    def get_closest_centroids(pixels_chunk, centroids):
        closest = []
        for pixel in pixels_chunk:
            min_dist = float('inf')
            best_idx = 0
            for idx, centroid in enumerate(centroids):
                dist = sum((p-c)**2 for p, c in zip(pixel, centroid))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            closest.append(best_idx)
        return closest

    @staticmethod
    def myreduce(mapped):
        return [idx for chunk in mapped for idx in chunk]

    def _move_centroids(self, closest_centroids, current_centroids):
        new_centroids = []
        for i in range(self.num_clusters):
            cluster_points = [self.pixels[j] for j in range(len(self.pixels)) if closest_centroids[j] == i]
            
            if not cluster_points:
                new_centroids.append(current_centroids[i])
                continue
            
            avg_r = sum(p[0] for p in cluster_points) // len(cluster_points)
            avg_g = sum(p[1] for p in cluster_points) // len(cluster_points)
            avg_b = sum(p[2] for p in cluster_points) // len(cluster_points)
            new_centroids.append((avg_r, avg_g, avg_b))
        
        return new_centroids

    def _kmeans_objective(self, centroids, closest_centroids):
        total = 0
        for i, pixel in enumerate(self.pixels):
            c = centroids[closest_centroids[i]]
            total += sum((p - c)**2 for p, c in zip(pixel, c))
        return total

    def solve(self):
        print("Job Started")
        print(f"Workers: {len(self.workers)}")
        self.read_input()
        centroids = self._init_centroids()
        objective_history = []
        iteration = 0
        
        while True:
            closest = self.get_closest_centroids_par(centroids)
            new_centroids = self._move_centroids(closest, centroids)
            objective = self._kmeans_objective(new_centroids, closest)
            
            print(f"Iteration {iteration}: Objective {objective}")
            
            if objective_history:
                if (objective_history[-1] - objective) / objective_history[-1] < 0.01 or iteration >= self.max_iter:
                    break
            
            objective_history.append(objective)
            centroids = new_centroids
            iteration += 1
        
        self.write_output(closest, centroids)
        print("Job Finished")