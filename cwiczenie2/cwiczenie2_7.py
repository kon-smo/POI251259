import csv
import numpy as np
import random


def load_xyz_file(filepath):
    
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                points.append([x, y, z])
            except ValueError:
                continue
    return np.array(points)

def save_xyz_file(filepath, points):
    
    with open(filepath, 'w') as f:
        for pt in points:
            f.write("{:.6f} {:.6f} {:.6f}\n".format(pt[0], pt[1], pt[2]))


def fit_plane_from_points(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None
    normal = normal / norm
    d = -np.dot(normal, p1)
    return normal, d

def distance_to_plane_vectorized(points, normal, d):
    
    return np.abs(np.dot(points, normal) + d)

def ransac_plane_fitting(points, distance_threshold=0.02, iterations=1000):
    best_inliers_count = 0
    best_plane = None
    best_inliers = None
    n_points = len(points)
    if n_points < 3:
        return None, None, None

    for i in range(iterations):
        
        indices = random.sample(range(n_points), 3)
        p1, p2, p3 = points[indices[0]], points[indices[1]], points[indices[2]]
        plane = fit_plane_from_points(p1, p2, p3)
        if plane is None:
            continue
        normal, d = plane

        
        distances = distance_to_plane_vectorized(points, normal, d)
        inlier_mask = distances < distance_threshold
        n_inliers = np.sum(inlier_mask)
        if n_inliers > best_inliers_count:
            best_inliers_count = n_inliers
            best_plane = (normal, d)
            best_inliers = points[inlier_mask]

    if best_plane is None or best_inliers is None or len(best_inliers) == 0:
        return None, None, None

    if len(best_inliers) >= 3:
        centroid = np.mean(best_inliers, axis=0)
        centered = best_inliers - centroid
        U, S, Vt = np.linalg.svd(centered)
        normal = Vt[-1, :]
        d = -np.dot(normal, centroid)
        best_plane = (normal, d)

    avg_distance = np.mean(distance_to_plane_vectorized(best_inliers, best_plane[0], best_plane[1]))
    return best_plane, best_inliers, avg_distance


def iterative_plane_extraction(points, iterations=6, distance_threshold=0.02, ransac_iterations=1000):
    remaining_points = points.copy()
    extracted_planes = []
    
    for k in range(iterations):
        if len(remaining_points) < 3:
            print("Zbyt mało punktów do dalszego dopasowania.")
            break
        
        plane, inliers, avg_distance = ransac_plane_fitting(remaining_points,
                                                             distance_threshold=distance_threshold,
                                                             iterations=ransac_iterations)
        if plane is None or inliers is None or len(inliers) == 0:
            print(f"Iteracja {k}: nie znaleziono płaszczyzny.")
            break
        
        extracted_planes.append(inliers)
        print(f"Iteracja {k}: wykryto płaszczyznę z {len(inliers)} inlierami (średnia odległość: {avg_distance:.4f}).")
        
        
        mask = distance_to_plane_vectorized(remaining_points, plane[0], plane[1]) >= distance_threshold
        remaining_points = remaining_points[mask]
        print(f"Iteracja {k}: po usunięciu, pozostało {len(remaining_points)} punktów.")
    
    return extracted_planes, remaining_points
def subsample_points(points, sample_ratio=0.2):
    n = len(points)
    sample_size = int(n * sample_ratio)
    indices = np.random.choice(n, size=sample_size, replace=False)
    return points[indices]

if __name__ == '__main__':
    
    input_file = 'cwiczenie2/conferenceRoom_1.txt'
    
    
    points = load_xyz_file(input_file)
    print("Wczytano chmurę punktów o wymiarach:", points.shape)
    
    # 5% punktow na raz
    subsample_ratio = 0.05
    points = subsample_points(points, sample_ratio=subsample_ratio)
    print("Po subsamplingu chmura ma wymiar:", points.shape)
    
    
    extracted_planes, remaining_points = iterative_plane_extraction(points,
                                                                    iterations=6,
                                                                    distance_threshold=0.02,
                                                                    ransac_iterations=200)
    
    
    for i, plane_cloud in enumerate(extracted_planes):
        filename = f"cwiczenie2/extracted_plane_{i}.xyz"
        save_xyz_file(filename, plane_cloud)
        print(f"Zapisano wyekstrahowaną płaszczyznę {i} do pliku: {filename}")
    
    
    if len(remaining_points) > 0:
        save_xyz_file("cwiczenie2/remaining_points.xyz", remaining_points)
        print("Zapisano pozostałe punkty do pliku: remaining_points.xyz")