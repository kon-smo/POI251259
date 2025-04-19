import csv
import numpy as np
import random
from sklearn.cluster import KMeans


vertical_file = 'cwiczenie1/vertical.xyz'
horizontal_file = 'cwiczenie1/horizontal.xyz'
cylindrical_file = 'cwiczenie1/cylindrical.xyz'


def load_xyz_file(filepath):

    points = []
    with open(filepath, newline='') as csvfile:
        
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if len(row) < 3:
                continue
            try:
                x, y, z = float(row[0]), float(row[1]), float(row[2])
                points.append([x, y, z])
            except ValueError:
                continue
    return np.array(points)


def fit_plane_from_points(p1, p2, p3):
    
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None  # punkty są współliniowe
    normal = normal / norm
    d = -np.dot(normal, p1)
    return normal, d

def distance_to_plane(point, normal, d):
    
    return abs(np.dot(normal, point) + d)

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

        
        inliers = []
        for point in points:
            if distance_to_plane(point, normal, d) < distance_threshold:
                inliers.append(point)
        if len(inliers) > best_inliers_count:
            best_inliers_count = len(inliers)
            best_plane = (normal, d)
            best_inliers = np.array(inliers)
    
    
    if best_plane is None:
        return None, None, None
    
    if len(best_inliers) >= 3:
        centroid = np.mean(best_inliers, axis=0)
        centered = best_inliers - centroid
        
        U, S, Vt = np.linalg.svd(centered)
        
        normal = Vt[-1, :]
        d = -np.dot(normal, centroid)
        best_plane = (normal, d)
    
    avg_distance = np.mean([distance_to_plane(p, best_plane[0], best_plane[1]) for p in best_inliers])
    return best_plane, best_inliers, avg_distance

def classify_plane(normal, avg_distance, plane_error_threshold=0.01):
    
    if avg_distance < plane_error_threshold:
        
        if abs(normal[2]) > 0.9:
            return "płaszczyzna pozioma"
        elif abs(normal[2]) < 0.1:
            return "płaszczyzna pionowa"
        else:
            return "płaszczyzna skośna"
    else:
        return "nie jest płaszczyzną"


if __name__ == '__main__':
    
    points_vertical = load_xyz_file(vertical_file)
    points_horizontal = load_xyz_file(horizontal_file)
    points_cylindrical = load_xyz_file(cylindrical_file)
    
    
    all_points = np.vstack((points_vertical, points_horizontal, points_cylindrical))
    
   
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(all_points)
    labels = kmeans.labels_
    
    
    clusters = {}
    for i in range(3):
        clusters[i] = all_points[labels == i]
    
    
    for cluster_id, cluster_points in clusters.items():
        print(f"\nKlaster {cluster_id}: liczba punktów = {len(cluster_points)}")
        plane, inliers, avg_distance = ransac_plane_fitting(cluster_points, distance_threshold=0.02, iterations=1000)
        if plane is None:
            print("Nie znaleziono modelu płaszczyzny dla tego klastra.")
            continue
        normal, d = plane
        classification = classify_plane(normal, avg_distance, plane_error_threshold=0.01)
        print("Wektor normalny płaszczyzny:", normal)
        print("Wyraz wolny d:", d)
        print("Średnia odległość inlierów od płaszczyzny:", avg_distance)
        print("Klasyfikacja chmury:", classification)