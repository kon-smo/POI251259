import matplotlib.pyplot as plt
import numpy as np

def generate_horizontal_plane(n_points, width, length, center=(0, 0, 0)):
    x = np.random.uniform(center[0] - width / 2, center[0] + width / 2, n_points)
    y = np.random.uniform(center[1] - length / 2, center[1] + length / 2, n_points)
    z = np.full(n_points, center[2])  # Stała wysokość
    return np.column_stack((x, y, z))

def generate_vertical_plane(n_points, width, height, center=(0, 0, 0)):
    x = np.random.uniform(center[0] - width / 2, center[0] + width / 2, n_points)
    y = np.full(n_points, center[1])  # Stała pozycja w osi Y
    z = np.random.uniform(center[2] - height / 2, center[2] + height / 2, n_points)
    return np.column_stack((x, y, z))

def generate_cylindrical_surface(n_points, radius, height, center=(0, 0, 0)):
    theta = np.random.uniform(0, 2 * np.pi, n_points)  # Losowy kąt
    z = np.random.uniform(center[2] - height / 2, center[2] + height / 2, n_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.column_stack((x, y, z))
    
def save_to_xyz(filename, points):
    np.savetxt(filename, points, fmt="%.6f", delimiter=" ")

horizontal_points = generate_horizontal_plane(1000, 10, 10)
vertical_points = generate_vertical_plane(1000, 10, 10)
cylindrical_points = generate_cylindrical_surface(1000, 5, 10)

save_to_xyz("horizontal.xyz", horizontal_points)
save_to_xyz("vertical.xyz", vertical_points)
save_to_xyz("cylindrical.xyz", cylindrical_points)