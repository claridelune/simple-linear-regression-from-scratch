import random

def generate_linear_data(num_points, alpha=2, beta=3, noise=0.5):
    """Generar datos sintéticos para una regresión lineal."""
    x = [random.random() for _ in range(num_points)]
    y = [alpha * x_i + beta + random.gauss(0, noise) for x_i in x]
    return x, y
