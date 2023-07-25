import matplotlib.pyplot as plt

def plot_regression(x, y, y_pred):
    """Crear un gráfico que muestra los puntos de datos y la línea de regresión."""
    plt.scatter(x, y, label='Datos reales')  # Gráfico de dispersión de los datos reales
    plt.plot(x, y_pred, color='red', label='Línea de regresión')  # Línea de regresión
    plt.xlabel('Variable Independiente')
    plt.ylabel('Variable Dependiente')
    plt.title('Regresión Lineal')
    plt.legend()
    plt.show()
