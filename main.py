from regression import LinearRegression
from visualization import plot_regression
from data.generate import generate_linear_data

def main():
    x, y = generate_linear_data(num_points=100, alpha=2, beta=3, noise=0.1)

    model = LinearRegression()

    # Ajustar el modelo usando el método de mínimos cuadrados
    model.least_squares_fit(x, y)

    # Realizar predicciones para todos los valores de x
    y_pred = [model.predict(x_i) for x_i in x]

    # Calcular el coeficiente de determinación R²
    r_squared = model.r_squared(x, y)

    # Imprimir resultados
    print("Coeficiente alpha:", model.alpha)
    print("Coeficiente beta:", model.beta)
    print("Coeficiente de determinación R²:", r_squared)

    # Visualizar los resultados
    plot_regression(x, y, y_pred)

if __name__ == "__main__":
    main()
