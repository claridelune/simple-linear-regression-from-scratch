from utils import de_mean, mean, standard_deviation, correlation
from gradient_descent import GradientDescent

class LinearRegression:
    def __init__(self):
        self.alpha = None
        self.beta = None
    
    def predict(self, x_i):
        if self.alpha is None or self.beta is None:
            raise ValueError("El modelo debe ajustarse antes de realizar predicciones.")
        return self.beta * x_i + self.alpha

    def error(self, x_i, y_i):
        return y_i - self.predict(x_i)

    def sum_of_squared_errors(self, x, y):
        return sum(self.error(x_i, y_i) ** 2 for x_i, y_i in zip(x, y))

    def least_squares_fit(self, x, y):
        """given training values for x and y,
        find the least-squares values of alpha and beta"""
        self.beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
        self.alpha = mean(y) - self.beta * mean(x)
        return self.alpha, self.beta
    
    def total_sum_of_squares(self, y):
        """the total squared variation of y_i's from their mean"""
        return sum(v ** 2 for v in de_mean(y))
    
    def r_squared(self, x, y):
        """the fraction of variation in y captured by the model, which equals 1 - the fraction of variation in y not captured by the model"""
        return 1.0 - (self.sum_of_squared_errors(x, y) / self.total_sum_of_squares(y))
    