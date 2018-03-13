import numpy as np
from numpy.linalg import inv

eps = 10 ** -8


def func(x):
    a, b, c, d, e, f = 1., 4., 0.001, 0., -1., 0.
    A = np.array([[a, c / 2],
                  [c / 2, b]])
    B = np.array([d, e])
    C = f
    return A.dot(x).dot(x) + B.dot(x) + C


def unit_vector(i, n):
    return np.array([0 if i != j else 1 for j in range(n)])


def derivative(f, x, h=eps):
    n = len(x)
    grad = [(f(x + unit_vector(i, n).dot(h)) - f(x - unit_vector(i, n).dot(h))) / (2. * h) for i in range(n)]
    return np.array(grad)


def second_derivative(f, x, h=eps):
    n = len(x)
    hesse = [np.array([(f(x + (unit_vector(i, n) + unit_vector(j, n)) * h)
                        - f(x + (unit_vector(i, n) - unit_vector(j, n)) * h)
                        - f(x - (unit_vector(i, n) - unit_vector(j, n)) * h)
                        + f(x - (unit_vector(i, n) + unit_vector(j, n)) * h))
                       / (4. * h ** 2) for j in range(n)]) for i in range(n)]
    return np.array(hesse)


def minimize_one_dimension_golden_ratio(f, b=1/eps, a=-1/eps, eps=eps):
    F = (1. + 5 ** 0.5) / 2
    while abs(b - a) > eps:
        x1 = b - (b - a) / F
        x2 = a + (b - a) / F
        if f(x1) >= f(x2):
            a = x1
        else:
            b = x2
    return (a + b) / 2


def minimize_one_dimension_brute_force(f, b, a, n: int):
    x_min = a + (b - a) / (n + 1)
    for i in range(2, n + 1):
        x = a + i * (b - a) / (n + 1)
        x_min = x if f(x) < f(x_min) else x_min
    return x_min


def minimize_one_dimension(f, method, eps=eps):
    if method == "golden_ratio":
        return minimize_one_dimension_golden_ratio(f, a=0)
    elif method == "brute_force":
        return minimize_one_dimension_brute_force(f, b=1, a=0, n=int(eps**-.5))


def choose_step_fastest(f, x, h, method):
    def f_a(a, f=f, x=x, h=h):
        return f(x + a * h)

    return minimize_one_dimension(f_a, method)


def choose_step_fragmentation(f, x, h, beta=1, λ=0.5, eps=eps):
    alpha = beta
    while f(x + alpha * h) - f(x) > eps * alpha * derivative(f, x, eps).dot(h):
        alpha *= λ
    return alpha


def choose_step(f, x, h, method):
    if method == "fragmentation":
        return choose_step_fragmentation(f, x, h)
    elif method == "fastest_golden_ratio":
        return choose_step_fastest(f, x, h, method="golden_ratio")
    elif method == "fastest_brute_force":
        return choose_step_fastest(f, x, h, method="brute_force")


def norm(x):
    return sum([i ** 2 for i in x]) ** 0.5


def minimize(f, x0, step_method, eps=eps, output=0):
    i = 0
    x = np.copy(x0)
    h = -inv(second_derivative(f, x, eps)).dot(derivative(f, x, eps))
    alpha = choose_step(f, x, h, step_method)
    print("x{0} = {1}\nα{0} = {2}\nf(x{0}) = {3}".format(i, x, alpha, f(x)))
    x1 = x + alpha * h
    while norm(x1 - x) > eps:
        i += 1
        x = np.copy(x1)
        h = -inv(second_derivative(f, x, eps)).dot(derivative(f, x, eps))
        alpha = choose_step(f, x, h, step_method)
        print("x{0} = {1}\nα{0} = {2}\nf(x{0}) = {3}".format(i, x, alpha, f(x))) if i <= output else None
        x1 = x + alpha * h
    return x1


x0 = np.array([0., 0.])
output = 3
print("* * *   Метод Ньютона   * * *\n")
print("Метод дроблення кроку")
min = minimize(func, x0, "fragmentation", eps, output)
print("Розв'язок: ", min,"\nmin f = ",func(min))
print("Метод найшвидшого спуску з використанням методу золотого перетину одновимірної оптимізації")
min = minimize(func, x0, "fastest_golden_ratio", eps, output)
print("Розв'язок: ", min,"\nmin f = ",func(min))
print("Метод найшвидшого спуску з використанням методу перебору одновимірної оптимізації")
min = minimize(func, x0, "fastest_brute_force", eps, output)
print("Розв'язок: ", min,"\nmin f = ",func(min))