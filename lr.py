#Linear Regression.
#Best fit line with less error rate.
from numpy import *

# y = mx + c
# m is slope, c is y-intercept
def compute_error_for_line_given_points(c, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + c)) ** 2
    return totalError / float(len(points))

def step_gradient(c_current, m_current, points, learningRate):
    c_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        c_gradient += -(2/N) * (y - ((m_current * x) + c_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + c_current))
    new_c = c_current - (learningRate * c_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_c, new_m]

def gradient_descent_runner(points, starting_c, starting_m, learning_rate, num_iterations):
    c = starting_c
    m = starting_m
    for i in range(num_iterations):
        c, m = step_gradient(c, m, array(points), learning_rate)
    return [c, m]

def main():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_c = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at c = {0}, m = {1}, error = {2}".format(initial_c, initial_m, compute_error_for_line_given_points(initial_c, initial_m, points)))
    print(".....................Optimal Result [c, m]..............................")
    [c, m] = gradient_descent_runner(points, initial_c, initial_m, learning_rate, num_iterations)
    print("After {0} iterations c = {1}, m = {2}, error = {3}".format(num_iterations, c, m, compute_error_for_line_given_points(c, m, points)))

if __name__ == '__main__':
    main()