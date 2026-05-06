#Author: Ze Yu Jiang

#Date: 5/1/2026

import math
import matplotlib.pyplot as plt
import numpy as np

#make sure to install numpy and matplotlib before running this code
#run the following command in terminal if necessary:

#pip install numpy matplotlib



#reuse the building systems of equations function from project 2
def build_system_of_equations(n, diagonal_value):
    #we have 3 diagonals: subdiagonal (-1s), diagonal(2s), superdiagonal(-1s)
    subdiagonal = [-1.0] * (n - 1)  # sub
    diagonal = [diagonal_value] * n  # diag
    superdiagonal = [-1.0] * (n - 1)  # super
    
    #vector b
    denominator = (n + 1) ** 4
    b = [0.0] * n
    b[0] = 1.0 + (1 ** 2) / denominator
    for i in range(1, n - 1):
        b[i] = ((i + 1) ** 2) / denominator
    b[n - 1] = 6.0 + (n ** 2) / denominator
    return subdiagonal, diagonal, superdiagonal, b
        
def matrix_vector_product(subdiagonal, diagonal, superdiagonal, x):
    n = len(diagonal)
    result = [0.0] * n
    #define the matrix-vector product for a tridiagonal matrix

    #the matrix A is represented by the three diagonals: subdiagonal, diagonal, superdiagonal
    #the vector x is the input vector we want to multiply with the matrix A
    for i in range(n):
        result[i] += diagonal[i] * x[i]
        if i > 0:
            result[i] += subdiagonal[i - 1] * x[i - 1]
        if i < n - 1:
            result[i] += superdiagonal[i] * x[i + 1]
    return result


#operations for vectors
def inner_product(v1, v2):
    sum = 0.0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def vector_subtraction(v1, v2):
    result = [0.0] * len(v1)
    for i in range(len(v1)):
        result[i] = v1[i] - v2[i]
    return result

#l2 norm aka euclidean norm
def l2_norm(v):
    sum = 0.0
    for i in range(len(v)):
        sum += v[i] ** 2
    return math.sqrt(sum)

#operations for scaled vector to another vector
def add_scaled_vector(v1, v2, scale):
    result = [0.0] * len(v1)
    for i in range(len(v1)):
        result[i] = v1[i] + scale * v2[i]
    return result

def subtract_scaled_vector(v1, v2, scale):
    result = [0.0] * len(v1)
    for i in range(len(v1)):
        result[i] = v1[i] - scale * v2[i]
    return result



#conjugate gradient method
#for solving Ax = b
#start with an initial guess y0, 
#and iteratively improve the solution until we reach the desired tolerance or max iterations
#first residual is r(0) = b - Ax(0)
#first search direction is  p(0) = r(0)
#repeat for k = 0, 1, 2, etc until it converges
    #Ap(k) = matrix-vector product of A and p(k)
    #alpha(k) = (r(k)Transpose *  r(k)) / (p(k)Transpose * Ap(k))
    #x(k+1) = x(k) + alpha(k) * p(k)
    #r(k+1) = r(k) - alpha(k) * Ap(k)
    #beta_k = (r(k+1)Transpose * r(k+1)) / (r(k)Transpose * r(k))
    #p(k+1) = r(k+1) + beta_k * p(k)
    #end the loop when the l2 norm of r(k) 
    #is less than the specified tolerance or when we reach the maximum number of iterations
def conjugate_gradient(subdiagonal, diagonal, superdiagonal, b, y0, n, tolerance=1e-8):
    x = y0.copy()
    r = vector_subtraction(b, matrix_vector_product(subdiagonal, diagonal, superdiagonal, x))
    p = r.copy()
    residual_norm = l2_norm(r)
    first_k_below_tolerance = None

    for k in range(1, n + 1):
        Ap = matrix_vector_product(subdiagonal, diagonal, superdiagonal, p)
        alpha = inner_product(r, r) / inner_product(p, Ap)
        x = add_scaled_vector(x, p, alpha)
        r_new = subtract_scaled_vector(r, Ap, alpha)
        residual_norm = l2_norm(r_new)
        
        if residual_norm < tolerance:
            first_k_below_tolerance = k
            r = r_new
            break

        beta = inner_product(r_new, r_new) / inner_product(r, r)
        p = add_scaled_vector(r_new, p, beta)
        r = r_new

    return x, k, residual_norm, first_k_below_tolerance





#test for n size 10, 100, 1000, 2000 
#test each n for both diagonal value of 2 and 3 and for both tolerance levels of 1e-8 and 1e-16
def main():
    n_tests = [10, 100, 1000, 2000]
    print("########################################################################\n")
    print("Conjugate Gradient Method Results:")
    print("Tolerance = 1e-8\n")

    for n in n_tests:
        print("########################################################################\n")
        print(f"Testing for n = {n}...")
        #test for diagonal value of 2
        print("A system: diagonal value = 2")
        subdiagonal, diagonal, superdiagonal, b = build_system_of_equations(n, diagonal_value=2.0)
        y0 = [0.0] * n
        x, iterations, final_residual_norm, first_k_below_tolerance = conjugate_gradient(subdiagonal, diagonal, superdiagonal, b, y0, n, tolerance=1e-8)

        print(f"Conjugate Gradient stopped after {iterations} iterations")
        print(f"Final residual norm: {final_residual_norm:.6e}")
        print(f"First iteration k below tolerance: {first_k_below_tolerance}")

        #test for diagonal value of 3
        print("\nA^ system: diagonal value = 3")
        subdiagonal, diagonal, superdiagonal, b = build_system_of_equations(n, diagonal_value=3.0)
        y0 = [0.0] * n

        x, iterations, final_residual_norm, first_k_below_tolerance = conjugate_gradient(subdiagonal, diagonal, superdiagonal, b, y0, n, tolerance=1e-8)

        print(f"Conjugate Gradient stopped after {iterations} iterations")
        print(f"Final residual norm: {final_residual_norm:.6e}")
        print(f"First iteration k below tolerance: {first_k_below_tolerance}")
        
    print("\n************************************************************************\n")
    print("Tolerance = 1e-16\n")

    for n in n_tests:
        print("************************************************************************\n")
        print(f"Testing for n = {n}...")
        #test for diagonal value of 2
        print("A system: diagonal value = 2")
        subdiagonal, diagonal, superdiagonal, b = build_system_of_equations(n, diagonal_value=2.0)
        y0 = [0.0] * n
        x, iterations, final_residual_norm, first_k_below_tolerance = conjugate_gradient(subdiagonal, diagonal, superdiagonal, b, y0, n, tolerance=1e-16)

        print(f"Conjugate Gradient stopped after {iterations} iterations")
        print(f"Final residual norm: {final_residual_norm:.6e}")
        print(f"First iteration k below tolerance: {first_k_below_tolerance}")

        #test for diagonal value of 3
        print("\nA^ system: diagonal value = 3")
        subdiagonal, diagonal, superdiagonal, b = build_system_of_equations(n, diagonal_value=3.0)
        y0 = [0.0] * n

        x, iterations, final_residual_norm, first_k_below_tolerance = conjugate_gradient(subdiagonal, diagonal, superdiagonal, b, y0, n, tolerance=1e-16)

        print(f"Conjugate Gradient stopped after {iterations} iterations")
        print(f"Final residual norm: {final_residual_norm:.6e}")
        print(f"First iteration k below tolerance: {first_k_below_tolerance}")


main()