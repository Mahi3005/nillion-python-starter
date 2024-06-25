from nada_dsl import *
import math

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return abs(a*b) // gcd(a, b)

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

def fibonacci(n):
    fib_seq = [0, 1]
    for i in range(2, n):
        fib_seq.append(fib_seq[-1] + fib_seq[-2])
    return fib_seq[:n]

def nada_main():
    party1 = Party(name="Party1")
    my_int1 = SecretInteger(Input(name="my_int1", party=party1))
    my_int2 = SecretInteger(Input(name="my_int2", party=party1))

    # Compute basic arithmetic operations
    sum_result = my_int1 + my_int2
    difference_result = my_int1 - my_int2
    abs_difference_result = abs(my_int1 - my_int2)
    product_result = my_int1 * my_int2
    division_result = my_int1 // my_int2 if my_int2 != 0 else "undefined"
    modulus_result = my_int1 % my_int2 if my_int2 != 0 else "undefined"
    exponentiation_result = my_int1 ** my_int2

    # Logical comparisons
    equality_result = my_int1 == my_int2
    greater_than_result = my_int1 > my_int2
    less_than_result = my_int1 < my_int2

    # Statistical operations
    mean_result = (my_int1 + my_int2) / 2
    max_result = max(my_int1, my_int2)
    min_result = min(my_int1, my_int2)

    # Conditional logic
    conditional_message = "my_int1 is greater" if my_int1 > my_int2 else "my_int2 is greater or equal"

    # Advanced functionalities
    prime_check1 = is_prime(my_int1)
    prime_check2 = is_prime(my_int2)
    gcd_result = gcd(my_int1, my_int2)
    lcm_result = lcm(my_int1, my_int2)
    factorial1 = factorial(my_int1)
    factorial2 = factorial(my_int2)
    fibonacci_sequence = fibonacci(my_int1)  # Generating Fibonacci sequence based on my_int1
    binary_representation1 = bin(my_int1)
    binary_representation2 = bin(my_int2)
    hex_representation1 = hex(my_int1)
    hex_representation2 = hex(my_int2)

    return [
        Output(sum_result, "sum_output", party1),
        Output(difference_result, "difference_output", party1),
        Output(abs_difference_result, "abs_difference_output", party1),
        Output(product_result, "product_output", party1),
        Output(division_result, "division_output", party1),
        Output(modulus_result, "modulus_output", party1),
        Output(exponentiation_result, "exponentiation_output", party1),
        Output(equality_result, "equality_output", party1),
        Output(greater_than_result, "greater_than_output", party1),
        Output(less_than_result, "less_than_output", party1),
        Output(mean_result, "mean_output", party1),
        Output(max_result, "max_output", party1),
        Output(min_result, "min_output", party1),
        Output(conditional_message, "conditional_message_output", party1),
        Output(prime_check1, "prime_check1_output", party1),
        Output(prime_check2, "prime_check2_output", party1),
        Output(gcd_result, "gcd_output", party1),
        Output(lcm_result, "lcm_output", party1),
        Output(factorial1, "factorial1_output", party1),
        Output(factorial2, "factorial2_output", party1),
        Output(fibonacci_sequence, "fibonacci_output", party1),
        Output(binary_representation1, "binary_representation1_output", party1),
        Output(binary_representation2, "binary_representation2_output", party1),
        Output(hex_representation1, "hex_representation1_output", party1),
        Output(hex_representation2, "hex_representation2_output", party1)
    ]
