import numpy as np


def multivariate_hypergeometric(m, n):
    """
    Efficient sampling from a multivariate hypergeometric distribution.

    Parameters:
    m (array-like): Number of items of each type in the population
    n (int): Number of draws

    Returns:
    array: Random sample
    """
    remaining = n
    result = np.zeros(len(m), dtype=np.int64)
    total = np.sum(m)

    for i in range(len(m) - 1):
        if total > 0:
            prob = m[i] / total
            draw = np.random.binomial(remaining, prob)
            result[i] = draw
            remaining -= draw
            total -= m[i]
        else:
            break

    result[-1] = remaining
    return result


# Example usage
if __name__ == "__main__":
    m = np.array([1000000, 2000000, 1500000], dtype=np.int64)  # Large population
    n = 100000  # Large number of draws

    sample = multivariate_hypergeometric(m, n)
    print("Sample:", sample)
    print("Sum of sample:", np.sum(sample))
    print("Expected proportions:", m / np.sum(m))
    print("Actual proportions:", sample / n)

    # Benchmark
    from timeit import timeit

    def benchmark(func, m, n, number=100):
        return timeit(lambda: func(m, n), number=number) / number

    time_taken = benchmark(multivariate_hypergeometric, m, n)
    print(f"Average time taken: {time_taken:.6f} seconds")
