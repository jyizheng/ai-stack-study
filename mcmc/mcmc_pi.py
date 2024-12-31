import random

def monte_carlo_pi(n_samples):
    inside = 0
    for _ in range(n_samples):
        x, y = random.random(), random.random()
        if x**2 + y**2 <= 1:
            inside += 1
    return 4 * inside / n_samples

print(monte_carlo_pi(1000000))

