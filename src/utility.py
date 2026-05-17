def ar1(n, mu, sigma, phi):
    eps = rng.normal(0, sigma, n)
    r = np.zeros(n)

    for i in range(1, n):
        r[i] = mu + phi * (r[i - 1] - mu) + eps[i]

    return r
