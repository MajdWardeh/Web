import numpy as np
import pandas as pd
from scipy.special import binom


def main():
    n = 2
    for k in range(n+1):
        print(binom(n, k))


if __name__ == '__main__':
    main()
