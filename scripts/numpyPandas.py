import numpy as np
import pandas as pd

def store(xs):
    df = pd.DataFrame({"a": xs})
    df.to_pickle("temp.pkl")

def restore():
    df = pd.read_pickle('temp.pkl')
    xs_df = df['a'].tolist()
    return xs_df


def main():
    np.random.seed(0)
    xs = [np.random.rand(500, 4) for _ in range(5000)]
    store(xs)    
    xs_df = restore()
    print(np.allclose(xs, xs_df))


if __name__ == '__main__':
    main()