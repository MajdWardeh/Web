import numpy as np

def main():
    x = np.random.rand(12)
    y = np.random.rand(2)
    z = np.concatenate([x, y], axis=0)
    print(x.shape)
    print(y.shape)
    print(z.shape)
    print(x, y)
    print(z)

if __name__ == '__main__':
    main()