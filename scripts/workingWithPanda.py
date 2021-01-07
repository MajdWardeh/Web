import numpy as np
# import pandas as import pd

# data = 
# df = pd.DataFrame(data, columns = ['Name', 'Age'])


from struct import pack, unpack


def write(array):
    with open('store', 'wb') as file:
        file.write(pack('d' * len(array) , *array))

def write(arr1, arr2):
    with open('store', 'wb') as file:
        arr1_pack = pack('d' * len(arr1) , *arr1)
        arr2_pack = pack('d' * len(arr2) , *arr2)
        file.write(arr1_pack + arr2_pack)
def read():
    with open('store', 'rb') as file:
        packed = file.read()
        array = unpack('d' * (len(packed) // 8), packed) # 8 bytes per double
    return array

def main():
    array = [0, 1, 39534.543, 834759435.3445643, 1.003024032, 0.032543, 434.020]
    array2 = [3, 4, 5]
    write(array, array2)
    array1 = read()
    print(type(array1))
    array1 = list(array1)
    print(type(array1))
    print(array1)
if __name__ == "__main__":
    main()
    