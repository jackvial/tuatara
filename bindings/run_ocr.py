import sys
import numpy as np

sys.path.append("../build/bindings/")
import pytuatara


def main():
    # pytuatara.image_to_data(
    #     "../images/resume_example.png",
    #     "../weights",
    #     "../outputs",
    #     "0"
    # )

    print("Test array interface")
    add_arrays_res = pytuatara.add_arrays(np.array([1, 2, 3]), np.array([4, 5, 6]))
    print("add_arrays_res: ", add_arrays_res)

main()
