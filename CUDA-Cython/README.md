# CUDA-Cython

This is the repo of a CUDA accelerated vector reduce function, which uses a Cython wrapper to be compiled to be a Python extension.

## Usage

1. Compile the `.cu` file first according to hardware. Here use architecture sm_86 as an example. Instead of obtian an excutable file, it will be a `reduce.lib` for compiling with Cython wrapper.

    ```bash
    nvcc -lib -O2 -arch=sm_86 -o reduce.lib reduce.cu
    ```

2. Compile the `reduce.lib` with the Cython wrapper by running `set.py`.
    
    ```bash
    python setup.py build_ext -i
    ```
3. run `main.py` for a test.

    ```bash
    python main.py
    ```

4. (optional)clean up

    ```bash
    del *.lib
    del *.cpp
    rmdir /s /q build
    ```

Dependency: All tested under Windows10, check `requirements.txt` for detail.