The ODE parameters and code philosophy closely resemble that of scipy's ODE and OdeSolver class on purpose, as some parts of the code (like the Runge-Kutta classes) have simply been translated from python to c++.
However this implementation targets performance, parallelization, higher flexibility, progress displaying, and better support for Event encounters during an ode integration.





# Installation and Compilation Guide

## Single-Command Installation to `/usr/include` Folder

```sh
git clone https://github.com/phyzan/odepack && cd odepack && chmod +x install.sh && sudo ./install.sh && cd ..
```

## Dependencies for MPFR and Eigen Support

```sh
sudo apt install libmpfrc++-dev
sudo apt install libeigen3-dev
```

---

# Optimized Compile Commands

## Compiling Python Extension (for Python 3.12)

```sh
g++ -O3 -Wall -march=x86-64 -shared -std=c++20 -fopenmp -fno-math-errno -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) <name>.cpp -o <name>$(python3-config --extension-suffix) -lmpfr -lgmp
```

## Compiling a C++ Program

```sh
g++ -O3 -Wall -march=native -std=c++20 -fopenmp -fno-math-errno -fPIC <name>.cpp -o <name> -lmpfr -lgmp
```
