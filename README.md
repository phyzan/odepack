Single-command installation to usr/include folder:

git clone https://github.com/phyzan/odepack && cd odepack && chmod +x install.sh && sudo ./install.sh && cd ..


compile python extension (replace 3.12 with your python version):

g++ -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) pyode.cpp -o odepack$(python3-config --extension-suffix)