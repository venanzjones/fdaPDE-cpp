<div align="center"> <h1> Clustering in fdaPDE </h1>

<h5> Physics-Informed Spatial and Functional Data Analysis </h5> </div>

![test-linux-gcc](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-cpp/test-linux-gcc.yml?branch=stable&label=test-linux-gcc)
![test-linux-clang](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-cpp/test-linux-clang.yml?branch=stable&label=test-linux-clang)
![test-macos-clang](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-cpp/test-macos-clang.yml?branch=stable&label=test-macos-clang)

This repository is a fork of the fdaPDE, an header only C++ library for the analysis of spatial and functional data observed over complex multidimensional domains.
It is built on top of the [fdaPDE Core Library](https://github.com/fdaPDE/fdaPDE-core).

This project was developed by MSc Mathematical Engineering student Alessandro Venanzi (10723478) under the supervision of professor Laura M. Sangalli, professor Eleonora Arnone, doctor Alessandro Palummo and doctor Michele Cavazzuti. 

Contribution features the whole implemenataion of the clustering module, as well as validation tests in both full-observable and partial-observable data.
The code structure of the library is presented in the image below:
<p align="center">
<img src="Images/fdaPDE_high_level_clust.png" width="400" height="410"  />
</p>

The header files added are the following:
<p align="center">
<img src="Images/folder_hierarchy.png"  width="260" height="350"/>
</p>

<h5> Installation and tests </h5> </div>

To reproduce our simulations, one must clone this repository via:

```bash
git clone --recursive https://github.com/venanzjones/fdaPDE-cpp.git -b develop-vena
```

To be able to compile the code, your system needs to have installed the following dependencies:
- A **C++20** compliant compiler
- **make**
- **CMake**
- **Eigen3 (>= 3.4)**
- **gtest (>= 1.14)**

After cloning the repository and making sure all dependencies are installed, move to the test folder and the following commands to compile the code and generate the executable:
```bash
cd fdaPDE-cpp/test
make
mkdir results
```

Finally, run the tests:
```bash
mkdir build
mv fdapde_test build
cd build
./fdapde_test
```

For more details, please refer to the documentation within the repository or contact me up alevena00@gmail.com




