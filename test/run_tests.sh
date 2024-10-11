cmake CMakeLists.txt
make

mkdir build 
mv fdapde_test build/
cd build/
./fdapde_test

cd ..
rm cmake_install.cmake 
rm CTestTestfile.cmake
rm Makefile
rm lib/ -r
rm bin/ -r
rm compile_commands.json

rm 'fdapde_test[1]_tests.cmake'
rm 'fdapde_test[1]_include.cmake'
rm CMakeFiles/ -r
rm CMakeCache.txt
rm _deps/ -r

