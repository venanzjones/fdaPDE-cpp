#include <gtest/gtest.h> // testing framework
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "src/clustering_test.cpp"

int main(int argc, char **argv){
  // start testing
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
