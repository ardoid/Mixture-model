#include <vector>
#include <iostream>

#include "Classcode/GP_DataSet.hh"

using namespace std;
using namespace CLASSCODE;

typedef GP_Vector InputType;
typedef int OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;

int main(int argc, char **argv) {
  if(argc < 2) {
    std::cerr << "usage:" << argv[0] << " <dataset.dat>" << std::endl;
    exit(1);
  }
   
  std::cout << "reading data from " << argv[1] << "..." << std::endl;
  DataSetType data;
  data.Read(std::string(argv[1]));
  
  std::cout << "size of dataset: " << data.Size() << std::endl;
  
  std::cout << "First 20 values:" << std::endl;
  for(size_t i = 0; i < 20; ++i) {
    InputType  x  = data.GetInput()[i];
    OutputType y  = data.GetOutput()[i];
    
    std::cout << "Input: " << x << " Output: " << y << std::endl;
  }

  return 0;
}

