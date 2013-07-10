#include <vector>
#include <iostream>

#include <cstdlib>

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
  size_t val1 = rand()%data.Size(), val2 = rand()%data.Size();
  uint rnk;
  InputType c1, c2;
  int k = 2;
  std::vector<std::vector<int> > cluster;
  std::vector<GP_Vector> cluster_center(k, GP_Vector(2));
  cluster_center[0] = data.GetInput()[val1];
  cluster_center[1] = data.GetInput()[val2];

  for(int i=0;i<20;i++)
  {
      // reassign data points
      cluster.push_back(std::vector<int>());
      cluster.push_back(std::vector<int>());
      for (int j = 0;j<data.Size();j++)  {
          float dist[2];

          for (int z=0;z<k;z++)  {
               dist[z] = (cluster_center[z] - data.GetInput()[j]).Norm();
          }
          if(dist[0]>dist[1])
              cluster[1].push_back(j);
          else
              cluster[0].push_back(j);
      }

      // Calculate Cluster Centers
      GP_Vector sum(2);
      for(int j=0; j<k;j++)  {
          for(int z=0; z<cluster[j].size(); z++)  {
              sum = sum + data.GetInput()[cluster[j][z]];
          }
          sum = sum/cluster[j].size();
          cluster_center[j] = sum;
      }

  }

  std::cout << "Cluster1:" << cluster_center[0] << std::endl;
  std::cout << "Cluster2:" << cluster_center[1] << std::endl;


  return 0;
}

