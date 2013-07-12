#include <vector>
#include <iostream>
#include <limits>

#include <cstdlib>

#include "Classcode/GP_DataSet.hh"
#include "Classcode/GP_Matrix.hh"

using namespace std;
using namespace CLASSCODE;

typedef GP_Vector InputType;
typedef int OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;

void kmeans(const DataSetType& data)  {
    int K = 2;
    std::vector<std::vector<int> > cluster(2);
    std::vector<GP_Vector> cluster_center(K, GP_Vector(2));

    for(int i=0;i<20;i++)
    {
        for (int k=0; k<K; ++k) {
            cluster_center[k] = data.GetInput()[rand()%data.Size()];
            cluster[k].clear();
        }

        // reassign data points
        for (int j = 0;j<data.Size();j++)  {
            float minDistance = std::numeric_limits<float>::max();
            int closestK = 0;
            for (int k=0; k<K; ++k) {
                 float dist = (cluster_center[k] - data.GetInput()[j]).Norm();
                 if (dist < minDistance) {
                     minDistance = dist;
                     closestK = k;
                 }
                cluster[closestK].push_back(j);
            }
        }

        // Calculate Cluster Centers
        for (int k=0; k<K;k++)  {
            GP_Vector sum(2);
            sum[0] = 0.0f;
            sum[1] = 0.0f;
            for(int z=0; z<cluster[k].size(); z++)  {
                sum = sum + data.GetInput()[cluster[k][z]];
            }
            sum = sum/cluster[k].size();
            cluster_center[k] = sum;
        }

        std::cout << i << " Cluster1:" << cluster_center[0];
        std::cout << " Cluster2:" << cluster_center[1] << std::endl;

        std::vector<int> positive(K, 0);
        std::vector<int> negative(K, 0);
        for (int k=0; k<K;k++)  {
            for(int z=0; z<cluster[k].size(); z++)  {
                int trueclass = 0;
                if (data.GetOutput()[cluster[k][z]] == 1) {
                    trueclass = 1;
                }
                if (k == trueclass )  {
                    ++positive[k];
                } else {
                    ++negative[k];
                }
            }
        }

        std::cout << " positive1: " << positive[0] << " negative1: " << negative[0] << std::endl;
        std::cout << " positive2: " << positive[1] << " negative2: " << negative[1] << std::endl;

    }

}

void em(int init)  {
    int dataSize = 2;
    std::vector<GP_Vector> mu(dataSize, GP_Vector(2));
    std::vector<GP_Matrix> sigma(dataSize, GP_Matrix(2,2));
    std::vector<float> pi(dataSize);

    for(int i=0; i<dataSize; i++)  {
        for(int j=0; j<2; j++)  {
            for(int k=j; k<2; k++)  {
                sigma[i][j][k] = rand() % init;
            }
        }
        for(int j=0; j<2; j++)  {
            mu[i][j] = rand() % init;
        }
        pi[i] = rand() % init;
    }



}

float det(GP_Matrix mat)  {
//    if(mat.Col() != 2 && mat.Row() != 2)  {
//        return 0;
//    }
    return mat[0][0]*mat[1][1] - mat[1][0]*mat[0][1];
}

GP_Matrix inverse(GP_Matrix mat)  {
    float temp = mat[0][0];
    mat[0][0] = mat[1][1];
    mat[1][1] = temp;
    temp = -mat[0][1];
    mat[0][1] = -mat[1][0];
    mat[1][0] = temp;
    return mat/det(mat);
}

GP_Matrix gaussian(float x, float mu, float sigma)  {

}


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

  kmeans(data);
  em(2);

  return 0;
}

