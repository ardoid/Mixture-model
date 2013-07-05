#include <algorithm>
#include <iomanip>
#include "Classcode/Thresholding.hh"

using namespace std;
using namespace CLASSCODE;

class CompareLessKey
{
  
public:
  
  bool operator()(std::pair<double, uint> const &p1, 
		  std::pair<double, uint> const &p2)
  {
    return (p1.first < p2.first);
  }
};



void Thresholding::Train(DataSet const &train_data,
			 TrainingParameters const &weights)
{  
  if(train_data.Size() != weights.size())
    throw GP_EXCEPTION("Number of weights does not match number of training points");

  vector<double> class1(train_data.Size(), 0.0), class2(train_data.Size(), 0.0);

  double max_weight = 0;
  for(uint i=0; i<weights.size(); ++i)
    max_weight = MAX(weights[i], max_weight);

  double lsum = 0, wsum = 0;
  for(uint i=0; i<train_data.Size(); i++){
    if(train_data.GetOutput()[i] == LOWER_CLASS_LABEL)
      lsum += exp(weights[i] - max_weight);
    wsum += exp(weights[i] - max_weight);
  }

  if(_sorted.size() == 0){
    _sorted.resize(train_data.Size());

    for(uint i=0; i<train_data.Size(); i++)
      _sorted[i] = make_pair(train_data.GetInput()[i], i);

    sort(_sorted.begin(), _sorted.end(), CompareLessKey());
  }

  double error1, error2, min_error = 1.0;
  uint argmin = 0;
  bool negmin = false;
  for(uint i=0; i<_sorted.size(); i++){

    int label = train_data.GetOutput()[_sorted[i].second];
    double weight = exp(weights[_sorted[i].second] - max_weight);

    if(label == 1)
      if(i==0){
	class1[0] = 0;
	class2[0] = weight;
      }
      else{
	class1[i] = class1[i-1];
	class2[i] = class2[i-1] + weight;
      }

    else {

      if(i==0){
	class1[0] = weight;
	class2[0] = 0;
      }
      else{
	class1[i] = class1[i-1] + weight;
	class2[i] = class2[i-1];
      }
    }

    // error1 is the misclass rate for positive sign, error2 for neg
    if(i==0)
      error1 = lsum / wsum;
    else
      error1 = (class2[i-1] + lsum - class1[i-1]) / wsum;

    error2 = 1.0 - error1;
    if(label == 1)
      error2 -= weight / wsum;
    else
      error2 += weight / wsum;

    if(error1 < min_error){
      min_error = error1;
      argmin = i;
      negmin = false;
    }
    if(error2 < min_error){
      min_error = error2;
      argmin = i;
      negmin = true;
    }
  }

  _neg = negmin;
  _thresh = _sorted[argmin].first;
  _train_error = CalcTrainingError(train_data, weights);
  Super::CalcUCC(train_data);
}

  
double Thresholding::ComputeDBDist(double f) const
{
  int sign = _neg ? -1 : 1;

  return sign * (f - _thresh);
}


double Thresholding::Predict(double const &test_feat) const
{
  double db_dist = ComputeDBDist(test_feat);

  if(db_dist < 0)
    return 0;

  else
   return 1;  
}


