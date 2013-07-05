#ifndef GP_EVALUATION_HH
#define GP_EVALUATION_HH

#include <vector>
#include <string>

#include "Classcode/GP_DataSet.hh"

namespace CLASSCODE {
 


  class GP_Evaluation
  {
  public:

    static double f_measure(double beta, double tp, double fp, double fn);
    
    static double calc_norm_entropy(double prob);
    
    
    static void plot_pr_curve(std::vector<double> zvals,
			      std::vector<int> yvals,
			      std::string filename, double step_size);
    
    template<typename Classifier, typename InputType, typename OutputType>
    static void plot_pr_curve(Classifier const &classifier,
			      GP_DataSet<InputType, OutputType> const &test_data,
			      std::string filename, double step_size)
    {
      std::vector<double> zvals(test_data.Size());
      std::vector<OutputType> yvals(test_data.Size());
      
      for(uint i=0; i<test_data.Size(); i++){
	zvals[i] = classifier.Prediction(test_data.GetInput()[i]);
	yvals[i] = test_data.GetOutput()[i];
      }
      
      plot_pr_curve(zvals, yvals, filename, step_size);
    }

  };


}

#endif
