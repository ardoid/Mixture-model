#include <vector>
#include <iostream>
#include <fstream>

#include "Classcode/GP_Constants.hh"
#include "Classcode/GP_CovarianceFunction.hh"
#include "Classcode/GP_DataSet.hh"

#include "Classcode/GP_InputParams.hh"
#include "Classcode/GP_Evaluation.hh"
#include "Classcode/GP_BinaryClassificationEP.hh"

using namespace std;
using namespace CLASSCODE;

typedef std::vector<double> InputType;
typedef int OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;
template <typename T> class KernelType : public GP_SquaredExponential<T> {};
typedef GP_BinaryClassificationEP<InputType, KernelType> EPClassifier;
typedef KernelType<InputType>::HyperParameters HyperParameters;


BEGIN_PROGRAM(argc, argv)
{
  if(argc < 3) {
    std::cerr << "usage:" << argv[0] << " <config-file> <dataset.dat>" << std::endl;
    exit(1);
  }
   
  gsl_rng_env_setup();

  // Read the program options from the config file
  GP_InputParams params;
  params.Read(std::string(argv[1]));
  params.Write("params.txt");

  // Read training and test data

  std::cout << "reading data from " << argv[1] << "..." << std::endl;
  DataSetType train_data;
  train_data.Read(std::string(argv[2]));

  DataSetType test_data;
  test_data = train_data.DownSample(1. - params.train_frac);

  std::cout << "training data size: " << train_data.Size() << std::endl;
  std::cout << "test data size: " << test_data.Size() << std::endl;
  
  train_data.Write("training_data.dat");

  // Initialize EP
  HyperParameters hparams = params.GetHyperParamsInit();

  std::cout << "Instantiating EP with kernel parameters ";
  for(uint j=0; j<hparams.Size(); ++j)
    std::cout << hparams.ToVector()[j] << " " << std::flush;
  std::cout << std::endl;

  EPClassifier classif_ep(train_data, hparams, params.lambda);
    
  // Train the hyperparameters
  if(params.do_optimization){
    std::cout << "training hyper parameters... " << std::endl;
    classif_ep.LearnHyperParameters(hparams, params.kparam_lower_bounds,
				    params.kparam_upper_bounds,
				    params.nb_iterations);
  }
  else
    classif_ep.Estimation();
  
  // We run through the test data set and classify them
  WRITE_FILE(cfile, "classification.dat");
  std::vector<double> zvals;
  std::vector<OutputType> yvals;
  
  double tp = 0, tn = 0, fp = 0, fn = 0, pos = 0, neg = 0;
  for(uint test_idx = 0; test_idx < test_data.Size(); ++test_idx){
    
    InputType x   = test_data.GetInput()[test_idx];
    OutputType y  = test_data.GetOutput()[test_idx];

    double mu_star, sigma_star;
    double z = classif_ep.Prediction(x, mu_star, sigma_star);
    
    cfile << test_idx << " " << z << " " << y << std::endl;
    
    zvals.push_back(z);
    yvals.push_back(y);
    
    if(z >= 0.5 && y == 1){
      // true positive
      tp++;
    }
    else if(z >= 0.5 && y == -1){
      // false positive
      fp++;
    }
    else if(z < 0.5 && y == 1){
      // false negative
      fn++;
    }
    else if(z < 0.5 && y == -1){
      // true negative
      tn++;
    }
    
    if(y == 1)
      pos++;
    else
      neg++;
  }
  
  // evaluate the classification
  double prec, rec, fmeas;
  if(tp + fp == 0)
    prec = 1.0;
  else
    prec = tp / (tp + fp);
  
  if(tp + fn == 0)
    rec = 1.0;
  else
    rec = tp / (tp + fn);
  
  fmeas = (1 + SQR(0.5)) * prec * rec / (SQR(0.5) * prec + rec);
  
  double corr = tp + tn;
  double incorr = fn + fp;
  
  std::cout << pos << " " << neg  << " pos neg" << std::endl;
  std::cout << tp << " " << fp << " " << fn << " " << tn << " tp fp fn tn" << std::endl;
  std::cout << prec << " " << rec << " " << fmeas << " prec rec fmeas" << std::endl;
  std::cout << corr << " correct classifications" << std::endl;
  std::cout << incorr << " incorrect classifications" << std::endl;
  std::cout << corr / (corr + incorr) << " classification rate" << std::endl;
  
  GP_Evaluation::plot_pr_curve(zvals, yvals, "prec_rec.dat", 0.05);  

}

END_PROGRAM


