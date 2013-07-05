#ifndef ADA_BOOST_HH
#define ADA_BOOST_HH

#include "Classcode/GP_Vector.hh"
#include "Classcode/Thresholding.hh"

#define BOOST_PREC 1e-12

namespace CLASSCODE {

  /*!
   * \class AdaBoost
   * Implements the standard binary AdaBoost classifier.
   */
  template <typename WeakClassifier = Thresholding>
  class AdaBoost : public BinaryClassifierWithWeights<GP_Vector>
  {
  
  public:

    typedef BinaryClassifierWithWeights<GP_Vector> Super; 
    typedef typename Super::DataSet DataSet;
    typedef typename Super::TrainingParameters TrainingParameters;

    /*!
     * Default constructor
     */
    AdaBoost() :
      Super(), _nb_stumps(0),
      _weak_clsf(), _epsilon(), _alphas()
    {}
      
    AdaBoost(uint nb_stumps, int feat_dim = -1, bool lower_class_label_neg = true) :
      Super(feat_dim, lower_class_label_neg),
      _nb_stumps(nb_stumps),
      _weak_clsf(), _epsilon(), _alphas()
    {}

    /*!
     * Returns true if the class name is correct
     */
    virtual bool IsA(char const *classname) const
    {
      return (strcmp(classname, "AdaBoost") == 0);
    }

    /*!
     * Returns the class name
     */
    static std::string ClassName()
    {
      return "AdaBoost";
    }


    void Train(DataSet const &train_data)
    { 
      TrainingParameters tparams;
      tparams._weights.resize(train_data.Size());
      for(uint i=0; i<tparams.size(); ++i)
	tparams._weights[i] = -log(tparams.size());
	//tparams._weights[i] = 1./tparams.size();

      Train(train_data, tparams);
    }

    virtual void Train(DataSet const &train_data,
		       TrainingParameters const &init_weights)
    { 
      TrainingParameters weights = init_weights;
      _weak_clsf.clear();
      _epsilon.clear();
      _dims.clear();
      _alphas.clear();

      uint dim;
      for(uint i=0; i<_nb_stumps; i++){

	_weak_clsf.push_back(WeakLearn(train_data, weights, dim));
	_epsilon.push_back(_weak_clsf[i].GetTrainingError());
	_dims.push_back(dim);

	if(_epsilon.back() >= 0.5){
	  std::cerr << "Error rate too high! " << _epsilon.back() << std::endl;
	  for(uint j=0; j<weights.size(); ++j)
	    std::cerr << weights[j] << " ";
	  std::cerr << std::endl;
	  break;
	}
	
	if(_epsilon.back() < BOOST_PREC)
	  _epsilon[_epsilon.size() - 1] = BOOST_PREC;
	
	// Compute alpha
	_alphas.push_back(log((1. - _epsilon.back()) / _epsilon.back()));
	
	// Update the data weighting coeffients
	UpdateWeights(i, train_data, _alphas.back(), _dims.back(), weights);
	
	printf("% 5.2lf%%\r", (double)i / _nb_stumps * 100.0);
	fflush(stdout);
      }

      _train_error = BinaryClassifier<GP_Vector>::CalcTrainingError(train_data);
      Super::CalcUCC(train_data, 0.5, true);
    }

    /*!
     * Computes the distance from the decision boundary
     */        
    double ComputeDBDist(GP_Vector const &test_feat) const
    {
      double alphasum = 0, sum = 0;

      for(uint i=0; i<_weak_clsf.size(); i++){
	alphasum += _alphas[i];
	sum += _alphas[i] * _weak_clsf[i].ComputeDBDist(test_feat);
      }
      
      return sum / alphasum;
    }

    /*!
     * Performs the classification by returning the class label
     * of a new data point, given by its feature vector 'feat'.
     */
    double Predict(GP_Vector const &test_feat) const
    {
      double alphasum = 0, sum = 0;

      for(uint i=0; i<_weak_clsf.size(); i++){
	alphasum += _alphas[i];
	
	if(_weak_clsf[i].Predict(test_feat[_dims[i]]) > 0.5)
	  sum += _alphas[i];
      }

      if(alphasum < BOOST_PREC)
	return 0;
      
      return sum  / alphasum;
    }

    uint GetNbStumps() const
    {
      return _nb_stumps;
    }

    double GetTrainingError() const
    {
      return _train_error;
    }

  private:

    uint _nb_stumps;
    DataSet _train_data;
    double _train_error;
    std::vector<WeakClassifier> _weak_clsf;
    std::vector<double> _epsilon;
    std::vector<uint>   _dims;
    std::vector<double> _alphas;

    virtual WeakClassifier WeakLearn(DataSet const &train_data,
				     TrainingParameters const &weights,
				     uint &opt_dim) const
    {
      double min_loss = HUGE_VAL, loss, ucc = HUGE_VAL;
      WeakClassifier opt_clsf;

      uint n = train_data.Size(), argmin;
      std::vector<double> projected_input(n);

      
      for(uint dim=0; dim < Super::GetFeatDim(); dim++){
      
	for(uint i=0; i<n; ++i)
	  projected_input[i] = train_data.GetInput()[i][dim];

	GP_DataSet<double, int> proj_data;
	proj_data.Append(projected_input, train_data.GetOutput());

	WeakClassifier weak_clsf((Super::LOWER_CLASS_LABEL < 0));

	typename WeakClassifier::TrainingParameters tparams(weights._weights);

	weak_clsf.Train(proj_data, tparams);
	loss = weak_clsf.GetTrainingError();

	if(loss < min_loss){
	  min_loss = loss;
	  opt_dim = dim;

	  // We got a new optimal classifier
	  opt_clsf = weak_clsf;
	  argmin = dim;
	}
      }

      return opt_clsf;
    }

    void UpdateWeights(uint clsf_idx, DataSet const &train_data,
		       double alpha, uint dim,
		       TrainingParameters &weights) const
    {
      if(weights.size() != train_data.Size())
	throw GP_EXCEPTION("Invalid number of weights");
      
      for(uint i=0; i<train_data.Size(); i++){

	double z = _weak_clsf[clsf_idx].Predict(train_data.GetInput()[i][dim]);
	int label = z > 0.5 ? 1 : Super::LOWER_CLASS_LABEL;

	if(label != train_data.GetOutput()[i]){
	  weights[i] += alpha;
	}
      }					       
    }


  };

}

#endif
