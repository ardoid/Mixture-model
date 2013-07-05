#ifndef BINARY_CLASSIFIER_WITH_WEIGHTS_HH
#define BINARY_CLASSIFIER_WITH_WEIGHTS_HH

#include <vector>
#include "Classcode/BinaryClassifier.hh"

namespace CLASSCODE {

  /*!
   * \class BinaryClassifierWithWeights
   * Generic class of a binary classifier with weights.
   */
  template <typename InputType>
  class BinaryClassifierWithWeights : public BinaryClassifier<InputType>
  {
  public:
  
    typedef BinaryClassifier<InputType> Super;
    typedef typename Super::DataSet DataSet;

    struct TrainingParameters : public Super::TrainingParameters
    {
      TrainingParameters() : _weights() {}
      TrainingParameters(typename Super::TrainingParameters const &tp) : _weights() {}

      std::vector<double> _weights;
      TrainingParameters(std::vector<double> const &w) : _weights(w) {}
      const double &operator[](uint i) const
      {
	return _weights[i];
      }
      double &operator[](uint i)
      {
	return _weights[i];
      }
      uint size() const
      {
	return _weights.size();
      }
    };


    BinaryClassifierWithWeights() :
      Super(0, true) {}

    /*!
     * The constructor expects the feature index and a flag 
     * that specifies whether class labels are -1 and 1 or 0 and 1. 
     * The first case is the default.
     */
    BinaryClassifierWithWeights(int feat_dim, bool lower_class_label_neg = true) :
      Super(feat_dim, lower_class_label_neg) {}

    virtual ~BinaryClassifierWithWeights() {}

    /*!
     * Returns true if the class name is correct
     */
    virtual bool IsA(char const *classname) const
    {
      return (Super::IsA(classname) ||
	      strcmp(classname, "BinaryClassifierWithWeights") == 0);
    }

    /*!
     * Returns the class name
     */
    static std::string ClassName()
    {
      return "BinaryClassifierWithWeights";
    }


    virtual void Train(DataSet const &train_data,
		       typename Super::TrainingParameters const &tparams)
    {
      TrainingParameters weights(tparams);
      Train(train_data, weights);
    }

    virtual void Train(DataSet const &train_data,
		       TrainingParameters const &init_weights) = 0;


  protected:

    double CalcTrainingError(DataSet const &train_data,
			     TrainingParameters const &weights) const
    {
      if(weights.size() != train_data.Size())
	throw GP_EXCEPTION("Invalid number of weights");

      double wsum = 0;
      double misclas = 0;

      for(uint i=0; i<train_data.Size(); i++){
    
	int label = (this->Predict(train_data.GetInput()[i]) >= 0.5 ? 1 : 
		     Super::LOWER_CLASS_LABEL);
    
	if(label != train_data.GetOutput()[i]){
	  misclas += exp(weights[i]);
	}

	wsum += exp(weights[i]);
      }					       

      //std::cout << "terr " << misclas << " " << wsum << std::endl;

      return misclas / wsum;
    }



  };
}

#endif
