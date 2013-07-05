#ifndef BINARY_CLASSIFIER_HH
#define BINARY_CLASSIFIER_HH

#include <vector>
#include <string.h>

#include "Classcode/GP_Vector.hh"
#include "Classcode/GP_DataSet.hh"
#include "Classcode/GP_Evaluation.hh"


namespace CLASSCODE {

  /*!
   * \class BinaryClassifier
   * Generic class of a binary classifier.
   */
  template <typename InputType>
  class BinaryClassifier
  {
  public:
  
    typedef GP_DataSet<InputType, int> DataSet;
    class TrainingParameters {};

    /*!
     * The constructor expects the feature index and a flag 
     * that specifies whether class labels are -1 and 1 or 0 and 1. 
     * The first case is the default.
     */
    BinaryClassifier(int feat_dim = -1, bool lower_class_label_neg = true) :
      LOWER_CLASS_LABEL(lower_class_label_neg ? -1 : 0), _feat_dim(feat_dim) {}

    virtual ~BinaryClassifier() {}

    /*!
     * Returns true if the class name is correct
     */
    virtual bool IsA(char const *classname) const
    {
      return (strcmp(classname, "BinaryClassifier") == 0);
    }

    /*!
     * Returns the class name
     */
    static std::string ClassName()
    {
      return "BinaryClassifier";
    }

    /*!
     * Returns the feature dimension
     */
    int GetFeatDim() const
    {
      return _feat_dim;
    }

    /*!
     * This method performs the traning for a given set of samples 
     * and corresponding weights. It is overridden by the derived classes.
     */
    virtual void Train(DataSet const &train_data,
		       TrainingParameters const &tparams) = 0;

    /*!
     * Performs the classification by returning the class label
     * of a new data point, given by its feature vector 'feat'.
     */
    virtual double Predict(InputType const &test_point) const = 0;

    
    virtual double GetTrainingError() const = 0;

    double GetUCC() const
    {
      return _ucc;
    }

    void CalcUCC(DataSet const &data, double thresh = 0.5, bool verbose = false)
    {
      double n0 = 0, n1 = 0, m0 = 0, m1 = 0, mu_z = 0, var_z = 0;
      
      for (uint i=0; i<data.Size(); ++i){
	double z = Predict(data.GetInput()[i]);
	double norment = GP_Evaluation::calc_norm_entropy(z);

	mu_z += norment;
	var_z += norment*norment;
	if (z >= thresh){ // classification "1"
	  if(data.GetOutput()[i] == 1){ // true label "1"
	    m0 += norment;
	    n0++;
	  }
	  else { // true label "-1"
	    m1 += norment;
	    n1++;
	  }
	}
	else { // classification "-1"
	  if(data.GetOutput()[i] == 1){ // true label "1"
	    m1 += norment;
	    n1++;
	  }
	  else { // true label "-1"
	    m0 += norment;
	    n0++;
	  }
	}
      }

      if(n0 >1e-15)
	m0 /= n0;
      if(n1 > 1e-15)
	m1 /= n1;

      mu_z  /= data.Size();
      var_z /= data.Size();
      var_z -= mu_z * mu_z;
      double n = n0+n1;
      double s_n = sqrt(MAX(var_z, 1e-15));
      

      _ucc = (m1 - m0) / s_n;
      
      /*
      if(n0 == 0 || n1 == 0)
      	_ucc = 0;
      else
      _ucc = (m1 - m0) / s_n * sqrt((n0 * n1) / (n * n));
      */
      //if(verbose)
      //	std::cout << "UCC " << mu_z << " " << var_z << " " << m0 << " " << m1 << " " << n0 << " " 
      //		  << n1 << " " << s_n << " " << n << " " << GetTrainingError() << " " << _ucc << std::endl;
    }

  protected:

    // this constant can be either 0 or -1, i.e. either the classifier
    // returns 0/1 or -1/1
    int LOWER_CLASS_LABEL;  

    virtual double CalcTrainingError(DataSet const &train_data) const
    {
      double wsum = 0;
      double misclas = 0;

      for(uint i=0; i<train_data.Size(); i++){
    
	int label = (this->Predict(train_data.GetInput()[i]) >= 0.5 ? 1 : 
		     LOWER_CLASS_LABEL);
    
	if(label != train_data.GetOutput()[i])
	  misclas++;

	wsum++;
      }					       

      return misclas / wsum;
    }



  private:
    
    int _feat_dim;
    double _ucc;
  };
}

#endif
