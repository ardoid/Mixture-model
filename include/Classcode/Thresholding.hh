#ifndef THRESHOLDING_HH
#define THRESHOLDING_HH

#include "Classcode/BinaryClassifierWithWeights.hh"

namespace CLASSCODE {

  /*!
   * \class Thresholding
   * Implements a simple binary classifier which determines
   * a threshold that optimally separates the training data
   * along one feature dimension.
   */
  class Thresholding : public BinaryClassifierWithWeights<double>
  {
  
  public:

    typedef BinaryClassifierWithWeights<double> Super;
    typedef Super::DataSet DataSet;
    typedef Super::TrainingParameters TrainingParameters;


    /*!
     * The constructor can be called with initial values for
     * the decision threshold, the sign, the featuer dimension,
     * and the lower class label flag.
     */
    Thresholding(bool lower_class_label_neg = true,
		 double thresh = 0, bool neg = false) :
      Super(1, lower_class_label_neg), 
      _thresh(thresh), _neg(neg), _sorted()
    {}
  
    virtual ~Thresholding() 
    {
      _sorted.clear();
    }

    /*!
     * Returns true if the class name is correct
     */    
    virtual bool IsA(char const *classname) const
    {
      return (strcmp(classname, "Thresholding") == 0);
    }

    /*!
     * Returns the class name
     */
    static std::string ClassName()
    {
      return "Thresholding";
    }

    /*!
     * Performs the training
     */
    virtual void Train(DataSet const &train_data,
		       TrainingParameters const &weights);

    /*!
     * Computes the distance from the decision boundary
     */    
    virtual double ComputeDBDist(double f) const;

    /*!
     * Performs the classification by returning the class label
     * of a new data point, given by its feature vector 'feat'.
     */
    virtual double Predict(double const &test_feat) const;

    /*!
     * Returns the decision threshold 
     */
    double GetThresh() const
    {
      return _thresh;
    }

    /*!
     * Returns the sign of the classifier (false for positive)
     */
    bool GetNeg() const
    {
      return _neg;
    }

    virtual double GetTrainingError() const
    {
      return _train_error;
    }

  private:
  
    std::pair<double, double> GetIntersection(double mu1, double sigma1,
					      double mu2, double sigma2) const;

    double  _thresh;
    bool _neg;
    std::vector<std::pair<double, uint> > _sorted;
    double _train_error;
  };

}

#endif
