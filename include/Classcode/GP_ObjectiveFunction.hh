#ifndef GP_OBJECTIVE_FUNCTION_HH
#define GP_OBJECTIVE_FUNCTION_HH

#include "Classcode/GP_Vector.hh"
#include "Classcode/GP_ObjectiveFunction.hh"
#include "Classcode/GP_CovarianceFunction.hh"

namespace CLASSCODE {

  /*!
   * \class GP_ObjectiveFunctionBase 
   *
   * \brief Base class for GP objective functions. 
   *
   * This abstract class only provides the interfaces of the objective function
   */
  class GP_ObjectiveFunctionBase 
  {
  public:
    
    GP_ObjectiveFunctionBase() {}
    virtual ~GP_ObjectiveFunctionBase() {}

    virtual uint GetNbArgs() const = 0;
    virtual double operator()(std::vector<double> const &args) const = 0;
    virtual double Deriv(std::vector<double> const &args, uint arg_idx) const = 0;
    virtual std::pair<double, GP_Vector> ValAndDeriv(std::vector<double> const &args) const = 0;
  };

  /*!
   * \class GP_ObjectiveFunction 
   *
   * \brief Objective function to be optimized in GP training
   * 
   * This is the functor that represents the objective function used to 
   * optimize the kernel parameters for the binary EP classification.
   */
  template<typename InputType, typename ClassificationAlgorithm>
  class GP_ObjectiveFunction : public GP_ObjectiveFunctionBase
  {
  public:

    typedef typename ClassificationAlgorithm::KernelType       KernelType;
    typedef typename ClassificationAlgorithm::HyperParameters  HyperParameters;
    
    /*!
     * The constructor needs the GP structure, a number of kernel parameters,
     * and a lower and an upper bound for the parameters.  Both the lower and
     * the upper bound(s)  must be positive or zero,  the upper bounds can be
     * infinite, in which case the parameters are not bounded from above.
     */
    GP_ObjectiveFunction(ClassificationAlgorithm &gp, uint nb_params, 
			 GP_Vector const &lower_bound = GP_Vector(),
			 GP_Vector const &upper_bound = GP_Vector()) :
      _classif(gp), _lower_bound(nb_params), _upper_bound(nb_params) 
    {
      if(lower_bound.Size() == 1){
	_lower_bound = GP_Vector(nb_params, lower_bound[0]);
      }
      else if(lower_bound.Size() == nb_params){
	_lower_bound = lower_bound;
      }
      else{
	_lower_bound = GP_Vector(nb_params);
      }

      if(upper_bound.Size() == 1){
	_upper_bound = GP_Vector(nb_params, upper_bound[0]);
      }
      else if(upper_bound.Size() == nb_params){
	_upper_bound = upper_bound;
      }
      else{
	_upper_bound = GP_Vector(nb_params);
      }
    }
    
    /*!
     * Returns the number of arguments of the objective function, i.e. the 
     * number of kernel parameters
     */
    virtual uint GetNbArgs() const
    {
      return _lower_bound.Size();
    }

    /*!
     * This is the main funtion.  It sets the given vector of 'args' as kernel
     * parameters for the underlying GP and re-runs the estimation. The return
     * value is the negative log marginal likelihood of the GP classifier.
     */
    virtual double operator()(std::vector<double> const &args) const
    {
      HyperParameters hyp(args);
      
      _classif.UpdateModelParameters(hyp.Transform(_lower_bound, _upper_bound));

      return -_classif.GetLogZ();
    }
    
    
    /*!
     * Computes the partial derivative of the objective function (i.e. the log
     * marginal likelihood) with respect to the kernel parameters with the given
     * index.
     */
    virtual double Deriv(std::vector<double> const &args, uint arg_idx) const
    {
      HyperParameters hyp(args);
      _classif.UpdateModelParameters(hyp.Transform(_lower_bound, _upper_bound));
      
      return -_classif.GetDeriv()[arg_idx] * hyp.TransformDeriv(args[arg_idx],
							      _lower_bound[arg_idx], 
							      _upper_bound[arg_idx]);
    }
    
    /*!
     * Computes both the value of the objective function and its derivative
     * with respect to all kernel parameters. This is sometimes more efficient
     * than computing function value and derivative separately.
     */
    std::pair<double, GP_Vector> ValAndDeriv(std::vector<double> const &args) const
    {
      HyperParameters hyp(args);
      _classif.UpdateModelParameters(hyp.Transform(_lower_bound, _upper_bound));
      
      GP_Vector deriv = _classif.GetDeriv();
      for(uint i=0; i<deriv.Size(); ++i)
	deriv[i] *= -hyp.TransformDeriv(args[i], _lower_bound[i],_upper_bound[i]);
      
      return std::make_pair(-_classif.GetLogZ(), deriv);
    }
    
  private:
    
    ClassificationAlgorithm &_classif;
    GP_Vector _lower_bound, _upper_bound;
  };

}

#endif
