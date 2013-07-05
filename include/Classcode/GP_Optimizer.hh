#ifndef GP_OPTIMIZER_HH
#define GP_OPTIMIZER_HH

#include <gsl/gsl_multimin.h>
#include "Classcode/GP_ObjectiveFunction.hh"

namespace CLASSCODE {

  /*!
   * Opimizer class
   *
   * This class optimizes the GP objective function, i.e. it is 
   * the core of GP hyper parameter training
   */
  class GP_Optimizer
  {
    
  public:

    /*!
     * Three different optimization algorithms can be chosen
     */
    typedef enum {FR, PR, LBFGS} OptimizerType;

    /*!
     * The constructor needs an objective function and some parameters
     */
    GP_Optimizer(GP_ObjectiveFunctionBase &fn, OptimizerType type = PR, 
		 double step = 0.01, double tol = 1e-4, double eps = 0.001);

    /*!
     * Default destructor
     */
    ~GP_Optimizer();
  
    /*!
     * Initializes the optimizer
     */
    void Init(std::vector<double> const &init);
    
    /*!
     * Performs one iteration of the optimization. Returns true if improvement 
     * is still possible. Can be used to do:  while(m.Iterate()){}
     */
    bool Iterate();
    
    /*!
     * Returns the current arguments (= kernel params) of the objective function
     */
    void GetCurrValues(std::vector<double> &vals) const;

    /*!
     * Returns the current value of the objective function
     */
    double GetCurrentError() const;
    
    /*!
     * Checks whether the optimization has converged
     */
    bool  TestConvergence() const;

  private:

    friend double fr_min_error_func(const gsl_vector *x, void *params);
    friend void fr_min_error_deriv(const gsl_vector *x, void *params, gsl_vector *df);
    friend void fr_min_funcs(const gsl_vector *x, void *params, double *f, gsl_vector *df);

    double _step_size; // initial step size
    double _tolerance; // needed for stop criterion
    double _conv_thresh;

    GP_ObjectiveFunctionBase    &_objFn;
    gsl_multimin_function_fdf    _func;
    gsl_multimin_fdfminimizer*   _minimizer;
    gsl_vector *_x;
  };


}



#endif
