#include "Classcode/GP_Optimizer.hh"

namespace CLASSCODE {

  double fr_min_error_func (const gsl_vector * x, void *params)
  {
    GP_ObjectiveFunctionBase *ef = static_cast<GP_ObjectiveFunctionBase*>(params);
  
    std::vector<double> args(ef->GetNbArgs());
  
    for (uint i=0; i<ef->GetNbArgs(); i++){
      args[i] = gsl_vector_get(x, i);
    }
  
    return (*ef)(args);
  }

  void fr_min_error_deriv (const gsl_vector * x, void *params, gsl_vector *df)
  {
    GP_ObjectiveFunctionBase *ef = static_cast<GP_ObjectiveFunctionBase*>(params);
  
    std::vector<double> args(ef->GetNbArgs());

    for (uint i=0; i<ef->GetNbArgs(); i++)
      args[i] = gsl_vector_get(x, i);

    for(uint i=0; i<ef->GetNbArgs(); i++)
      gsl_vector_set (df, i, ef->Deriv(args, i));
  }

  void fr_min_funcs (const gsl_vector * x, void *params,
		     double *f, gsl_vector *df)
  {
    GP_ObjectiveFunctionBase *ef = static_cast<GP_ObjectiveFunctionBase*>(params);

    std::vector<double> args(ef->GetNbArgs());
    for (uint i=0; i<ef->GetNbArgs(); i++)
      args[i] = gsl_vector_get(x, i);

    std::pair<double, GP_Vector> fdf = ef->ValAndDeriv(args);

    for(uint i=0; i<ef->GetNbArgs(); i++){
      gsl_vector_set (df, i, fdf.second[i]);
    }

    *f = fdf.first;
  }
  

  GP_Optimizer::GP_Optimizer(GP_ObjectiveFunctionBase &fn, OptimizerType type , 
			     double step, double tol, double eps) :
    _step_size(step), _tolerance(tol), _conv_thresh(eps), _objFn(fn), 
    _func(), _minimizer(0), _x(gsl_vector_alloc(fn.GetNbArgs()))
  {
    if(_x == 0)
      _x = gsl_vector_alloc(fn.GetNbArgs());

    _func.n      = _objFn.GetNbArgs();
    _func.f      = &fr_min_error_func;
    _func.df     = &fr_min_error_deriv;
    _func.fdf    = &fr_min_funcs;
    _func.params = &_objFn;
    
    if(type == FR)
      _minimizer = 
	gsl_multimin_fdfminimizer_alloc (gsl_multimin_fdfminimizer_conjugate_fr, _func.n);
    else if (type == PR)
      _minimizer = 
	gsl_multimin_fdfminimizer_alloc (gsl_multimin_fdfminimizer_conjugate_pr, _func.n);
    else if (type == LBFGS)
      _minimizer = 
	gsl_multimin_fdfminimizer_alloc (gsl_multimin_fdfminimizer_vector_bfgs2, _func.n);
    else
      throw GP_EXCEPTION("Unknown optimizer type.");
  }
  
  GP_Optimizer::~GP_Optimizer()
  {
    gsl_multimin_fdfminimizer_free (_minimizer);
    gsl_vector_free(_x);
    _x = 0;
  }
  
  void GP_Optimizer::Init(std::vector<double> const &init)
  {
    if (init.size() != _func.n)
      throw GP_EXCEPTION("Length of init array does not match number"
			 "of function arguments!");

    for (uint i=0; i<init.size(); i++){
      gsl_vector_set(_x, i, init[i]);
    }
  
    gsl_multimin_fdfminimizer_set(_minimizer, &_func, _x, _step_size, _tolerance);
  }
  
  bool GP_Optimizer::Iterate()
  {
    return (gsl_multimin_fdfminimizer_iterate(_minimizer) != GSL_ENOPROG);
  }
  
  void GP_Optimizer::GetCurrValues(std::vector<double> &vals) const
  {
    vals = std::vector<double> (_func.n);

    for(uint i=0; i<vals.size(); i++)
      vals[i] = gsl_vector_get(_minimizer->x, i);
  }
  
  double GP_Optimizer::GetCurrentError() const
  {
    return _minimizer->f;
  }
  
  bool  GP_Optimizer::TestConvergence() const
  {
    return (gsl_multimin_test_gradient (_minimizer->gradient, _conv_thresh) == GSL_SUCCESS);
  }
  
}
