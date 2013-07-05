
#ifndef GP_SIGMOID_FUNCTION_HH
#define GP_SIGMOID_FUNCTION_HH

#include "Classcode/GP_Constants.hh"

#include <gsl/gsl_math.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_integration.h>
#include <math.h>
#include <string>

namespace CLASSCODE {

  /*!
   * Base class for sigmoid functions
   */
  class GP_SigmoidFunction
  {
  public:

    virtual double operator()(double x) const = 0;
  
    // Numerical derivative
    virtual double Deriv(double x) const;

    // Integrates the function from 0 to x numerically
    virtual double Integrate(double x);

    // writes x and y values to a file
    void Plot(char const *filename, double from = -1.0, 
	      double to = 1.0, double step = 0.01);
  
    virtual ~GP_SigmoidFunction() {};

  private:

    static double integrand(double z, void *params);
  };

  /*!
   * The logit-sigmoid function
   */

  class GP_LogisticSigmoid : public GP_SigmoidFunction
  {

  public:

    static std::string ClassName() 
    {
      return "GP_LogisticSigmoid";
    }

    double operator()(double x) const;
  
    double Deriv(double x) const;

    double Integrate(double x) const;

    double LogDeriv(double x) const;

    // log-likelihood for the binary case
    double LogLikelihoodBin(double y, double f) const;

    // derivative of the log-likelihood for the binary case
    double LogLikelihoodBinDeriv(double y, double f) const;

    // 2nd derivative of the log-likelihood for the binary case
    double LogLikelihoodBin2ndDeriv(double y, double f) const;

    static double integrand(double z, void *params);
    
    // Computes \Int sigma(x) * Gauss(x | mean, cov) dx 
    double IntegrateWithGaussian(double mean, double cov) const;
  };

  class GP_CumulativeGaussian : public GP_SigmoidFunction
  {

  public:

    static std::string ClassName() 
    {
      return "GP_CumulativeGaussian";
    }

    double operator()(double x) const;

    double Deriv(double x) const;

    double Log(double x) const;

    double LogDeriv(double x) const;

    /*!
     * returns the inverse of the cumulative Gaussian
     */ 
    double Inv(double x) const;

    double LogLikelihoodBin(double y, double f, double lambda = 1., 
			    double sigma = 0., double bias = 0.) const;

    double LogLikelihoodBinDeriv(double y, double f) const;

    double LogLikelihoodBin2ndDeriv(double y, double f) const;

    // Computes \Int sigma(x) * Gauss(x | mean, cov) dx 
    double IntegrateWithGaussian(double mean, double cov) const;
  };

}

#endif
