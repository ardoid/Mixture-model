#include <gsl/gsl_math.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_integration.h>
#include <math.h>
#include <fstream>


#include "Classcode/GP_Constants.hh"
#include "Classcode/GP_Exception.hh"
#include "Classcode/GP_SigmoidFunction.hh"

namespace CLASSCODE {

  
  double GP_SigmoidFunction::Deriv(double x) const 
  {
    double h = GSL_SQRT_DBL_EPSILON;
    double a[4], d[4], a3;
    
    for (uint i = 0; i < 4; i++){
      
      a[i] = x + (i - 2.0) * h;
      d[i] = this->operator()(a[i]);
    }

    for (uint k = 1; k < 5; k++)
      for (uint i = 0; i < 4 - k; i++)
	d[i] = (d[i + 1] - d[i]) / (a[i + k] - a[i]);
    
    a3 = fabs (d[0] + d[1] + d[2] + d[3]);
    
    if (a3 < 100.0 * GSL_SQRT_DBL_EPSILON)
      a3 = 100.0 * GSL_SQRT_DBL_EPSILON;
    
    h = pow (GSL_SQRT_DBL_EPSILON / (2.0 * a3), 1.0 / 3.0);
    
    if (h > 100.0 * GSL_SQRT_DBL_EPSILON)
      h = 100.0 * GSL_SQRT_DBL_EPSILON;
    
    double x1 = x + h;
    double y1 = this->operator()(x1);
    
    double x2 = x - h;
    double y2 = this->operator()(x2);
    
    return (y1 - y2) / (2.0 * h);
  }

  double GP_SigmoidFunction::Integrate(double x)
  {
    gsl_integration_workspace *w =
      gsl_integration_workspace_alloc(1000);

    gsl_function F;
    F.function = &integrand;
    F.params   = this;

    double result, error;
    gsl_integration_qag(&F, 0, x, 0, 1e-7, 1000, 5, w, &result, &error);

    gsl_integration_workspace_free(w);

    return result;
  }

  void GP_SigmoidFunction::Plot(char const *filename, double from,
				double to, double step)
  {
    WRITE_FILE(ofile, filename);
    for(double x=from; x<= to; x+= step)
      ofile << x << " " << this->operator()(x) << std::endl;
    ofile.close();
  }
  
  double GP_SigmoidFunction::integrand(double z, void *params)
  {
    GP_SigmoidFunction *sig_func = (GP_SigmoidFunction*)params;
    
    return (*sig_func)(z);
  }


  double GP_LogisticSigmoid::operator()(double x) const 
  {
    return (1.0 / (1.0 + exp(-x)));
  }
  
  
  double GP_LogisticSigmoid::Deriv(double x) const
  {
    double exp_min_x = exp(-x);
    
    return (exp_min_x / SQR(1.0 + exp_min_x));
  }

  double GP_LogisticSigmoid::Integrate(double x) const
  {
    return (x - LOG2 + ::log(1.0 + exp(-x)));
  }

  double GP_LogisticSigmoid::LogDeriv(double x) const
  {
    double exp_min_x = exp(-x);

    return (exp_min_x / (1.0 + exp_min_x));
  }

  double GP_LogisticSigmoid::LogLikelihoodBin(double y, double f) const
  {
    return -::log(1.0 + exp(-y * f));
  }

  double GP_LogisticSigmoid::LogLikelihoodBinDeriv(double y, double f) const
  {
    return (y + 1.0)/2.0 - this->operator()(f);
  }

  double GP_LogisticSigmoid::LogLikelihoodBin2ndDeriv(double y, double f) const
  {
    double p = this->operator()(f);
    return -p * (1.0 - p);
  }

  double GP_LogisticSigmoid::integrand(double z, void *params)
  {
    double mean = ((double*)params)[0];
    double var  = ((double*)params)[1];
    double normalizer = sqrt(2. * M_PI * var);

    GP_LogisticSigmoid sig_func;
    
    // Mulitply sigm(z) with Gauss 
    return sig_func(z) * (exp(SQR(mean - z) / (-2.0 * var)) / normalizer);
  }

  // Computes \Int sigma(x) * Gauss(x | mean, cov) dx 
  double GP_LogisticSigmoid::IntegrateWithGaussian(double mean, double cov) const
  {
    double mean_cov [2];
    mean_cov[0] = mean;
    mean_cov[1] = cov;

    gsl_integration_workspace *w =
      gsl_integration_workspace_alloc(1000);

    gsl_function F;
    F.function = &integrand;
    F.params   = mean_cov;

    double result, error;
    gsl_integration_qagi(&F, 0, 1e-7, 1000, w, &result, &error);

    gsl_integration_workspace_free(w);

    return result;
  }

  double GP_CumulativeGaussian::operator()(double x) const
  {
    return gsl_sf_erfc(-x / SQRT2) / 2.0;
  }

  double GP_CumulativeGaussian::Deriv(double x) const
  {
    // Gaussian with zero mean and covariance 1
    return gsl_ran_ugaussian_pdf(x);
  }

  double GP_CumulativeGaussian::Log(double x) const
  {
    return gsl_sf_log_erfc(-x / SQRT2) - LOG2;
  }

  double GP_CumulativeGaussian::LogDeriv(double x) const
  {
    return -(LOG2PI + SQR(x)) / 2.0;
  }

  double GP_CumulativeGaussian::Inv(double x) const
  {
    return gsl_cdf_ugaussian_Pinv(x);
  }

  double GP_CumulativeGaussian::LogLikelihoodBin(double y, double f, double lambda, 
						 double sigma, double bias) const
  {
    return Log(y * (f + bias) / sqrt(1./SQR(lambda) + sigma));
  }

  double GP_CumulativeGaussian::LogLikelihoodBinDeriv(double y, double f) const
  {
    return y * Deriv(f) / this->operator()(y * f);
  }

  double GP_CumulativeGaussian::LogLikelihoodBin2ndDeriv(double y, double f) const
  {
    double frac =  Deriv(f) / this->operator()(y*f);

    return -SQR(frac) - y * f * frac;
  }

  double GP_CumulativeGaussian::IntegrateWithGaussian(double mean, double cov) const
  {
    return this->operator()(mean / sqrt(1.0 + cov));
  }

}

