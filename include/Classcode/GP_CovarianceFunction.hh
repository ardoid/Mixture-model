
#ifndef GP_COVARIANCE_FUNCTION_HH
#define GP_COVARIANCE_FUNCTION_HH

#include <vector>
#include <fstream>
#include <cmath>

#include "Classcode/GP_Constants.hh"
#include "Classcode/GP_Exception.hh"
#include "Classcode/GP_Vector.hh"
#include "Classcode/GP_SigmoidFunction.hh"


namespace CLASSCODE {

/*!
 * Base class for kernel (=covariance) functions
 */
template <typename ArgType>
class GP_CovarianceFunction
{

public:

  typedef ArgType ArgumentType;

  /*!
   * The hyper-parameters of the kernel
   */
  class HyperParameters {

  public:

    /*!
     * Default constructor
     */
    HyperParameters() { }

#ifdef USE_BOOST_SERIALIZATION
    /*!
     * Serializes the object to an archive using boost::serialization
     */
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    { }
#endif

    /*!
     * Index operator
     */
    virtual double operator[](uint i) const = 0;

    /*!
     * Returns the number of hyper parameters
     */
    virtual uint Size() const = 0;

    /*!
     * returns the hyperparameters in an STL vector
     */
    virtual std::vector<double> ToVector() const = 0;
    /*!
     * assigns the hyperparameters from an STL vector
     */
    virtual void FromVector(std::vector<double> const &vec) = 0;
  };

};

template <typename ArgType>
class GP_SquaredExponential : public GP_CovarianceFunction<ArgType>
{

public:

  class HyperParameters :
public GP_CovarianceFunction<ArgType>::HyperParameters
{

public:

  typedef typename GP_CovarianceFunction<ArgType>::HyperParameters Super;

  /*!
   * Default constructor
   */
  HyperParameters() : sigma_f_sqr(1.0), length_scale(1.0), sigma_n_sqr(1.0) {}

  /*!
   * Constructs parameters with an STL vector. No range checking
   */
  HyperParameters(std::vector<double> const &vec) :
    Super(), sigma_f_sqr(1.0), length_scale(1.0), sigma_n_sqr(1.0)
  {
    FromVector(vec);
  }

#ifdef USE_BOOST_SERIALIZATION
  /*!
   * Serializes the object to an archive using boost::serialization
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    // serialize base class information
    ar & boost::serialization::base_object<Super>(*this);
    ar & sigma_f_sqr;
    ar & length_scale;
    ar & sigma_n_sqr;
  }
#endif

  double sigma_f_sqr;  // signal variance
  double length_scale; // length scale
  double sigma_n_sqr;  // noise variance

  virtual double operator[](uint i) const
  {
    if(i == 0)
      return sigma_f_sqr;
    else if(i == 1)
      return length_scale;
    else if(i == 2)
      return sigma_n_sqr;

    throw GP_EXCEPTION("Invalid index into kernel paramters");
  }

  virtual uint Size() const
  {
    return 3;
  }

  int Read(std::string filename, int pos = 0)
  {
    READ_FILE(ifile, filename.c_str());
    ifile.seekg(pos);
    ifile >> sigma_f_sqr >> length_scale >> sigma_n_sqr;
    return ifile.tellg();
  }

  void Write(std::string filename) const
  {
    APPEND_FILE(ofile, filename.c_str());
    ofile << " " << sigma_f_sqr << " " << length_scale
        << " " << sigma_n_sqr << std::endl;
  }

  /*!
   * returns the hyperparameters in an STL vector
   */
  virtual std::vector<double> ToVector() const
          {
    std::vector<double> vec(3);
    vec[0] = sigma_f_sqr;
    vec[1] = length_scale;
    vec[2] = sigma_n_sqr;

    return vec;
          }

  /*!
   * assigns the hyperparameters from an STL vector
   */
  virtual void FromVector(std::vector<double> const &vec)
  {
    if(vec.size() < Size()){
      std::cerr << vec.size() << " " << Size() << std::endl;
      throw GP_EXCEPTION("Invalid number of hyperparameters");
    }

    sigma_f_sqr  = vec[0];
    length_scale = vec[1];
    sigma_n_sqr  = vec[2];
  }

  /*!
   * We do the optimization in log-space to guarantee positive values
   * This function transforms the paramters back into the space where
   * they are used.
   */
  template <typename IndexedContainer>
  HyperParameters Transform(IndexedContainer const &lbounds,
      IndexedContainer const &ubounds) const
  {
    if(lbounds.size() != Size() ||
        ubounds.size() != Size())
      throw GP_EXCEPTION("Number of bounds must match "
          "number of hyper params!");

    std::vector<double> transf(Size());

    for(uint i=0; i<Size(); ++i){

      if(std::isinf(ubounds[i])){
        transf[i] = exp((*this)[i]) + lbounds[i];
      }
      else
        transf[i] = (ubounds[i] - lbounds[i]) * GP_CumulativeGaussian()((*this)[i]) + lbounds[i];

      if(std::isnan(transf[i])){
        throw GP_EXCEPTION2("hparm is nan! value: %lf", (*this)[i]);
      }
    }
    return HyperParameters(transf);
  }

  template <typename IndexedContainer>
  HyperParameters TransformInv(IndexedContainer const &lbounds,
      IndexedContainer const &ubounds) const
  {
    if(lbounds.size() != Size() ||
        ubounds.size() != Size())
      throw GP_EXCEPTION("Number of bounds must match "
          "number of hyper params!");

    std::vector<double> transf(Size());

    for(uint i=0; i<Size(); ++i){

      if(std::isinf(ubounds[i])){
        double diff = (*this)[i] - lbounds[i] + EPSILON;
        if(diff < 0)
          throw GP_EXCEPTION("Invalid hyper parameter. Did you respect the bounds?");

        transf[i] = ::log(diff);
      }

      else {
        if((*this)[i] < lbounds[i] - EPSILON ||
            (*this)[i] > ubounds[i] + EPSILON)
          throw GP_EXCEPTION("Invalid hyper parameter. Did you respect the bounds?");

        double d = ((*this)[i] - lbounds[i] + EPSILON) / (ubounds[i] - lbounds[i] + EPSILON);
        d = MAX(EPSILON, d);
        d = MIN(1-EPSILON, d);
        transf[i] = GP_CumulativeGaussian().Inv(d);
      }
    }
    return HyperParameters(transf);
  }

  double TransformDeriv(double arg, double lbound, double ubound) const
  {
    if(std::isinf(ubound))
      return exp(arg);
    else
      return (ubound - lbound) * GP_CumulativeGaussian().Deriv(arg);
  }

};

static std::string ClassName()
{
  return "GP_SquaredExponential";
}

static double SqrDistance(double x1, double x2)
{
  return SQR((x1 - x2));
}

static double SqrDistance(std::vector<double> const &x1, std::vector<double> const &x2)
{
  if(x1.size() != x2.size())
    throw GP_EXCEPTION("Lengths of arguments must agree.");

  double sum = 0;
  for(uint i=0; i<x1.size(); ++i)
    sum += (x1[i] - x2[i]) * (x1[i] - x2[i]);

  return sum;
}

static double SqrDistance(GP_Vector const &x1, GP_Vector const &x2)
{
  if(x1.Size() != x2.Size())
    throw GP_EXCEPTION("Lengths of arguments must agree.");

  GP_Vector diff = x1 - x2;

  return diff.Dot(diff);
}

static double SqrDistance(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams)
{
  return SqrDistance(arg1, arg2) / SQR(hparams.length_scale);
}

static double SqrDistanceOneDim(double x1, double x2, uint idx,
    HyperParameters const &hparams)
{
  return SqrDistance(x1, x2) /  SQR(hparams.length_scale);
}

double operator()(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams) const
{
  double sqr_dist = SqrDistance(arg1, arg2);
  double retval = hparams.sigma_f_sqr *
      exp(-sqr_dist / (2.0 * SQR(hparams.length_scale)));

  // add noise variance to diagonal
  if(sqr_dist < 1e-12)
    retval += hparams.sigma_n_sqr;

  return retval;
}

// computes k(arg, arg), i.e. the diagonal of the covariance matrix
double operator()(ArgType const &arg,
    HyperParameters const &hparams) const
{
  return hparams.sigma_f_sqr + hparams.sigma_n_sqr;
}

double PartialDerivative(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams, uint idx) const
{
  double sqr_dist =  SqrDistance(arg1, arg2);
  double sqr_length = SQR(hparams.length_scale);

  // derivative with respect to sigma_f_sqr
  if(idx == 0)
    return exp(-sqr_dist/(2.0 * sqr_length));

  // derivative with respect to the length scale
  else if(idx == 1)
    return hparams.sigma_f_sqr * sqr_dist / (hparams.length_scale * sqr_length) *
        exp(-sqr_dist / (2.0 * sqr_length));

  // derivative with respect to the noise variance
  else
    return (sqr_dist < 1e-12);
}

double PartialDerivative(ArgType const &arg,
    HyperParameters const &hparams, uint idx) const
{
  // derivative with respect to sigma_f_sqr
  if(idx == 0)
    return 1;

  // derivative with respect to the length scale
  else if(idx == 1)
    return 0;

  // derivative with respect to the noise variance
  else
    return 1.;
}

std::vector<double> Derivative(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams) const
    {
  std::vector<double> deriv(hparams.Size());

  for(uint i=0; i<deriv.size(); ++i)
    deriv[i] = PartialDerivative(arg1, arg2, hparams, i);

  return deriv;
    }

};

template <typename ArgType>
class GP_SquaredExponentialARD : public GP_CovarianceFunction<ArgType>
{

public:

  class HyperParameters :
public GP_CovarianceFunction<ArgType>::HyperParameters
{

public:

  typedef typename  GP_CovarianceFunction<ArgType>::HyperParameters Super;

  /*!
   * Default constructor
   */
  HyperParameters() : sigma_f_sqr(1.0), length_scale(), sigma_n_sqr(1.0) {}

  /*!
   * Constructs parameters with an STL vector. No range checking
   */
  HyperParameters(std::vector<double> const &vec) :
    Super()
  {
    //proper length initialization needs to be done in GP_InputParams::GetHyperParamsInit
    if (vec.size() > 2) {
      size_t input_dim = vec.size() - 2;
      length_scale = std::vector<double>(input_dim);
      FromVector(vec);
    } else {
      throw GP_EXCEPTION("Invalid number of hyperparameters for GP_SquaredExponentialARD.");
    }
  }

#ifdef USE_BOOST_SERIALIZATION
  /*!
   * Serializes the object to an archive using boost::serialization
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    // serialize base class information
    ar & boost::serialization::base_object<Super>(*this);
    ar & sigma_f_sqr;
    ar & length_scale;
    ar & sigma_n_sqr;
  }
#endif

  double sigma_f_sqr;  // signal variance
  std::vector<double> length_scale; // length scale
  double sigma_n_sqr;  // noise variance

  virtual double operator[](uint i) const
  {
    if(i == 0)
      return sigma_f_sqr;
    else if(i < length_scale.size() + 1)
      return length_scale[i-1];
    else if(i == length_scale.size() + 1)
      return sigma_n_sqr;

    throw GP_EXCEPTION2("Invalid index %d into kernel paramters", i);
  }


  virtual uint Size() const
  {
    return 2 + length_scale.size();
  }

  virtual std::vector<double> ToVector() const
          {
    std::vector<double> vec(Size());
    for(uint i=0; i<Size(); ++i)
      vec[i] = (*this)[i];

    return vec;
          }

  /*!
   * returns the hyperparameters in an STL vector
   */
  virtual void FromVector(std::vector<double> const &vec)
  {
    if(vec.size() != Size())
      throw GP_EXCEPTION("Invalid number of hyperparameters");

    sigma_f_sqr  = vec[0];
    for(uint i=0; i<length_scale.size(); ++i)
      length_scale[i] = vec[i+1];

    sigma_n_sqr = vec.back();
  }
  /*!
   * assigns the hyperparameters from an STL vector
   */
  template <typename IndexedContainer>
  HyperParameters Transform(IndexedContainer const &lbounds,
      IndexedContainer const &ubounds) const
  {
    if(lbounds.size() != Size() ||
        ubounds.size() != Size())
      throw GP_EXCEPTION("Number of bounds must match "
          "number of hyper params!");

    std::vector<double> transf(Size());

    for(uint i=0; i<Size(); ++i){

      if(std::isinf(ubounds[i])){
        transf[i] = exp((*this)[i]) + lbounds[i];
      }
      else
        transf[i] = (ubounds[i] - lbounds[i]) * GP_CumulativeGaussian()((*this)[i]) + lbounds[i];

      if(std::isnan(transf[i])){
        throw GP_EXCEPTION2("hparm is nan! value: %lf", (*this)[i]);
      }
    }
    return HyperParameters(transf);
  }

  template <typename IndexedContainer>
  HyperParameters TransformInv(IndexedContainer const &lbounds,
      IndexedContainer const &ubounds) const
  {
    if(lbounds.size() != Size() ||
        ubounds.size() != Size())
      throw GP_EXCEPTION("Number of bounds must match "
          "number of hyper params!");

    std::vector<double> transf(Size());

    for(uint i=0; i<Size(); ++i){

      if(std::isinf(ubounds[i])){
        double diff = (*this)[i] - lbounds[i] + EPSILON;
        if(diff < 0)
          throw GP_EXCEPTION("Invalid hyper parameter. Did you respect the bounds?");

        transf[i] = ::log(diff);
      }

      else {
        if((*this)[i] < lbounds[i] - EPSILON ||
            (*this)[i] > ubounds[i] + EPSILON)
          throw GP_EXCEPTION("Invalid hyper parameter. Did you respect the bounds?");

        double d = ((*this)[i] - lbounds[i] + EPSILON) / (ubounds[i] - lbounds[i] + EPSILON);
        d = MAX(EPSILON, d);
        d = MIN(1-EPSILON, d);
        transf[i] = GP_CumulativeGaussian().Inv(d);
      }
    }
    return HyperParameters(transf);
  }

  double TransformDeriv(double arg, double lbound, double ubound) const
  {
    if(std::isinf(ubound))
      return exp(arg);
    else
      return (ubound - lbound) * GP_CumulativeGaussian().Deriv(arg);
  }


};

static std::string ClassName()
{
  return "GP_SquaredExponentialARD";
}


static double SqrDistance(double x1, double x2,
    std::vector<double> const &scale)
{
  // here, scale is a vector of size 1
  double diff = x1 - x2;

  return diff / SQR(scale[0]) * diff;
}

static double SqrDistance(std::vector<double> const &x1, std::vector<double> const &x2,
    std::vector<double> const &scale)
{
  if(x1.size() != x2.size() || x1.size() != scale.size())
    throw GP_EXCEPTION("Argument lengths must be the same.");

  double sum = 0;
  for(uint i=0; i<x1.size(); ++i)
    sum += scale[i] * (x1[i] - x2[i]) *  (x1[i] - x2[i]);

  return sum;
}

static double SqrDistance(ArgType const&arg1, ArgType const &arg2,
    HyperParameters const &hparams)
{
  return SqrDistance(arg1, arg2, hparams.length_scale);
}

static double SqrDistanceOneDim(double x1, double x2, uint idx,
    HyperParameters const &hparams)
{
  return SQR(x1 - x2) / SQR(hparams.length_scale[idx]);
}

double operator()(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams) const
{
  double sqr_dist = SqrDistance(arg1, arg2, hparams.length_scale);
  double retval   = hparams.sigma_f_sqr * exp(-0.5 * sqr_dist);

  // add noise variance to diagonal
  if(sqr_dist < 1e-12)
    retval += hparams.sigma_n_sqr;

  return retval;
}

double PartialDerivative(double x1, double x2,
    HyperParameters const &hparams, uint idx) const
{
  double sqr_dist = SqrDistance(x1, x2, hparams.length_scale);

  if(idx == 0)
    return exp(-0.5 * sqr_dist);
  else if(idx < hparams.length_scale.size() + 1)
    return exp(-0.5 * sqr_dist) * hparams.sigma_f_sqr * SQR(x1 - x2) /
        CUB(hparams.length_scale[0]);

  // derivative with respect to the noise variance
  else
    return (sqr_dist < 1e-12);
}

double PartialDerivative(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams, uint idx) const
{
  double sqr_dist =  SqrDistance(arg1, arg2, hparams.length_scale);

  if(idx == 0)
    return exp(-0.5 * sqr_dist);
  else if(idx < hparams.length_scale.size() + 1)
    return exp(-0.5 * sqr_dist) * hparams.sigma_f_sqr *
        SQR(arg1[idx-1] - arg2[idx-1]) /
        CUB(hparams.length_scale[idx-1]);

  // derivative with respect to the noise variance
  else
    return (sqr_dist < 1e-12);
}

std::vector<double> Derivative(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams) const
    {
  std::vector<double> deriv(hparams.Size());

  for(uint i=0; i<deriv.size(); ++i)
    deriv[i] = PartialDerivative(arg1, arg2, hparams, i);

  return deriv;
    }

};


template <typename ArgType>
class GP_Matern : public GP_CovarianceFunction<ArgType>
{
  // we use nu = 3/2

public:

  class HyperParameters :
public GP_CovarianceFunction<ArgType>::HyperParameters
{

public:

  typedef typename GP_CovarianceFunction<ArgType>::HyperParameters Super;

  /*!
   * Default constructor
   */
  HyperParameters() : sigma_f_sqr(1.0), length_scale(1.0) {}

  HyperParameters(std::vector<double> const &vec) :
    Super(vec) {}

#ifdef USE_BOOST_SERIALIZATION
  /*!
   * Serializes the object to an archive using boost::serialization
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    // serialize base class information
    ar & boost::serialization::base_object<Super>(*this);
    ar & sigma_f_sqr;
    ar & length_scale;
  }
#endif

  double sigma_f_sqr;  // signal variance
  double length_scale; //

  virtual double operator[](uint i) const
  {
    if(i == 0)
      return sigma_f_sqr;
    else if(i == 1)
      return length_scale;
    throw GP_EXCEPTION("Invalid index into kernel paramters");
  }

  virtual uint Size() const
  {
    return 2;
  }

  /*!
   * returns the hyperparameters in an STL vector
   */
  virtual std::vector<double> ToVector() const
          {
    std::vector<double> vec(2);
    vec[0] = sigma_f_sqr;
    vec[1] = length_scale;

    return vec;
          }

};

static std::string ClassName()
{
  return "GP_Matern";
}

static double NormSqr(double x1, double x2)
{
  return SQR(x1 - x2);
}

static double NormSqr(std::vector<double> const &x1, std::vector<double> const &x2)
{
  if(x1.size() != x2.size())
    throw GP_EXCEPTION("Argmuent sizes must be the same");

  double sum = 0;
  for(uint i=0; i<x1.size(); ++i)
    sum += (x1[i] - x2[i]) * (x1[i] - x2[i]);
  return sum;
}


double operator()(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams) const
{
  double temp = sqrt(3.) / hparams.length_scale * sqrt(NormSqr(arg1, arg2));
  double retval = hparams.sigma_f_sqr * (1. + temp) * exp(-temp);

  return retval;
}


double PartialDerivative(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams, uint idx) const
{
  double norm_sqr = NormSqr(arg1, arg2);
  double temp = sqrt(3.) / hparams.length_scale * sqrt(norm_sqr);

  if(idx == 0)
    return (1 + temp) * exp(-temp);

  else
    return hparams.sigma_f_sqr * 3. *
        norm_sqr / CUB(hparams.length_scale) * exp(-temp);
}

std::vector<double> Derivative(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams) const
    {
  std::vector<double> deriv(hparams.Size());

  for(uint i=0; i<deriv.size(); ++i)
    deriv[i] = PartialDerivative(arg1, arg2, hparams, i);

  return deriv;
    }

};

template <typename ArgType>
class GP_InverseSine : public GP_CovarianceFunction<ArgType>
{

public:

  class HyperParameters :
public GP_CovarianceFunction<ArgType>::HyperParameters
{

public:
  typedef typename GP_CovarianceFunction<ArgType>::HyperParameters Super;

  HyperParameters(ArgType const &t) : theta(t) {}

#ifdef USE_BOOST_SERIALIZATION
/*!
 * Serializes the object to an archive using boost::serialization
 */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    // serialize base class information
    ar & boost::serialization::base_object<Super>(*this);
    ar & theta;
  }
#endif

  ArgType theta; // length scale

  virtual double operator[](uint i) const
  {
    if(i == 0)
      return theta;
    throw GP_EXCEPTION("Invalid index into kernel paramters");
  }


  virtual uint Size() const
  {
    return theta.size();
  }

  /*!
   * returns the hyperparameters in an STL vector
   */
  virtual std::vector<double> ToVector() const
          {
    std::vector<double> vec(1);
    vec[0] = theta;

    return vec;
          }

  /*!
   * assigns the hyperparameters from an STL vector
   */
  virtual void FromVector(std::vector<double> const &vec)
  {
    if(vec.size() != Size())
      throw GP_EXCEPTION("Invalid number of hyperparameters");

    theta  = vec[0];
  }


};

static std::string ClassName()
{
  return "GP_InverseSine";
}


static double SumSqr(double x1, double theta)
{
  return SQR(x1) * theta;
}

static double SumSqr(std::vector<double> const &x1, std::vector<double> const &theta)
{
  if(x1.size() != theta.size())
    throw GP_EXCEPTION("Argument sizes must be the same");

  double sum = 0;
  for(uint i=0; i<x1.size(); ++i)
    sum += x1[i] * x1[i] * theta[i];
  return sum;
}

static double Sine(double x1, double x2, double theta)
{
  return x1 * theta * x2;
}

static double Sine(std::vector<double> const &x1, std::vector<double> const &x2,
    std::vector<double> const &theta)
{
  if(x1.size() != x2.size() || x1.size() != theta.size())
    throw GP_EXCEPTION("Argument sizes must be the same");

  double sum = 0;
  for(uint i=0; i<x1.size(); ++i)
    sum += x1[i] * x2[i] * theta[i];

  return sum;
}


double operator()(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams) const
{
  double d1 = sqrt(1 + SumSqr(arg1, hparams.theta));
  double d2 = sqrt(1 + SumSqr(arg2, hparams.theta));

  return asin(Sine(arg1, arg2, hparams.theta) / (d1 * d2));
}

double PartialDerivative(ArgType const &x1, ArgType const &arg2,
    HyperParameters const &hparams, uint idx) const
{
  double deriv;

  // TODO: Implement me!
  throw GP_EXCEPTION("This function is not implemented!");

  return deriv;
}

std::vector<double> Derivative(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams) const
    {
  std::vector<double> deriv(hparams.Size());

  for(uint i=0; i<deriv.size(); ++i)
    deriv[i] = PartialDerivative(arg1, arg2, hparams, i);

  return deriv;
    }

};


template <typename ArgType,
template <typename> class CovarianceFuncType1,
template <typename> class CovarianceFuncType2>
class GP_CovarianceFunctionSum : public GP_CovarianceFunction<ArgType>
{

public:

  class HyperParameters :
public GP_CovarianceFunction<ArgType>::HyperParameters
{

public:

  typedef typename GP_CovarianceFunction<ArgType>::HyperParameters Super;

#ifdef USE_BOOST_SERIALIZATION
/*!
 * Serializes the object to an archive using boost::serialization
 */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    // serialize base class information
    ar & boost::serialization::base_object<Super>(*this);
  }
#endif

};

static std::string ClassName()
{
  return "GP_CovarianceFunctionSum";
}


double operator()(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams) const
{
  return its_cov_func1(arg1, arg2, hparams) + its_cov_func2(arg1, arg2, hparams);
}

double PartialDerivative(ArgType const &arg1, ArgType const &arg2,
    HyperParameters const &hparams, uint idx) const
{
  if(idx < its_cov_func1.Size())
    return its_cov_func1.PartialDerivative(arg1, arg2, hparams, idx);
  else
    return its_cov_func2.PartialDerivative(arg1, arg2, hparams, idx - its_cov_func1.Size());
}

private:

CovarianceFuncType1<ArgType> its_cov_func1;
CovarianceFuncType2<ArgType> its_cov_func2;
};

}
#endif
