
#ifndef GP_BINARY_CLASSIFICATION_EP_HH
#define GP_BINARY_CLASSIFICATION_EP_HH

#include <iomanip>
#include <complex>

#include "Classcode/GP_Matrix.hh"
#include "Classcode/GP_BinaryClassification.hh"
#include "Classcode/GP_ObjectiveFunction.hh"
#include "Classcode/GP_Optimizer.hh"

namespace CLASSCODE {

template<typename InputType,
template <typename> class CovarianceFuncType = GP_SquaredExponential>
/*!
 * Standard GP algorithm for binary classification
 *
 * This class implements the Expectation Propagation (EP) algorithm for binary GP
 * classification. The full covariance matrix is calculated and stored, so for
 * large training sets this class is inefficient.
 *
 * The sigmoid function is fixed to be the cumulative Gaussian. The kernel can be chosen,
 * but it is set to the squared exponential (Gaussian) by default.
 */
class GP_BinaryClassificationEP :
public GP_BinaryClassification<InputType, GP_CumulativeGaussian,
CovarianceFuncType>
{
public:

  typedef GP_BinaryClassificationEP<InputType, CovarianceFuncType> Self;
  typedef GP_BinaryClassification<InputType,
      GP_CumulativeGaussian, CovarianceFuncType> Super;
  typedef typename Super::DataSet          DataSet;
  typedef typename Super::KernelType       KernelType;
  typedef typename Super::HyperParameters  HyperParameters;

  /*!
   * Default constructor
   */
  GP_BinaryClassificationEP() :
    Super(), _hparms(),
    _K(), _L(), _nu_tilde(), _tau_tilde(), _deriv(), _logZ(0),
    _bias(0), _lambda(1.0), _EPthresh(1e-4), _maxEPiter(10), _predictionInitialized(false)
  {}

  /*!
   * The constructor needs a training data set and some
   * hyper parameters for the kernel. 'lambda' is the slope of the sigmoid.
   */
  GP_BinaryClassificationEP(DataSet const &train_data,
      HyperParameters const &hparms = HyperParameters(),
      double lambda = 1.0) :
        Super(train_data), _hparms(hparms),
        _K(), _L(), _nu_tilde(), _tau_tilde(), _deriv(), _logZ(0),
        _bias(0), _lambda(lambda), _EPthresh(1e-4), _maxEPiter(10), _predictionInitialized(false)
  {
    // estimate data bias
    uint class0_size = 0, class1_size = 0;
    for(uint i=0; i<Super::Size(); ++i)
      if(Super::GetY()[i] == 1)
        ++class0_size;
      else
        ++class1_size;

    if(class0_size == 0 || class0_size == Super::Size())
      throw GP_EXCEPTION("Only one class in training data!");

    _bias = Super::GetSigFunc().Inv(class0_size / (double)Super::Size());
  }

  /*!
   * From super class
   */
  virtual bool IsA(char const *classname) const
  {
    return (Super::IsA(classname) ||
        std::string(classname) == "GP_BinaryClassificationEP");
  }

  /*!
   * Assignment operator
   */
  GP_BinaryClassificationEP<InputType, CovarianceFuncType> const &
  operator=(GP_BinaryClassificationEP<InputType, CovarianceFuncType> const &other)
  {
    Super::operator=(other);

    _hparms    = other._hparms;
    _K         = other._K;
    _L         = other._L;
    _nu_tilde  = other._nu_tilde;
    _tau_tilde = other._tau_tilde;
    _deriv     = other._deriv;
    _logZ      = other._logZ;

    _predictionInitialized = other._predictionInitialized;
    s_sqrt = other.s_sqrt;
    nu_biased = other.nu_biased;
    z = other.z;
    k_star = other.k_star;

    return *this;
  }

  /*!
   * Returns the hyper parameters of the used kernel
   */
  HyperParameters const &GetHyperParams() const
  {
    return _hparms;
  }

  /*!
   * Runs the Expectation Propagation algorithm until convergence.
   * The result is a set of site parameters \f$\tilde{nu}\f$, \f$\tilde{tau}\f$,
   * posterior parameters \f$\mu\f$ and \f$\Sigma\f$, as well as
   * a covariance matrix K and a Cholesky decompostion L
   */
  virtual void Estimation()
  {
    // first we compute the covariance matrix
    _K = Super::ComputeCovarianceMatrix(_hparms);

    // initialize the site and posterior params
    uint n = Super::Size();
    _nu_tilde = _tau_tilde = _mu = GP_Vector(n);
    _Sigma = _K;

    // now we run EP until convergence (or max number of iterations reached)
    uint iter = 0;
    double max_delta = 1.;
    do {
      // run EP; the maximum difference in tau is our convergence criterion
      GP_Vector delta_tau = ExpectationPropagation(_mu, _Sigma);
      max_delta = delta_tau.Abs().Max();

      // re-compute mu and Sigma for numerical stability; also computes L
      ComputePosteriorParams();

    } while(max_delta > _EPthresh && ++iter < _maxEPiter);

    // Does not happen often, it's not a big problem even if it does.
    // If it happens too often, just increase '_maxEPiter'
    if(iter == _maxEPiter){
      std::cerr << "Warning! EP did not converge" << std::endl;
    }

    // compute the log posterior and  derivatice wrt. the kernel params
    ComputeLogZ();
    ComputeDerivLogZ();
  }

  /*!
   * Sets the hyperparameters before running Estimation
   */
  void Estimation(HyperParameters const &new_hyp)
  {
    _hparms = new_hyp;
    Estimation();
  }

  /*!
   * Updates the site parameters and the posterior mean and covariance.
   * In general, this is equal to Estimation(), but derived classes may
   * use this to avoid a full estimation run.
   */
  virtual void UpdateModelParameters(HyperParameters const &new_hyp)
  {
    Estimation(new_hyp);
  }

  /*!
   * Performs one EP update step, i.e. runs once through the data and updates
   * site parameters (nu, tau) and posterior parameters  (mu, Sigma). Returns
   * a vector of delta values, one per data point. These represent the amount
   * of change and can be used as a stopping criterion.
   */
  GP_Vector ExpectationPropagation(GP_Vector &mu, GP_Matrix &Sigma,
      std::list<uint> const &index_list = std::list<uint>())
  {
    uint n = mu.Size();
    if(index_list.size() == n || index_list.empty()) {
      return ExpectationPropagation(mu, Sigma, index_list.begin(), index_list.end());
    }
    else
      throw GP_EXCEPTION("Size of sub-set must match length of posterior "
          "mean vector. Could not run EP step");

  }

  template <class Iterator>
  GP_Vector ExpectationPropagation(GP_Vector &mu, GP_Matrix &Sigma,
      const Iterator& begin, const Iterator& end)
  {
    uint n = mu.Size();
    GP_Vector delta_tau(n);
    Iterator it = begin;

    for(uint i=0; i<n; i++) {

      // compute approximate cavity parameters
      double tau_min = 1./Sigma[i][i] - _tau_tilde[i];
      double nu_min  = mu[i] / Sigma[i][i] - _nu_tilde[i];

      // compute marginal moments using the derivatives
      double dlZ, d2lZ;
      uint tgt_idx;
      if(begin == end)
        tgt_idx = i; // label index is the next one from the training set
      else
        tgt_idx = *it++; // label index is the next one from the chosen subset

      ComputeDerivatives(Super::GetY()[tgt_idx], nu_min, tau_min, dlZ, d2lZ);

      // update site parameters
      double old_tau_tilde = _tau_tilde[i];
      double denom = 1.0 - d2lZ / tau_min;
      _tau_tilde[i] = MAX(d2lZ / denom, EPSILON);
      _nu_tilde[i]  = (dlZ + nu_min / tau_min * d2lZ) / denom;
      delta_tau[i]     = _tau_tilde[i] - old_tau_tilde;

      // update approximate posterior
      GP_Vector si = Sigma.Col(i);
      denom = 1.0 +  delta_tau[i] * Sigma[i][i];
      if(fabs(denom) > EPSILON)
        Sigma -= delta_tau[i] / denom * GP_Matrix::OutProd(si);
      else
        Sigma -= delta_tau[i] / EPSILON * GP_Matrix::OutProd(si);

      //TODO:is this a bug?
      mu = Sigma * _nu_tilde;
    }

    return delta_tau;
  }

  /*!
   * Learns hyper parameters from the training data given in the constructor.
   * The optimization  starts with  an inital estimate 'init', and guarantees
   * that the  parameters are never smaller then 'lower_bound'. The parameter
   * 'nb_iterations' is  only  used in derived  classes. The function returns
   * the residual that resulted from the optimization.
   */
  virtual double LearnHyperParameters(HyperParameters &init,
      GP_Vector lower_bounds = GP_Vector(1, 0.0),
      GP_Vector upper_bounds = GP_Vector(1, 1.0),
      uint nb_iterations = 0)
  {
    double residual;
    Optimization(init.ToVector(), lower_bounds, upper_bounds, residual);
    init = _hparms;
    return residual;
  }

  /*!
   * Returns the covariance matrix
   */
  inline GP_Matrix GetCovMat() const
  {
    return _K;
  }

  /*!
   * Returns the Cholesky decoposition used for the calculation
   */
  inline GP_Matrix GetCholMat() const
  {
    return _L;
  }

  /*!
   * Returns the mean values of the likelihood obtained after EP
   */
  inline GP_Vector GetSiteMean() const
  {
    return _nu_tilde / _tau_tilde;
  }

  /*!
   * Returns the variance values of the likelihood obtained after EP
   */
  inline GP_Vector GetSiteVar() const
  {
    return 1./_tau_tilde;
  }

  /*!
   * Returns the log-posterior
   */
  inline double GetLogZ() const
  {
    return _logZ;
  }

  /*!
   * Returns the derivative of the log-posterior with respect to the
   * kernel parameters
   */
  inline GP_Vector GetDeriv() const
  {
    return _deriv;
  }

  /*!
   * Plots the log-posterior in 2D by assigning values between 'min' and 'max'
   * with a given 'step' to the first two kernel hyper parameters.
   */
  void PlotLogZ(double min, double max, double step)
  {
    HyperParameters store_hparms = _hparms;
    std::vector<double> param_vec = _hparms.ToVector();
    static uint idx_logz = 0;
    std::stringstream fname;

    fname << "logz" << std::setw(3) << std::setfill('0') << idx_logz++ << ".dat";
    WRITE_FILE(ofile, fname.str().c_str());

    for(param_vec[0] = min; param_vec[0] < max; param_vec[0] += step){
      for(param_vec[1] = min; param_vec[1] < max; param_vec[1] += step){
        HyperParameters hparms;
        hparms.FromVector(param_vec);
        UpdateModelParameters(hparms);

        ofile << param_vec[0] << " " << param_vec[1] << " " << _logZ << std::endl;
      }
      ofile << std::endl;
    }

    ofile.close();
    _hparms = store_hparms;
  }

  double Prediction(InputType const &test_input)
  {
    double mu_star, sigma_star;
    return Prediction(test_input, mu_star, sigma_star);
  }

  /*!
   * Performs the prediction step for a given 'test_input'. Returns the probability
   * that the test point has label 1, as well as the predictive mean and variance
   */
  virtual double Prediction(InputType const &test_input,
      double &mu_star, double &sigma_star)
  {
    if(_nu_tilde.Size() == 0 || _tau_tilde.Size() == 0)
      throw GP_EXCEPTION("Could not do prediction. Run estimation first.");

    uint n=Super::Size();

    if(_L.Rows() == 0 || _L.Cols() == 0)
      throw GP_EXCEPTION("Cholesky matrix not computed. Run 'Expectation Propagation' first.");

    if (!_predictionInitialized) {
      s_sqrt = MakeSqrtS();
      nu_biased = _nu_tilde + _bias * _tau_tilde;
      z = s_sqrt * _L.SolveChol(s_sqrt * (_K * nu_biased));

      k_star = GP_Vector(n);
      _predictionInitialized = true;
    }

    // Compute k_star
    for(uint i=0; i<n; i++)
      k_star[i] = Super::CovFunc(Super::GetX()[i], test_input, _hparms);

    mu_star = k_star.Dot(nu_biased - z);
    GP_Vector v = _L.ForwSubst(s_sqrt * k_star);
    sigma_star = Super::CovFunc(test_input, test_input, _hparms) - v.Dot(v);
    sigma_star = MAX(sigma_star, 0);

    double pi_star = Super::SigFunc(mu_star /
        sqrt(1.0 /SQR(_lambda) + sigma_star));

    return pi_star;
  }


  int Read(std::string filename, int pos = 0)
  {
    int new_pos = Super::Read(filename, pos);

    new_pos = _hparms.Read(filename, new_pos);
    new_pos = _K.Read(filename, new_pos);
    new_pos = _L.Read(filename, new_pos);
    new_pos = _Sigma.Read(filename, new_pos);
    new_pos = _nu_tilde.Read(filename, new_pos);
    new_pos = _tau_tilde.Read(filename, new_pos);
    new_pos = _deriv.Read(filename, new_pos);
    new_pos = _mu.Read(filename, new_pos);
    READ_FILE(ifile, filename.c_str());
    ifile.seekg(new_pos);
    ifile >> _logZ >> _bias >> _lambda;

    return ifile.tellg();
  }

  void Write(std::string filename) const
  {
    Super::Write(filename);
    _hparms.Write(filename);
    _K.Write(filename);
    _L.Write(filename);
    _Sigma.Write(filename);
    _nu_tilde.Write(filename);
    _tau_tilde.Write(filename);
    _deriv.Write(filename);
    _mu.Write(filename);
    APPEND_FILE(ofile, filename.c_str());
    ofile << _logZ << " " << _bias << " " <<  _lambda << std::endl;
  }

#ifdef USE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  /*!
   * Serializes the object to an archive using boost::serialization
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    // serialize base class information
    ar & boost::serialization::base_object<Super>(*this);
    ar & _hparms;
    ar & _K & _L & _Sigma;
    ar & _nu_tilde & _tau_tilde;
    ar & _deriv & _mu;
    ar & _logZ & _bias & _lambda;

    ar & _EPthresh;
    ar & _maxEPiter;
  }
#endif

protected:

  HyperParameters _hparms;
  GP_Matrix _K, _L, _Sigma;
  GP_Vector _nu_tilde, _tau_tilde;
  GP_Vector _deriv, _mu;
  double _logZ, _bias, _lambda;

  double _EPthresh; // threshold for EP stopping criterion
  uint _maxEPiter; // maximum number of EP iterations

  bool _predictionInitialized;
  GP_Vector s_sqrt;
  GP_Vector nu_biased;
  GP_Vector z;
  GP_Vector k_star;

  /*!
   * Returns true if the optimizer has converged; 'lower_bound' is the
   * minimal required value of the kernel parameters.
   */
  bool Optimization(GP_Vector const &lower_bounds,
      GP_Vector const &upper_bounds,
      double &residual)
  {
    return Optimization(_hparms.ToVector(), lower_bounds,
        upper_bounds, residual);
  }

  /*!
   * Numerically stable version to compute dlZ and d2lZ
   */
  void ComputeDerivatives(int y, double nu_min, double tau_min,
      double &dlZ, double &d2lZ) const
  {
    double c;
    double u = ComputeZeroMoment(y, nu_min, tau_min, c);

    dlZ = c * exp(Super::GetSigFunc().LogDeriv(u) -
        Super::GetSigFunc().Log(u));

    if(std::isnan(dlZ)){

      std::stringstream msg;
      msg << c << " " << u << " " << tau_min;
      throw GP_EXCEPTION2("dlz is nan: %s", msg.str());
    }

    d2lZ = dlZ * (dlZ + u * c);
  }

  virtual void ComputeLogZ()
  {
    double term1 = 0, term2 = 0, term3 = 0, term4 = 0, term5 = 0;

    GP_Vector sigma_diag = _Sigma.Diag();
    GP_Vector tau_n = 1. / sigma_diag - _tau_tilde;
    GP_Vector nu_n  = _mu / sigma_diag - _nu_tilde;

    for(uint i=0; i<tau_n.Size(); ++i){
      double zi = ComputeZeroMoment(Super::GetY()[i], nu_n[i], tau_n[i]);
      if(fabs(_tau_tilde[i]) > EPSILON){
        double arg;
        if(fabs(tau_n[i]) > EPSILON)
          arg = 1. + _tau_tilde[i] / tau_n[i];
        else
          arg = 1. + _tau_tilde[i] / EPSILON;
        if(fabs(arg) < EPSILON)
          arg = EPSILON;
        term1 += abs(log(std::complex<double>(arg,0)));
      }
      term2 += log(_L[i][i]);
      term3 += Super::GetSigFunc().Log(zi);
      if(fabs(nu_n[i]) > EPSILON){
        term5 += nu_n[i] *
            (_tau_tilde[i] / tau_n[i] * nu_n[i] - 2. * _nu_tilde[i]) /
            (_tau_tilde[i] + tau_n[i]);
      }
    }

    GP_Matrix SigmaNoDiag = _Sigma;
    for(uint i=0; i<SigmaNoDiag.Cols(); ++i)
      SigmaNoDiag[i][i] = 0.;

    term4 = _nu_tilde.Dot(SigmaNoDiag * _nu_tilde);
    _logZ = term3 - term2 + (term1 + term4 + term5) / 2.;
  }

  virtual void ComputeDerivLogZ()
  {
    GP_Vector s_sqrt = MakeSqrtS();
    GP_Vector b =
        _nu_tilde - s_sqrt * _L.SolveChol(s_sqrt * (_K * _nu_tilde));

    GP_Matrix Z = GP_Matrix::Diag(s_sqrt), C;
    Z = GP_Matrix::OutProd(b) - Z * _L.SolveChol(Z);

    _deriv = GP_Vector(_hparms.Size());
    for(uint j=0; j<_deriv.Size(); ++j){

      C = Super::ComputePartialDerivMatrix(_hparms, j);
      _deriv[j] = Z.ElemMult(C).Sum() / 2.;
    }
  }

  typedef GP_ObjectiveFunction<InputType, Self> ObjectiveFunction;

  GP_Vector MakeSqrtS() const
  {
    uint n = _tau_tilde.Size();
    GP_Vector s_sqrt(n);

    for(uint i=0; i<n; i++)
      if(_tau_tilde[i] > EPSILON)
        s_sqrt[i] = sqrt(_tau_tilde[i]);
      else
        s_sqrt[i] = sqrt(EPSILON);

    return s_sqrt;
  }

  /*!
   * Computes y * nu / (tau * sqrt(1 + 1/tau)) in a numerically stable way
   */
  double ComputeZeroMoment(int y, double nu, double tau) const
  {
    double c;
    return ComputeZeroMoment(y, nu, tau, c);
  }

  double ComputeZeroMoment(int y, double nu, double tau, double &c) const
  {
    std::complex<double> tau_cplx(tau, 0);

    double denom = MAX(abs(sqrt(tau_cplx *
        (tau_cplx / SQR(_lambda) + 1.))),
        EPSILON);

    c = y * tau / denom;

    if(fabs(nu) < EPSILON)
      return c * _bias;

    return y * nu / denom + c * _bias;
  }

  /*!
   * Computes mu and Sigma from the site parameters nu_tilde and tau_tilde
   */
  void ComputePosteriorParams()
  {
    uint n = _mu.Size();
    GP_Vector s_sqrt = MakeSqrtS();

    _L = _K;
    _L.CholeskyDecompB(s_sqrt);

    // Compute S^0.5 * K in place
    _Sigma = GP_Matrix(n, n);
    for(uint i=0; i<n; i++)
      for(uint j=0; j<n; j++)
        _Sigma[i][j] = s_sqrt[i] * _K[i][j];

    GP_Matrix V = _L.ForwSubst(_Sigma);

    _Sigma = _K - V.TranspTimes(V);
    _mu    = _Sigma * _nu_tilde;
  }

  /*!
   * Returns true if the optimizer has converged
   */
  bool Optimization(std::vector<double> const &init_params,
      GP_Vector const &lower_bounds,
      GP_Vector const &upper_bounds,
      double &residual)
  {
    std::cout << "init " << std::flush;

    for(uint i=0; i<init_params.size(); ++i)
      std::cout << init_params[i] << " " << std::flush;
    std::cout << std::endl;

    ObjectiveFunction err(*this, _hparms.Size(), lower_bounds, upper_bounds);
    GP_Optimizer min(err, GP_Optimizer::PR);

    uint nb_params = init_params.size();
    std::vector<double> init(nb_params), cur_vals(nb_params);

    HyperParameters hparams(init_params);
    hparams = hparams.TransformInv(lower_bounds, upper_bounds);
    min.Init(hparams.ToVector());
    std::cout << "starting optimizer..." << std::endl << std::flush;
    uint nb_max_iter = 50, iter = 0;
    _deriv = GP_Vector(_deriv.Size(), 1.);

    char bar = '-';
    while(min.Iterate() && _deriv.Abs().Max() > 1e-5 && iter < nb_max_iter){

      //	std::cout << bar << "\r" << std::flush;
      //	if(bar == '-')
      //	  bar = '\\';
      //	else if(bar == '\\')
      //	  bar = '|';
      //	else if(bar == '|')
      //	  bar = '/';
      //	if(bar == '/')
      //	  bar = '-';

      std::cout << "Iteration " << iter << "/" << nb_max_iter << " ..." << std::endl << std::flush;

      min.GetCurrValues(cur_vals);

      HyperParameters cur_parms(cur_vals);
      cur_parms = cur_parms.Transform(lower_bounds, upper_bounds);
      ++iter;
    }
    std::cout << std::endl;
    bool retval = min.TestConvergence();
    if(retval)
      std::cout << "converged!" << std::endl;

    min.GetCurrValues(init);
    residual = min.GetCurrentError();

    std::cout << "found parameters: " << std::flush;
    HyperParameters final_parms(init);
    _hparms = final_parms.Transform(lower_bounds, upper_bounds);

    for(uint i=0; i<_hparms.Size(); ++i){
      std::cout << _hparms[i] << " " << std::flush;
    }
    std::cout << std::endl;

    return retval;
  }

};

}
#endif
