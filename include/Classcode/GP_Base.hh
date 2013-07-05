
#ifndef GP_BASE_HH
#define GP_BASE_HH

#include <vector>
#include <list>

#include "Classcode/GP_Matrix.hh"
#include "Classcode/GP_DataSet.hh"
#include "Classcode/GP_CovarianceFunction.hh"

namespace CLASSCODE {

  /*! 
   * \class GP_Base
   *
   * Base class for Gaussian Processes. 
   *
   * This class stores the training data and the covariance function
   */
  template<typename InputType, typename OutputType, 
	   template <typename> class CovarianceFuncType = GP_SquaredExponential>
  class GP_Base
  {
  public:

    typedef GP_DataSet<InputType, OutputType>     DataSet;
    typedef CovarianceFuncType<InputType>         KernelType;
    typedef typename KernelType::HyperParameters  HyperParameters;

  protected:
    
    /*!
     * Default constructor
     */
    GP_Base() : 
      _cov_func(), _train_data()
    {}

    /*!
     * This constructor needs a training data set
     */
    GP_Base(DataSet const &train_data) : 
      _cov_func(), _train_data(train_data)
    {}

    /*!
     * Returns true if the class name is correct
     */
    virtual bool IsA(char const *classname) const
    {
      return std::string(classname) == "GP_Base";
    }

  public:

    /*!
     * Returns the size of the training set
     */
    uint Size() const
    {
      return _train_data.Size();
    }

    /*!
     * Returns the training set
     */
    GP_DataSet<InputType, OutputType> const &GetTrainData() const
    {
      return _train_data;
    }
  
    /*!
     * Returns all input data from the training set
     */
    std::vector<InputType> const &GetX() const
    {
      return _train_data.GetInput();
    }

    /*!
     * Returns all output data from the training set
     */
    std::vector<OutputType> const &GetY() const
    {
      return _train_data.GetOutput();
    }

    /*!
     * Returns the dimension of the output data
     */
    uint GetOutputDim() const
    {
      return _train_data.GetOutputDim();
    }

    /*!
     * Returns all output data from the training set into a 
     * given container of OutputType
     */
    template<typename OutContType>
    void GetOutput(OutContType &out) const
    {
      _train_data.GetOutput(out);
    }

    /*!
     * Applies the kernel function to two given input values
     */
    double CovFunc(InputType const &input1, InputType const &input2,
		   HyperParameters const &hparms) const
    {
      return _cov_func(input1, input2, hparms);
    }

    /*!
     * Computes the partial derivative with respect to the hyper-parameter
     * with the given 'idx' to two input values
     */
    double CovFuncPartialDeriv(InputType const &input1, InputType const &input2,
			       HyperParameters const &hparms, uint idx) const
    {
      return _cov_func.PartialDerivative(input1, input2, hparms, idx);
    }

    /*!
     * Computes the covariance matrix that corresponds to the training
     * data points that are indexed in the given 'index_list'
     */
    GP_Matrix
    ComputeCovarianceMatrix(std::list<uint> const &index_list,
			    HyperParameters const &hparms) const
    {
      GP_Matrix K(index_list.size(), index_list.size());
      ComputeCovarianceMatrix(index_list, hparms, K);

      return K;
    }

    
    /*!
     * Computes the covariance matrix 'K' that corresponds to the training
     * data points that are indexed in the given 'index_list'
     */
    void
    ComputeCovarianceMatrix(std::list<uint> const &index_list,
			    HyperParameters const &hparms, 
			    GP_Matrix &K) const
    {
      std::list<uint>::const_iterator it1, it2;
      
      it1 = index_list.begin();
      for(uint i=0; i<K.Rows(); ++i, ++it1){
	K[i][i] = _cov_func(_train_data.GetInput(*it1), hparms);

	it2 = it1; ++it2;
	for(uint j=i+1; j<K[i].size(); ++j, ++it2)
	  K[i][j] = K[j][i] = _cov_func(_train_data.GetInput(*it1), 
					   _train_data.GetInput(*it2), hparms);
      }
    }

    /*!
     * Computes the covariance matrix from two given data sets
     */
    GP_Matrix
    ComputeCovarianceMatrix(GP_DataSet<InputType, OutputType> const &data1, 
			    GP_DataSet<InputType, OutputType> const &data2,
			    HyperParameters const &hparms) const
    {
      GP_Matrix K(data1.Size(), data2.Size());
      if(data1.Size() == data2.Size())
				for(uint i=0; i<K.Rows(); i++)
					for(uint j=i; j<K[i].size(); j++)
						K[i][j] = K[j][i] = _cov_func(data1.GetInput(i),
									data2.GetInput(j), hparms);
      else
				for(uint i=0; i<K.Rows(); i++)
					for(uint j=0; j<K[i].size(); j++)
						K[i][j] = _cov_func(data1.GetInput(i),
							data2.GetInput(j), hparms);
      return K;
    }

    /*!
     * Computes the partial derivatives of the covariance matrix that 
     * corresponds to the training data points that are indexed in the 
     * given 'index_list'
     */
    GP_Matrix
    ComputePartialDerivMatrix(std::list<uint> const &index_list,
			      HyperParameters const &hparms, uint idx) const
    {
      GP_Matrix C(index_list.size(), index_list.size());
      std::list<uint>::const_iterator it1, it2;
      
      it1 = index_list.begin();
      for(uint i=0; i<C.Rows(); ++i, ++it1){

	C[i][i] = _cov_func.PartialDerivative(_train_data.GetInput(*it1), hparms, idx);

	it2 = it1; ++it2;
	for(uint j=i+1; j<C[i].size(); ++j, ++it2)
	  C[i][j] = C[j][i] = _cov_func.PartialDerivative(_train_data.GetInput(*it1), 
							  _train_data.GetInput(*it2), 
							  hparms, idx);
      }
      return C;
    }

    /*!
     * Computes the partial derivatives of the covariance matrix that 
     * corresponds to the two given data sets
     */
    GP_Matrix
    ComputePartialDerivMatrix(GP_DataSet<InputType, OutputType> const &data1, 
			      GP_DataSet<InputType, OutputType> const &data2,
			      HyperParameters const &hparms, uint idx) const
    {
      GP_Matrix C(data1.Size(), data2.Size());

      if(data1.Size() == data2.Size())
	for(uint i=0; i<C.Rows(); i++)
	  for(uint j=i; j<C[i].size(); j++)
	    C[i][j] = C[j][i] = _cov_func.PartialDerivative(data1.GetInput(i), 
							    data2.GetInput(j), 
							    hparms, idx);
      else
	for(uint i=0; i<C.Rows(); i++)
	  for(uint j=0; j<C[i].size(); j++)
	    C[i][j] = _cov_func.PartialDerivative(data1.GetInput(i), 
						  data2.GetInput(j), hparms, idx);
      return C;
    }

    /*!
     * Computes the covariance matrix from the training and a given data set
     */
    GP_Matrix
    ComputeCovarianceMatrix(GP_DataSet<InputType, OutputType> const &data,
			    HyperParameters const &hparms) const
    {
      return ComputeCovarianceMatrix(_train_data, data, hparms);
    }

    /*!
     * Computes the covariance matrix from the training set with itself
     */
    GP_Matrix
    ComputeCovarianceMatrix(HyperParameters const &hparms) const
    {
      return ComputeCovarianceMatrix(_train_data, _train_data, hparms);
    }

    /*!
     * Computes the partial derivative matrix from the training set with itself
     */
    GP_Matrix
    ComputePartialDerivMatrix(HyperParameters const &hparms, uint idx) const
    {
      return ComputePartialDerivMatrix(_train_data, _train_data, hparms, idx);
    }

    int Read(std::string filename, int pos = 0)
    {
      std::string cfname;
      READ_FILE(ifile, filename.c_str());
      ifile.seekg(pos);
      ifile >> cfname;

      if(cfname != _cov_func.ClassName())
	throw GP_EXCEPTION2("Could not load model file: "
			    "incorrect covariance function type %s", cfname);

      return _train_data.Read(filename, ifile.tellg());
    }

    void Write(std::string filename) const
    {
      WRITE_FILE(ofile, filename.c_str());
      ofile << CovarianceFuncType<InputType>::ClassName() << std::endl;
      _train_data.Write(filename);
    }

#ifdef USE_BOOST_SERIALIZATION
    friend class boost::serialization::access;
    /*!
     * Serializes the object to an archive using boost::serialization
     */
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
  	  ar & _train_data;
    }
#endif

  protected:

    /*!
     * Returns the training set, non-const version
     */
    GP_DataSet<InputType, OutputType> &GetTrainData()
    {
      return _train_data;
    }

  private:
  
    KernelType _cov_func;
    DataSet _train_data;
  };

}

#endif
