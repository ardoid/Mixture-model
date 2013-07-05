#ifndef GP_DATA_SET_HH
#define GP_DATA_SET_HH

#include <vector>
#include <list>
#include <set>
#include <map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <math.h>
#include <gsl/gsl_rng.h>

#include "Classcode/GP_Constants.hh"
#include "Classcode/GP_Exception.hh"
#include "Classcode/GP_Vector.hh"

namespace CLASSCODE {

  /*!
   * DataSet class
   *
   * This class stores the data used later for training and testing.
   * It consists of an 'input' part x, and an 'output' part y.
   */
  template <typename InputType, typename OutputType>
  class GP_DataSet
  {

  public:

    /*!
     * Default constructor
     */
    GP_DataSet() : _in_dim(0), _out_dim(0), _xvec(), _yvec() {}

    /*!
     * Returns the number of data points 
     */
    uint Size() const {  return _xvec.size(); }

    /*!
     * Returns all input data in a vector
     */
    std::vector<InputType> const &GetInput() const
    {
      return _xvec;
    }

    /*!
     * Returns a particular input datum
     */
    InputType const &GetInput(uint idx) const
    {
      if(idx >= _xvec.size())
	throw GP_EXCEPTION2("Invalid index %d of input data point.", idx);

      return _xvec[idx];
    }

    /*!
     * Returns all output data in a vector
     */
    std::vector<OutputType> const &GetOutput() const
    {
      return _yvec;
    }

    /*!
     * Returns a particular output datum
     */
    OutputType const &GetOutput(uint i) const
    {
      if(i >= _yvec.size())
	throw GP_EXCEPTION("Invalid index of input data point.");

      return _yvec[i];
    }

    /*!
     * Returns the number of dimensions of the input data
     */
    uint GetInputDim() const
    {
      return _in_dim;
    }

    /*!
     * Returns the number of dimensions of the output data
     */
    uint GetOutputDim() const
    {
      return _out_dim;
    }
  
    /*!
     * Replaces the input datum with a given index
     */
    void SetInput(uint index, InputType const &xval)
    {
      if(_in_dim == 0)
	_in_dim = GetDimension(xval);

      else if (_in_dim != GetDimension(xval))
	throw GP_EXCEPTION("Dimension of input type must be consistent.");
      
      if(index >= _xvec.size())
	throw GP_EXCEPTION("Invalid index into input vector.");

      _xvec[index] = xval;
    }

    /*!
     * Replaces the output datum with a given index
     */
    void SetOutput(uint index, OutputType const &yval)
    {
      if(_out_dim == 0)
	_out_dim = GetDimension(yval);

      else if (_out_dim != GetDimension(yval))
	throw GP_EXCEPTION("Dimension of output type must be consistent.");
      
      if(index >= _yvec.size())
	throw GP_EXCEPTION("Invalid index into output vector.");

      _yvec[index] = yval;
    }


    /*!
     * Selects all the data with indices given by those in 'cont', i.e. 
     * ContainerType must be a collection of indices into the DataSet
     */
    template<typename ContainerType>
    GP_DataSet<InputType, OutputType> GetSubset(ContainerType const &cont) const
    {
      std::list<InputType> new_input;
      std::list<OutputType> new_output;

      typename ContainerType::const_iterator it;
      for(it = cont.begin(); it!= cont.end(); ++it){
	new_input.push_back(_xvec[*it]);
	new_output.push_back(_yvec[*it]);
      }
      
      GP_DataSet<InputType, OutputType> subset;
      subset.Append(new_input, new_output);

      return subset;
    }


    /*!
     * Selects that part of the data for which the output equals 'out'
     */
    GP_DataSet<InputType, OutputType> GetSubsetIf(OutputType const &out) const
    {
      GP_DataSet<InputType, OutputType> subset;
      
      for(uint i=0; i<_xvec.size(); ++i){
	if(_yvec[i] == out){
	  subset.Add(_xvec[i], _yvec[i]);
	}
      }

      return subset;
    }


    /*!
     * Returns the mean of all input values
     */
    InputType CalcInputMean() const
    {
      if(_xvec.size() == 0)
	return InputType();

      InputType mean = _xvec[0];
      for(uint i=1; i<_xvec.size(); ++i)
	mean += _xvec[i];
      mean /= _xvec.size();

      return mean;
    }

    
    /*!
     * Returns the variance of all input values
     */
    InputType CalcInputVariance() const
    {
      if(_xvec.size() == 0)
	return InputType();

      InputType var = _xvec[0];
      for(uint i=1; i<_xvec.size(); ++i)
	var += _xvec[i] * _xvec[i];
      var /= _xvec.size();

      InputType mean = CalcInputMean();

      return var - mean * mean;
    }

    /*!
     * Inserts all elements in the containers 'x_cont' and 'y_cont at the end of 
     * each vector. 
     */
    template <typename ContainerType1, typename ContainerType2>
    void Append(ContainerType1 const &x_cont, ContainerType2 const &y_cont)
	      
    {
      _xvec.insert(_xvec.end(), x_cont.begin(), x_cont.end());
      _yvec.insert(_yvec.end(), y_cont.begin(), y_cont.end());

      for(uint i=0; i<_yvec.size(); i++){
	uint xdim = GetDimension(_xvec[i]);
	uint ydim = GetDimension(_yvec[i]);

	if(_in_dim == 0)
	  _in_dim = xdim;
	else if (_in_dim != xdim){
	  std::cout << _in_dim << " " << xdim << std::endl;
	  throw GP_EXCEPTION("All output values must have the same dimension.");
	}

	if(_out_dim == 0)
	  _out_dim = ydim;
	else if (_out_dim != ydim)
	  throw GP_EXCEPTION("All output values must have the same dimension.");
      }
    }

    /*!
     * Inserts all elements of the 'other' data set at the end of this one. 
     */
    void Append(GP_DataSet<InputType, OutputType> const &other)
    {
      Append(other._xvec, other._yvec);
    }

    /*!
     * Adds a single pair of input / output data to the end of this data set
     */
    void Add(InputType const &input, OutputType const &output)
    {
      if(_xvec.size() == 0 && _yvec.size() == 0){
	_in_dim = GetDimension(input);
	_out_dim = GetDimension(output);
      }
	
      else if(GetDimension(input)  != _in_dim ||
	      GetDimension(output) != _out_dim)
	throw GP_EXCEPTION("Incorrect dimension of data to add.");

      _xvec.push_back(input);
      _yvec.push_back(output);
    }

    /*!
     * Removes the data given by a vector of indices 
     */
    void Remove(std::vector<uint> const &idcs)
    {
      std::vector<bool> to_remove(_xvec.size(), false);
      for(uint i=0; i<idcs.size(); ++i)
	to_remove[idcs[i]] = true;

      std::vector<InputType> _xvec_new;
      std::vector<OutputType> _yvec_new;

      for(uint i=0; i<to_remove.size(); ++i){
	if(!to_remove[i]){
	  _xvec_new.push_back(_xvec[i]);
	  _yvec_new.push_back(_yvec[i]);
	}
      }
      _xvec = _xvec_new;
      _yvec = _yvec_new;
    }

    /*!
     * Clears all data in the data set and replaces it with the new data
     */
    template <typename ContainerType1, typename ContainerType2>
    void Set(ContainerType1 const &x_cont, ContainerType2 const &y_cont)
    {
      if(x_cont.size() != y_cont.size())
	throw GP_EXCEPTION("Number of input values must match number of output values.");
      
      Clear();
      Append(x_cont, y_cont);	 
    }
  
    /*!
     * Randomly shuffles the data set. If 'rand_seed' is true, then
     * the random seed is the time in microseconds.
     */
    void Shuffle(bool rand_seed = false)
    {
      if(_xvec.size() == 0 || _yvec.size() == 0)
	return;
      
      std::vector<uint> smpls = MakeSamples(_xvec.size(), rand_seed);
      std::vector<InputType> _xvec_new(_xvec.size());
      std::vector<OutputType> _yvec_new(_yvec.size());
      for(uint i=0; i<smpls.size(); ++i){
	_xvec_new[i] = _xvec[smpls[i]];
	_yvec_new[i] = _yvec[smpls[i]];
      }

      _xvec = _xvec_new;
      _yvec = _yvec_new;
    }

    /*!
     * Deletes all stored data
     */
    void Clear()
    {
      _xvec.clear();
      _yvec.clear();
      _in_dim = _out_dim = 0;
    }

    /*!
     * Reads the data from an ASCII file
     */
    int Read(std::string filename, int pos = 0)
    {
      return Read(InputType(), OutputType(), filename, pos);
    }

    /*!
     * Exports the data set into a file, that can be loaded,
     * e.g. by gnuplot
     */
    void Write(std::string filename) const
    {
      //APPEND_FILE(gfile, filename.c_str());
      WRITE_FILE(gfile, filename.c_str());
      gfile << _xvec.size() << " "
          << (_xvec.size() != 0 ? _xvec[0].size() : 0) << std::endl;
      for(uint i=0; i<_xvec.size(); ++i){
	gfile << _xvec[i] << " " << _yvec[i] << std::endl;
      }
      gfile.close();
      //Write(InputType(), OutputType(), filename);
    }

    /*!
     * Samples a given fraction of the data. A specific instantiaion of this
     * methods exists for classification data,  i.e. where the OutputType is
     * 'int'. This guarantees that from each class the same fraction of data
     * is sampled.
     */
     GP_DataSet<InputType, OutputType> DownSample(double fac, bool rand_seed = false)
    {
      InputType x;
      OutputType y;
      return DownSample(x, y, fac, rand_seed);
    }

    /*!
     * Creates a vector of randomly sampled unsigned integers 
     * between 0 and  'nb_samples', where every value appears
     * appears exactly once. If 'rand_seed' is true, the seed
     * of the random number generator is the time in usec.
     */
    static std::vector<uint> MakeSamples(uint nb_samples, bool rand_seed = false)
    {
      std::vector<uint> smpls(nb_samples);
      std::vector<uint> idxvec(nb_samples);
      for(uint i=0; i<nb_samples; ++i)
	idxvec[i] = i;
      
      gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
      
      if(rand_seed){
	struct timeval tv;
	gettimeofday(&tv, 0);
	gsl_rng_set (rng, tv.tv_usec);
      }
      
      for(uint i=0; i<nb_samples; ++i){
	uint smpl = gsl_rng_uniform_int(rng, idxvec.size());
	smpls[i] = idxvec[smpl];
	idxvec[smpl] = idxvec.back();
	idxvec.pop_back();
      }
      
      gsl_rng_free(rng);
      
      return smpls;
    }

#ifdef USE_BOOST_SERIALIZATION
    friend class boost::serialization::access;
    /*!
     * Serializes the object to an archive using boost::serialization
     */
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
         ar & _in_dim & _out_dim;
         ar & _xvec;
         ar & _yvec;
    }
#endif

  private:
  
    uint _in_dim, _out_dim;
    std::vector<InputType> _xvec;
    std::vector<OutputType> _yvec;

    uint GetDimension(int const &scalar) const
    {
      return 1;
    };

    uint GetDimension(uint const &scalar) const
    {
      return 1;
    };

    uint GetDimension(double const &scalar) const
    {
      return 1;
    };

    uint GetDimension(std::vector<double> const &vec) const
    {
      return vec.size();
    };

    uint GetDimension(GP_Vector const &vec) const
    {
      return vec.Size();
    };

    int Read(GP_Vector x, uint y, std::string filename, int pos = 0) 
    {
      READ_FILE(gfile, filename.c_str());
      gfile.seekg(pos);
      uint size, dim;

      gfile >> size >> dim;

      _xvec.clear();
      _yvec.clear();
      _xvec.resize(size, InputType(dim));
      _yvec.resize(size);

      for(uint i=0; i<size; ++i) {
        for(uint j=0; j<dim; ++j) {
          gfile >> _xvec[i][j];
        }
        gfile >> _yvec[i];
      }
      return gfile.tellg();
    }

    GP_DataSet<InputType, OutputType> 
    DownSample(InputType x, int y, double fac, bool rand_seed = false)
    {
      std::map<int, uint> labels;
      std::map<int, uint>::iterator it;
      std::vector<std::vector<uint> > idxvecs;

      for(uint i=0; i<Size(); ++i){
	int label = GetOutput()[i];
	uint idx = 0;
	it = labels.find(label);
	if(it == labels.end()){
	  idx = idxvecs.size();
	  labels[label] = idx;
	  idxvecs.push_back(std::vector<uint>());
	}
	else
	  idx = it->second;

	idxvecs[idx].push_back(i);
      }

      std::vector<uint> discard(idxvecs.size());
      for(uint i=0; i<discard.size(); ++i)
	discard[i] = (uint)floor(fac * idxvecs[i].size());

      // random sampling of points to be removed from the training data
      gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);

      if(rand_seed){
	struct timeval tv;
	gettimeofday(&tv, 0);
	gsl_rng_set (rng, tv.tv_usec);
      }

      std::vector<uint> to_remove;

      for(uint i=0; i<discard.size(); ++i){
	for(uint j=0; j<discard[i]; ++j){
	  uint smpl = gsl_rng_uniform_int(rng, idxvecs[i].size());
	  to_remove.push_back(idxvecs[i][smpl]);
	  idxvecs[i][smpl] = idxvecs[i].back();
	  idxvecs[i].pop_back();
	}
      }

      gsl_rng_free(rng);

      std::cout << to_remove.size() << " to remove" << std::endl;

      std::vector<InputType> xvec_rem;
      std::vector<OutputType> yvec_rem;


      for(uint i=0; i<to_remove.size(); ++i){
	xvec_rem.push_back(_xvec[to_remove[i]]);
	yvec_rem.push_back(_yvec[to_remove[i]]);
      }
      
      Remove(to_remove);

      GP_DataSet<InputType, OutputType> rem_data;
      rem_data.Append(xvec_rem, yvec_rem);

      return rem_data;
    }

  };

}

#endif
