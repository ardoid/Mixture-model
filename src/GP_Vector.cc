#include "Classcode/GP_Vector.hh"
#include "Classcode/GP_Constants.hh"
#include "Classcode/GP_Exception.hh"

#include <iomanip>
#include <fstream>
#include <math.h>

namespace CLASSCODE {


  
  double GP_Vector::operator[](uint i) const
  {
    if(i >= _data.size())
      throw GP_EXCEPTION2("Invalid index %d into vector", i);
    return _data[i];
  }

  double &GP_Vector::operator[](uint i)
  {
    if(i >= _data.size())
      throw GP_EXCEPTION2("Invalid index %d into vector", i);
    return _data[i];
  }

  GP_Vector GP_Vector::operator*(double x) const
  {
    GP_Vector out(_data.size());
    for(uint i=0; i<_data.size(); ++i)
      out[i] = _data[i] * x;
    return out;
  }
  
  GP_Vector GP_Vector::operator/(double x) const
  {
    if(fabs(x) < 1e-15)
      throw GP_EXCEPTION("Division by zero.");

    GP_Vector out(_data.size());
    for(uint i=0; i<_data.size(); ++i)
      out[i] = _data[i] / x;
    return out;    
  }

  GP_Vector GP_Vector::operator+(GP_Vector const &other) const
  {
    if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }
    
    GP_Vector out(_data.size());
    for(uint i=0; i<_data.size(); ++i)
      out[i] = _data[i] + other[i];
    return out;
  }
  
  GP_Vector GP_Vector::operator-(GP_Vector const &other) const
  {
    if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }

    GP_Vector out(_data.size());
    for(uint i=0; i<_data.size(); ++i)
      out[i] = _data[i] - other[i];
    return out;  
  }
  
  GP_Vector GP_Vector::operator*(GP_Vector const &other) const
  {
    if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }
  
    GP_Vector out(_data.size());
    for(uint i=0; i<_data.size(); ++i)
      out[i] = _data[i] * other[i];
    return out;
  }
  
  GP_Vector GP_Vector::operator/(GP_Vector const &other) const
  {
    if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }

    GP_Vector out(_data.size());
    for(uint i=0; i<_data.size(); ++i)
      out[i] = _data[i] / other[i];
    return out;
  }

  double GP_Vector::Sum() const
  {
    double sum = 0;
    for(uint i=0; i<_data.size(); ++i)
      sum += _data[i];

    return sum;
  }
  
  double GP_Vector::Dot(GP_Vector const &other) const
  {
    if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }

    double sum = 0;
    for(uint i=0; i<_data.size(); ++i)
      sum += _data[i] * other[i];

    return sum;
  }
  
  double GP_Vector::Norm() const
  {
    return sqrt(Dot(*this));
  }

  GP_Vector GP_Vector::Abs() const
  {
    GP_Vector out(_data.size());
    
    for(uint i=0; i<_data.size(); i++)
      out[i] = fabs(_data[i]);
    
    return out;
  }

  double GP_Vector::Max() const
  {
    if(_data.size() == 0)
      return -HUGE_VAL;

    double m = _data[0];

    for(uint i=0; i<_data.size(); i++)
      if(_data[i] > m)
	m = _data[i];
    
    return m;
  }

  uint GP_Vector::ArgMax() const
  {
    if(_data.size() == 0)
      return 0;

    double max = _data[0];
    uint argmax = 0;

    for(uint i=0; i<_data.size(); i++)
      if(_data[i] > max){
	max = _data[i];
	argmax = i;
      }
    
    return argmax;
  }
  
  GP_Vector GP_Vector::Sqr() const
  {
    GP_Vector out(_data.size());
    
    for(uint i=0; i<_data.size(); i++)
      out[i] = _data[i] * _data[i];
    
    return out;
  }
 
  GP_Vector GP_Vector::Exp() const
  {
    GP_Vector out(_data.size());
    
    for(uint i=0; i<_data.size(); i++)
      out[i] = exp(_data[i]);
    
    return out;
  }
 
  void GP_Vector::Append(double x)
  {
    _data.push_back(x);
  }

  int GP_Vector::Read(std::string filename, int pos)
  {
    READ_FILE(ifile, filename.c_str());
    ifile.seekg(pos);
    uint size;
    ifile >> size;
    _data.clear();
    _data.resize(size);
    for(uint i=0; i<size; ++i){
      ifile >> _data[i];
    }

    return ifile.tellg();
  }

  void GP_Vector::Write(std::string filename) const
  {
    APPEND_FILE(ofile, filename.c_str());
    ofile << _data.size() << std::endl;
    for(uint i=0; i<_data.size(); ++i)
      ofile << _data[i] << " ";
    ofile << std::endl;
    ofile.close();
  }


  GP_Vector operator* (double x, GP_Vector const &m)
  {
    return m * x;
  }

  GP_Vector operator/ (double x, GP_Vector const &m)
  {
    GP_Vector out(m.Size());

    for(uint i=0; i<out.Size(); ++i)
	out[i] = x / m[i];

    return out;
  }

  std::ostream &operator<<(std::ostream &stream, GP_Vector const &vec)
  {
    stream.precision(4);
    stream.setf(std::ios::fixed, std::ios::floatfield);

    stream << "\t";
    for(uint i=0; i<vec.Size(); i++){
      stream << std::setw(7) << vec[i] << " ";
    }
    stream << std::flush;
	  
    return stream;
  }

  std::istream &operator>>(std::istream &stream, GP_Vector &vec)
  {
    for(uint i=0; i<vec.Size(); ++i)
      stream >> vec[i];
    return stream;
  }
}
