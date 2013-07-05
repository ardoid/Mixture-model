#ifndef GP_UNIVERSAL_DATA_READER_HH
#define GP_UNIVERSAL_DATA_READER_HH

#include <fstream>
#include "Classcode/GP_DataReader.hh"


namespace CLASSCODE {

  template<typename InputType, typename OutputType>
  class GP_UniversalDataReader : public GP_DataReader<InputType, OutputType>
  {
  public:

    typedef GP_DataReader<InputType, OutputType> Super;

    GP_UniversalDataReader(GP_InputParams const &params) :
      GP_DataReader<InputType, OutputType>(params) {}

    virtual GP_DataSet<InputType, OutputType> 
    Read(bool train = true) const
    {
      std::vector<InputType> xvec;
      std::vector<OutputType> yvec;

      GP_InputParams params = Super::GetParams();
      std::string filename;
      if(train) {
	filename = params.train_file_name;
      }
      else {
	filename = params.test_file_name;
      }

      std::cout << "loading " << filename << std::endl;

      std::string fext;
      uint d = filename.find_last_of('.');
      if(d != std::string::npos) 
	fext = filename.substr(d+1);
      else
	throw GP_EXCEPTION("Could not load data file. File name has no file extension.");
      
      READ_FILE(ifile, filename.c_str());

      if(fext == "nfv") { // non-indexed feature vectors
	do {
	  std::vector<double> vec = GP_InputParams::ReadVector(ifile);
	  if(!ifile) break;
	  OutputType out = vec.back();
	  vec.pop_back();
	  xvec.push_back(InputType(vec));
	  yvec.push_back(out);

	} while(1);
      }
      else if (fext == "ifv") {

	OutputType y;
	uint max_len = 0;
	while(ifile >> y) {

	  // we read teh whole line and parse each element
	  std::vector<double> vec;
	  std::string line;
	  std::getline(ifile, line);
	  
	  size_t pos = 0;
	  std::vector<double> vals;
	  uint non_zero_idx, curr_idx = 0;
	  double val;
	  
	  do {
	    std::string substr;
	    size_t next_pos = line.find_first_of(' ', pos+1);
	    substr = line.substr(pos, next_pos - pos);
	    
	    sscanf(substr.c_str(), "%d:%lf", &non_zero_idx, &val);
	    if(non_zero_idx >= curr_idx)
	      vals.resize(non_zero_idx + 1);
	    vals[non_zero_idx] = val;
	    pos = next_pos;
	    
	  } while(pos < line.size());
	  
	  max_len = MAX(vals.size(), max_len);
	  if(max_len > vals.size())
	    vals.resize(max_len);

	  xvec.push_back(InputType(vals));
	  yvec.push_back(y);
	}
      }
      
      std::cout << "loaded " << xvec.size() << " feature vectors." << std::endl;
      GP_DataSet<InputType, OutputType> data;
      data.Append(xvec, yvec);

      return data;
    }

  private:
    
  };

}


#endif
