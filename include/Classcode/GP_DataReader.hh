#ifndef GP_DATA_READER_HH
#define GP_DATA_READER_HH

#include <limits>
#include <iomanip>
#include <dirent.h>
#include <regex.h>

#include "Classcode/GP_InputParams.hh"
#include "Classcode/GP_DataSet.hh"

namespace CLASSCODE {

  /*!
   * Abstract class for reading data files
   */
  template<typename InputType, typename OutputType>
  class GP_DataReader 
  {

  public:

    GP_DataReader(GP_InputParams const &params) : 
    its_params(params) {}

    virtual ~GP_DataReader() {}

    GP_InputParams const &GetParams() const
    {
      return its_params;
    }

    virtual GP_DataSet<InputType, OutputType> 
    Read(bool train = true) const = 0;

  private:

    GP_InputParams its_params;

  };

  /*!
   * This class is used to read laser-style data
   */
  template<typename InputType, typename OutputType>
  class GP_LaserDataReader : public GP_DataReader<InputType, OutputType>
  {
  public:

    typedef GP_DataReader<InputType, OutputType> Super;

    GP_LaserDataReader(GP_InputParams const &params) : 
    GP_DataReader<InputType, OutputType>(params) {}

    virtual GP_DataSet<InputType, OutputType> Read(bool train = true) const
    {
      GP_InputParams params = Super::GetParams();
      uint nb_lines = params.GetNbLinesForTraining();
      OutputType y;

      if(train){
	return readTrainData(params.data_file_name, nb_lines, 
			     params.label1, params.label2, y);
      }
      else {
	return readTestData(params.data_file_name, nb_lines,
			    params.label1, params.label2, y);
      }
    }

  private:

    GP_DataSet<InputType, OutputType> 
    readDataBinary(std::string filename, uint from, uint to,
		   uint class_label1, uint class_label2) const
    {
      std::vector<InputType> xvec;
      std::vector<OutputType> yvec;
  
      std::cout << "opening binary class data file " << filename << std::endl;
      std::cout << from << " " << to << std::endl;

      READ_FILE(ifile, filename.c_str());
      FSKIP_LINE(ifile);
      FSKIP_LINE(ifile);

      uint nb_feat, class_label_id;
      std::vector<double> feat;
      std::string class_name;
      uint line_id = 0;

      while(ifile >> nb_feat){

	feat = std::vector<double>(nb_feat);

	for(uint j=0; j<nb_feat; j++)
	  ifile >> feat[j];
	ifile >> class_label_id >> class_name;
    
	if(line_id >= from && line_id < to){
	  if(class_label_id == class_label1) { 
	    xvec.push_back(feat);
	    yvec.push_back(-1);
	  }
	  else if(class_label_id == class_label2){
	    xvec.push_back(feat);
	    yvec.push_back(1);
	  }
	}
	++line_id;
      }

      if(xvec.size() == 0){
	std::cerr << "WARNING! Empty training data. "
		  << "Did you specify the right class labels?" << std::endl;
      }

      std::cout << "got data" << std::endl;
      GP_DataSet<InputType, OutputType> data;
      data.Append(xvec, yvec);

      return data;
    }

    GP_DataSet<InputType, OutputType> 
    readDataPolyadic(std::string filename, uint64_t from, uint64_t to) const
    {
      std::vector<InputType> xvec;
      std::vector<OutputType> yvec;
  
      std::cout << "opening multi-class data file " << filename << std::endl;
      READ_FILE(ifile, filename.c_str());
      FSKIP_LINE(ifile);
      FSKIP_LINE(ifile);

      uint nb_feat, class_label;
      std::vector<double> feat;
      std::string class_name;
      uint64_t line_id = 0;

      while(ifile >> nb_feat){

	feat = std::vector<double>(nb_feat);

	for(uint j=0; j<nb_feat; j++)
	  ifile >> feat[j];
	ifile >> class_label >> class_name;
    

	if(line_id >= from && line_id < to){
	  xvec.push_back(feat);
	  yvec.push_back(class_label);
	}
	++line_id;
      }

      if(xvec.size() == 0){
	std::cerr << "WARNING! Empty training data. "
		  << "Did you specify the right class labels?" << std::endl;
      }

      GP_DataSet<InputType, OutputType> data;
      data.Append(xvec, yvec);

      return data;
    }

    GP_DataSet<InputType, OutputType> 
    readTrainData(std::string filename, uint64_t nb_lines,
		  uint64_t label1, uint64_t label2, int y) const
    {
      return readDataBinary(filename, 0, nb_lines, label1, label2);
    }
  
    GP_DataSet<InputType, OutputType> 
    readTestData(std::string filename, uint64_t start,
		 uint64_t label1, uint64_t label2, int y) const
    {
      return readDataBinary(filename, start, 
			    std::numeric_limits<uint>::max(), 
			    label1, label2);
    }

    GP_DataSet<InputType, OutputType> 
    readTrainData(std::string filename, uint64_t nb_lines,
		  uint64_t label1, uint64_t label2, uint y) const
    {
      return readDataPolyadic(filename, 0, nb_lines);
    }
  
    GP_DataSet<InputType, OutputType> 
    readTestData(std::string filename, uint64_t start,
		 uint64_t label1, uint64_t label2, uint y) const
    {
      return readDataPolyadic(filename, start, 
			      std::numeric_limits<uint64_t>::max());
    }


  };


  /*!
   * This class is used to read GTSRB data
   */
  template<typename InputType, typename OutputType>
  class GP_GTSRBDataReader : public GP_DataReader<InputType, OutputType>
  {
  public:

    typedef GP_DataReader<InputType, OutputType> Super;

    GP_GTSRBDataReader(GP_InputParams const &params) : 
    GP_DataReader<InputType, OutputType>(params), maxNum(50) {}

    virtual GP_DataSet<InputType, OutputType> 
    Read(bool train = true) const
    {
      GP_InputParams params = Super::GetParams();
      GP_DataSet<InputType, OutputType> dataClass1, dataClass2, data;

      static bool train_file_read = false;
      
      if(train){
	if(!train_file_read){
	  dataClass1 = this->readData<200>(params.data_file_prefix1, 0, 
					   params.gtsrb_sep1, 
					   params.label1, 1);
	  dataClass2 = this->readData<200>(params.data_file_prefix2, 0, 
					   params.gtsrb_sep2, 
					   params.label2, -1);
	  train_file_read = true;
	}
	else {
	  dataClass1 = this->readData<200>(params.eval_file_name, 0, 
					   params.gtsrb_sep1, 
					   params.label3, 1);
	}
      }
      else {
	dataClass1 = this->readData<200>(params.data_file_prefix1, 
					 params.gtsrb_sep1, maxNum,
					 params.label1, 1);
	dataClass2 = this->readData<200>(params.data_file_prefix2, 
					 params.gtsrb_sep2, maxNum, 
					 params.label2, -1);
      }

      data.Append(dataClass1);
      data.Append(dataClass2);
      
      return data;
    }


  protected:

    template<uint NbFeat>
    void readFromFile(std::string filename, uint wanted_label, int given_label,
		      std::vector<InputType> &xvec,
		      std::vector<OutputType> &yvec) const
    {
      std::cout << "rading file " << std::endl;
      uint label;
      std::string comp_str;
      READ_FILE(ifile, filename.c_str());
      
      while(ifile >> label){
	std::vector<double> feat(NbFeat);
	uint idx;
	double comp;
	for(uint j=0; j<feat.size(); ++j){
	  ifile >> comp_str;
	  sscanf(comp_str.c_str(), "%d:%lf", &idx, &comp);
	  feat[idx] = comp;
	}
	if(label == wanted_label){
	  xvec.push_back(feat);
	  yvec.push_back(given_label);
	}
      }

      std::cout << "loaded " << xvec.size() << " feature vectors"  << std::endl;
      ifile.close();
    }

  private:
   
    uint maxNum;

    template<uint NbFeat>
    GP_DataSet<InputType, OutputType> 
    readData(std::string filename_prefix, uint from, uint to,
	     uint wanted_label, int given_label) const
    {
      std::vector<InputType> xvec;
      std::vector<OutputType> yvec;

      for(uint i=from; i<to; ++i){
	std::stringstream fname;
	fname << filename_prefix << std::setw(5) 
	      << std::setfill('0') << i << ".txt";
	std::cout << "reading " << fname.str() << std::endl;

	std::ifstream testfile(fname.str().c_str());
	if(testfile.good()){	  
	  readFromFile<NbFeat>(fname.str(), wanted_label, given_label, xvec, yvec);
	}
      }
    
      GP_DataSet<InputType, OutputType> data;
      data.Append(xvec, yvec);
      return data;
    }
  };


  template<typename InputType, typename OutputType>
  class GP_TLRDataReader : public GP_GTSRBDataReader<InputType, OutputType>
  {
  public:

    typedef GP_GTSRBDataReader<InputType, OutputType> Super;

    GP_TLRDataReader(GP_InputParams const &params) : 
    GP_GTSRBDataReader<InputType, OutputType>(params) {}

    virtual GP_DataSet<InputType, OutputType> 
    Read(bool train = true) const
    {
      GP_InputParams params = Super::GetParams();
      GP_DataSet<InputType, OutputType> data;
      std::vector<InputType> xvec;
      std::vector<OutputType> yvec;
      
      DIR *dir_p = opendir(params.data_file_prefix1.c_str());
      if(dir_p == 0)
	throw GP_EXCEPTION2("Could not open data directory '%s'. It must be "
			    "specified under 'DATA_FILE_PREFIX1' in your config "
			    "file.", params.data_file_prefix1);
      struct dirent *entry_p;

      if(train){

	static bool train_file_read = false;

	if(!train_file_read){
	  while((entry_p = readdir(dir_p))){
	    
	    regex_t pg;
	    if(regcomp(&pg, "training", REG_EXTENDED | REG_ICASE | REG_NOSUB) != 0){
	      throw GP_EXCEPTION("Could not create regular expression.");
	    }
	    
	    if(regexec(&pg, entry_p->d_name, (size_t) 0, NULL, 0) == 0){
	      
	      std::string total_name = 
		params.data_file_prefix1 + '/' + entry_p->d_name;
	      std::cout << "loading file " << total_name << std::endl;
	      readFromFile(total_name, xvec, yvec);
	    }
	    regfree(&pg);
	  }
	  train_file_read = true;
	}
	else {
	  std::cout << "loading evaluation file " 
		    << params.eval_file_name << std::endl;
	  readFromFile(params.eval_file_name, xvec, yvec);
	}
      }

      else if (params.feature_scale == 0){

	static bool test_file_read = false;

	if(!test_file_read){
	  while((entry_p = readdir(dir_p))){
	    
	    regex_t pg;
	    if(regcomp(&pg, "testing", REG_EXTENDED | REG_ICASE | REG_NOSUB) != 0){
	      throw GP_EXCEPTION("Could not create regular expression.");
	    }
	    
	    if(regexec(&pg, entry_p->d_name, (size_t) 0, NULL, 0) == 0){
	      
	      std::string total_name = params.data_file_prefix1 + '/' + entry_p->d_name;
	      std::cout << "loading file " << total_name << std::endl;
	      readFromFile(total_name, xvec, yvec);
	    }
	    regfree(&pg);
	  }
	  test_file_read = true;
	}
      }
      else { // testing with different scales
	
	std::stringstream pattstr;
	pattstr << "scale_" << std::setw(3) 
		<< std::setprecision(1) << std::fixed << params.feature_scale;

	regex_t pg;
	if(regcomp(&pg, pattstr.str().c_str(), 
		   REG_EXTENDED | REG_ICASE | REG_NOSUB) != 0){
	  throw GP_EXCEPTION("Could not create regular expression.");
	}

	static uint frame_id = 0;
	std::vector<std::string> fnames;
	while((entry_p = readdir(dir_p)))
	  fnames.push_back(entry_p->d_name);

	sort(fnames.begin(), fnames.end());

	for(uint i=0; i<fnames.size(); ++i){
	  if(regexec(&pg, fnames[i].c_str(), (size_t) 0, NULL, 0) == 0){
	  
	    std::string total_name = params.data_file_prefix1 + '/' + fnames[i];
	    std::stringstream fnstr;
	    fnstr << "Features_frame_%08d_" << pattstr.str() << ".txt";

	    uint idx;
	    sscanf(fnames[i].c_str(), fnstr.str().c_str(), &idx);
	    if(frame_id == 0 || frame_id < idx){
	      frame_id = idx;
	      std::cout << "loading file " << total_name << std::endl;
	      readFromFile(total_name, xvec, yvec);
	      break;
	    }
	  }
	}
	regfree(&pg);
      }
      
      data.Append(xvec, yvec);

      return data;
    }
    

    template<typename VectorType>
    void ReadVector(std::ifstream &ifile, VectorType &vec) const
    {
      std::string line;
      std::getline(ifile, line);
      
      std::vector<double> values;
      size_t pos = 0;
      do {
	std::string substr;
	size_t next_pos = line.find_first_of(' ', pos+1);
	substr = line.substr(pos, next_pos - pos);
	
	std::istringstream sstream(substr);
	double val;
	uint idx;
	std::string val_str;
	sstream >> val_str;
	sscanf(val_str.c_str(), "%d:%lf", &idx, &val);
	//vec[idx] = val;
	std::cout << val << " " << std::flush;
	values.push_back(val);

	pos = next_pos;
      } while(pos < line.size());

      std::cout << std::endl;

      if(values.size())
	vec = VectorType(values);
    }


    void readFromFile(std::string filename,
		      std::vector<InputType> &xvec, std::vector<OutputType> &yvec) const
    {
      OutputType label;
      std::string comp_str;
      READ_FILE(ifile, filename.c_str());
      
      while(ifile >> label){
	InputType feat;
	ReadVector(ifile, feat);
	//std::cout << "got " << feat.size() << " features" << std::endl;
	xvec.push_back(feat);
	yvec.push_back(label);
      }

      std::cout << "loaded " << xvec.size() << " feature vectors"  << std::endl;
      ifile.close();
    }

  };
}



#endif
