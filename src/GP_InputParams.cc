#include "Classcode/GP_InputParams.hh"
#include "Classcode/GP_Constants.hh"

#include <math.h>
#include <wordexp.h>
#include <limits>
#include <fstream>


namespace CLASSCODE {

  void GP_InputParams::Read(std::string const &filename)
  {
    eval_file_name = "none";
    feature_scale = 0;
    READ_FILE(ifile, filename.c_str());
    std::string tag;
    while(ifile >> tag){
      if(tag == "TRAIN_FRAC"){
	ifile >> train_frac;
      }
      else if(tag == "DATA_FORMAT"){
	std::string format;
	ifile >> format;
	if(format == "LASER")
	  data_format = LASER_DATA;
	else if(format == "GTSRB"){
	  data_format = GTSRB_DATA;
	}
	else if(format == "TLR"){
	  data_format = TLR_DATA;
	}
      }
      else if(tag == "DATA_FILE_NAME"){
	std::string filename;
	ifile >> filename;
	
	wordexp_t expansion;
	wordexp(filename.c_str(), &expansion, 0);
	data_file_name = std::string(expansion.we_wordv[0]);
	wordfree(&expansion);
      }
      else if(tag == "DATA_FILE_PREFIX1"){
	std::string filename;
	ifile >> filename;
	
	wordexp_t expansion;
	wordexp(filename.c_str(), &expansion, 0);
	data_file_prefix1 = std::string(expansion.we_wordv[0]);
	wordfree(&expansion);
      }
      else if(tag == "DATA_FILE_PREFIX2"){
	std::string filename;
	ifile >> filename;
	
	wordexp_t expansion;
	wordexp(filename.c_str(), &expansion, 0);
	data_file_prefix2 = std::string(expansion.we_wordv[0]);
	wordfree(&expansion);
      }
      else if(tag == "TRAIN_FILE_NAME"){
	std::string filename;
	ifile >> filename;
	
	wordexp_t expansion;
	wordexp(filename.c_str(), &expansion, 0);
	train_file_name = std::string(expansion.we_wordv[0]);
	wordfree(&expansion);
      }
      else if(tag == "TEST_FILE_NAME"){
	std::string filename;
	ifile >> filename;
	
	wordexp_t expansion;
	wordexp(filename.c_str(), &expansion, 0);
	test_file_name = std::string(expansion.we_wordv[0]);
	wordfree(&expansion);
      }
      else if(tag == "EVAL_FILE_NAME"){
	std::string filename;
	ifile >> filename;
	
	wordexp_t expansion;
	wordexp(filename.c_str(), &expansion, 0);
	eval_file_name = std::string(expansion.we_wordv[0]);
	wordfree(&expansion);
      }
      else if(tag == "ALGORITHM_NAME"){
	std::string algname;
	ifile >> algname;
	if(algname == "STANDARD_GP"){
	  algorithm_name = STANDARD_GP;
	}
	else if(algname == "IVM"){
	  algorithm_name = IVM;
	}
      }
      else if(tag == "GTSRB_SEP1"){
	ifile >> gtsrb_sep1;
      }
      else if(tag == "GTSRB_SEP2"){
	ifile >> gtsrb_sep2;
      }
      else if(tag == "INIT_SIGMA_F_SQR"){
	ifile >> init_sigma_f_sqr;
      }
      else if(tag == "INIT_LENGTH_SCALE"){
	ifile >> init_length_scale;
      }
      else if(tag == "INIT_SIGMA_N_SQR"){
	ifile >> init_sigma_n_sqr;
      }
      else if(tag == "LAMBDA"){
	ifile >> lambda;
      }
      else if(tag == "ACTIVE_SET_FRAC"){
	ifile >> active_set_frac;
      }
      else if(tag == "NB_ITERATIONS"){
	ifile >> nb_iterations;
      }
      else if(tag == "NB_QUESTIONS"){
	ifile >> nb_questions;
      }
      else if(tag == "BATCH_SIZE"){
	ifile >> batch_size;
      }
      else if(tag == "MAX_ENTROPY"){
	ifile >> max_entropy;
      }
      else if(tag == "LABEL1"){
	ifile >> label1;
      }
      else if(tag == "LABEL2"){
	ifile >> label2;
      }
      else if(tag == "LABEL3"){
	ifile >> label3;
      }
      else if(tag == "DO_OPTIMIZATION"){
	ifile >> do_optimization;
      }
      else if(tag == "RELEARN"){
	std::string scorename;
	ifile >> scorename;
	if(scorename == "FULL"){
	  relearn = FULL;
	}
	else if(scorename == "ONLY_AS"){
	  relearn = ONLY_AS;
	}
	else if(scorename == "PASSIVE"){
	  relearn = PASSIVE;
	}
      }
      else if(tag == "FORGET"){
	ifile >> forget;
      }
      else if(tag == "FORGET_MODE"){
	std::string mode;
	ifile >> mode;
	if(mode == "MIN_DELTA_H"){
	  forget_mode = MIN_DELTA_H;
	}
	else if(mode == "MIN_VAR"){
	  forget_mode = MIN_VAR;
	}
      }
      else if(tag == "MAX_TRAIN_DATA_SIZE"){
	ifile >> max_train_data_size;
      }
      else if(tag == "USE_EP"){
	ifile >> useEP;
      }
      else if(tag == "DIM_RED_PCA"){
	ifile >> dim_red_pca; 
      }
      else if(tag == "KPARAM_LOWER_BOUNDS"){
	kparam_lower_bounds = ReadVector(ifile);
      }
      else if(tag == "KPARAM_UPPER_BOUNDS"){
	kparam_upper_bounds = ReadVector(ifile);
      }
      else if(tag == "RETRAINING_SCORE"){
	std::string scorename;
	ifile >> scorename;
	if(scorename == "EIG"){
	  retraining_score = EIG;
	}
	else if(scorename == "NE"){
	  retraining_score = NE;
	}
	else if(scorename == "BALD"){
	  retraining_score = BALD;
	}
	else if(scorename == "RAND"){
	  retraining_score = RAND;
	}
      }
      else if(tag == "FEATURE_SCALE"){
	ifile >> feature_scale; 
      }
      else if(tag == "SHUFFLE"){
	ifile >> shuffle; 
      }
    }
    ifile.close();
  }

  void GP_InputParams::Write(std::string filename) const
  {
    WRITE_FILE(ofile, filename.c_str());

    ofile << "TRAIN_FRAC\t\t\t" << train_frac << std::endl;
    ofile << "DATA_FORMAT\t\t\t" 
	  << (data_format == LASER_DATA ? "LASER" : 
	      (data_format == GTSRB_DATA ? "GTSRB" : "TLR" )) << std::endl;
    ofile << "DATA_FILE_NAME\t\t\t" << data_file_name << std::endl;
    ofile << "DATA_FILE_PREFIX1\t\t" << data_file_prefix1 << std::endl;
    ofile << "DATA_FILE_PREFIX2\t\t" << data_file_prefix2 << std::endl;
    ofile << "EVAL_FILE_NAME\t\t\t" << eval_file_name << std::endl;
    ofile << "ALGORITHM_NAME\t\t\t" 
	  << (algorithm_name == STANDARD_GP ? "STANDARD_GP" :
	      (algorithm_name == IVM ? "IVM" : "UNKOWN")) << std::endl;
    ofile << "GTSRB_SEP1\t\t\t" << gtsrb_sep1 << std::endl;
    ofile << "GTSRB_SEP2\t\t\t" <<  gtsrb_sep2 << std::endl;
    ofile << "INIT_SIGMA_F_SQR\t\t" << init_sigma_f_sqr << std::endl;
    ofile << "INIT_LENGTH_SCALE\t\t" << init_length_scale << std::endl;
    ofile << "INIT_SIGMA_N_SQR\t\t" << init_sigma_n_sqr << std::endl;
    ofile << "LAMBDA\t\t\t\t" << lambda << std::endl;
    ofile << "ACTIVE_SET_FRAC\t\t\t" << active_set_frac << std::endl;
    ofile << "BATCH_SIZE\t\t\t" << batch_size << std::endl;
    ofile << "MAX_ENTROPY\t\t\t" << max_entropy << std::endl;
    ofile << "NB_ITERATIONS\t\t\t" << nb_iterations << std::endl;
    ofile << "NB_QUESTIONS\t\t\t" << nb_questions << std::endl;
    ofile << "LABEL1\t\t\t\t" << label1 << std::endl;
    ofile << "LABEL2\t\t\t\t" << label2 << std::endl;
    ofile << "LABEL3\t\t\t\t" << label3 << std::endl;
    ofile << "DO_OPTIMIZATION\t\t\t" << do_optimization << std::endl;
    ofile << "RELEARN\t\t\t\t" 
	  << (relearn == FULL ? "FULL" :
	      (relearn == ONLY_AS ? "ONLY_AS" : 
	       (relearn == PASSIVE ? "PASSIVE" :"UNKNOWN"))) << std::endl;
    ofile << "FORGET\t\t\t\t" << forget << std::endl;    
    ofile << "FORGET_MODE\t\t\t" 
	  << (forget_mode == MIN_DELTA_H ? "MIN_DELTA_H" :
	      (forget_mode == MIN_VAR ? "MIN_VAR" : "UNKNOWN")) << std::endl;
    ofile << "MAX_TRAIN_DATA_SIZE\t\t" << max_train_data_size << std::endl;
    ofile << "USE_EP\t\t\t\t" << useEP << std::endl;
    ofile << "DIM_RED_PCA\t\t\t" << dim_red_pca << std::endl;
    ofile << "KPARAM_LOWER_BOUNDS\t\t";
    for(uint i=0; i<kparam_lower_bounds.size(); ++i)
      ofile << kparam_lower_bounds[i] << " " << std::flush;
    ofile << std::endl; 
    ofile << "KPARAM_UPPER_BOUNDS\t\t";
    for(uint i=0; i<kparam_upper_bounds.size(); ++i)
      ofile << kparam_upper_bounds[i] << " " << std::flush;
    ofile << std::endl; 
    ofile << "RETRAINING_SCORE\t\t" 
	  << (retraining_score == EIG ? "EIG" :
	      (retraining_score == NE ? "NE" : 
	       (retraining_score == BALD ? "BALD" :
		(retraining_score == RAND ? "RAND" :"UNKNOWN")))) << std::endl;
    ofile << "FEATURE_SCALE\t\t\t" << feature_scale << std::endl;
    ofile << "SHUFFLE\t\t\t\t" << shuffle << std::endl;
    ofile.close();
  }

  uint64_t GP_InputParams::GetNbLines() const
  {
    uint64_t number_of_lines = 0;
    std::string line;
    std::ifstream ifile(data_file_name.c_str());
  
    while (std::getline(ifile, line))
      ++number_of_lines;

    return number_of_lines;
  }

  uint64_t GP_InputParams::GetNbLinesForTraining() const
  {
    uint64_t number_of_lines = GetNbLines();
    return (uint64_t) ceil(train_frac * number_of_lines);
  }

  std::vector<double> GP_InputParams::GetHyperParamsInit() const
  {
    std::vector<double> hparams;
    hparams.push_back(init_sigma_f_sqr);
    hparams.push_back(init_length_scale);
    if(init_sigma_n_sqr > 0)
      hparams.push_back(init_sigma_n_sqr);

    return hparams;
  }

std::vector<double> GP_InputParams::ReadVector(std::ifstream &ifile)
{
  std::string line;
  std::getline(ifile, line);
  
  size_t pos = 0;
  std::vector<double> vals;
  do {
    std::string substr;
    size_t next_pos = line.find_first_of(' ', pos+1);
    substr = line.substr(pos, next_pos - pos);
    
    std::istringstream sstream(substr);
    double val;
    std::string val_str;
    sstream >> val_str;
    if(val_str == "inf" || val_str == "INF")
      vals.push_back(HUGE_VAL);
    else{
      sscanf(val_str.c_str(), "%lf", &val);
      vals.push_back(val);
    }
    
    pos = next_pos;
  } while(pos < line.size());
  
  std::vector<double> vec(vals.size());
  for(uint64_t i=0; i<vals.size(); ++i)
    vec[i] = vals[i];
  
  return vec;
}
  
  
}
