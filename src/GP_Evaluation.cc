#include "Classcode/GP_Evaluation.hh"
#include "Classcode/GP_Constants.hh"
#include <math.h>
#include <fstream>

namespace CLASSCODE {

  double GP_Evaluation::f_measure(double beta, double tp, double fp, double fn)
  {
    double beta_sqr = SQR(beta);
    
    return (1 + beta_sqr) * tp / ((1+beta_sqr) * tp + beta_sqr * fn + fp);
  }

  double GP_Evaluation::calc_norm_entropy(double prob)
  {
    if(prob < EPSILON || 1 - prob < EPSILON)
      return 0;

    return -(prob * ::log(prob) + (1.-prob) * ::log(1.-prob)) / LOG2;
  }

  void GP_Evaluation::plot_pr_curve(std::vector<double> zvals,
		     std::vector<int> yvals,
		     std::string filename, double step_size)
  {
    WRITE_FILE(ofile, filename.c_str());
    for(double thresh = 0.; thresh <= 1.0; thresh += step_size){

      double tp = 0, tn = 0, fp = 0, fn = 0;
      for(uint i=0; i<zvals.size(); i++){
	if(zvals[i] >= thresh && yvals[i] == 1){
	  tp++;
	}
	else if(zvals[i] >= thresh && yvals[i] == -1){
	  fp++;
	}
	else if(zvals[i] < thresh && yvals[i] == 1){
	  fn++;
	}
	else if(zvals[i] < thresh && yvals[i] == -1){
	  tn++;
	}
      }

      double prec, rec, fmeas;
      if(tp + fp == 0)
	prec = 1.0;
      else
	prec = tp / (tp + fp);

      if(tp + fn == 0)
	rec = 1.0;
      else
	rec = tp / (tp + fn);
      
      fmeas = f_measure(0.5, tp, fp, fn);
      ofile << prec << " " << rec << " " << fmeas << " " << thresh << std::endl;
    } 
    ofile.close();
  }


}
