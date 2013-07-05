
#ifndef GP_CONSTANTS_HH
#define GP_CONSTANTS_HH

#define LOG2   0.69314718055994528623
#define LOG2PI 1.8378770664093453391
#define SQRT2  1.4142135623730951455
#define SQRT3  1.7320508075688772935
#define BALDC  1.04345246425115179
#define EPSILON 1e-15


#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

#ifndef MIN3
#define MIN3(a,b,c) MIN((a),MIN((b),(c)))
#endif

#ifndef MAX
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif

#ifndef MAX3
#define MAX3(a,b,c) MAX((a),MAX((b),(c)))
#endif

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif

#ifndef CUB
#define CUB(x) ((x)*(x)*(x))
#endif


#ifdef __cplusplus

#define BEGIN_PROGRAM(ac,av) \
   int main(int ac, char **av){ \
     try
#define END_PROGRAM \
     catch(GP_Exception e){ \
       e.Handle(); exit(1);\
     }\
   }

#define READ_FILE(fvar,fname) \
   std::ifstream fvar(fname);\
   if (!fvar) throw GP_Exception("Could not open file %s",fname, __FILE__, __LINE__);
#define WRITE_FILE(fvar,fname) \
  std::ofstream fvar(fname);					\
  if (!fvar) throw GP_Exception("Could not open file %s",fname, __FILE__, __LINE__);
#define APPEND_FILE(fvar,fname) \
  std::ofstream fvar(fname, std::ios::app);				\
  if (!fvar) throw GP_Exception("Could not open file %s",fname, __FILE__, __LINE__);
#define FSKIP_LINE(f) \
   {char c=' ';while(c != '\n' && !(f).eof()) (f).get(c);}

#define DEL_FEXT(fname) \
   {unsigned long int d = fname.find_last_of('.'); \
     if (d!=std::string::npos) fname = fname.substr(0,d);}


#endif




#endif
