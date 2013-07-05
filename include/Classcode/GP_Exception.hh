#ifndef GP_EXCEPTION_HH
#define GP_EXCEPTION_HH


#include <exception>
#include <string>
#include <sstream>
#include <sys/types.h>


namespace CLASSCODE {


  class GP_Exception : public std::exception
  {
  public:
    
    GP_Exception(std::string const &message, std::string const &filename,
		 uint line_number) :
      std::exception(), _message(message), _filename(filename), _line_number(line_number)
    {}

    template<typename ArgType>
    GP_Exception(std::string const &message, ArgType const &arg,
		 std::string const &filename,
		 uint line_number) :
      std::exception(), _message(message), _filename(filename), _line_number(line_number)
    {
      std::string subs1, subs2;
      uint sep_pos = _message.find_first_of('%');
      uint end_sep = _message.find_first_of(' ', sep_pos);

      subs1 = _message.substr(0, sep_pos);
      subs2 = _message.substr(end_sep);
      std::stringstream str;
      str << subs1 << arg << subs2;

      _message = str.str();
    }

    virtual ~GP_Exception() throw()
    {}
    
    void Handle() const;


  private:

    std::string _message;
    std::string _filename;
    uint _line_number;
  };

}


#define GP_EXCEPTION(msg) GP_Exception(msg, __FILE__, __LINE__)
#define GP_EXCEPTION2(msg, arg) GP_Exception(msg, arg, __FILE__, __LINE__)

#endif
