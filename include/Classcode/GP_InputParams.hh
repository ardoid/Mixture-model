#ifndef GP_UTILS_HH
#define GP_UTILS_HH

#include <string>
#include <inttypes.h>
#include "Classcode/GP_DataSet.hh"


namespace CLASSCODE {

  struct GP_InputParams
  {
    typedef enum {LASER_DATA, GTSRB_DATA, TLR_DATA} DataFormat;    
    typedef enum {STANDARD_GP, IVM} AlgorithmName;

    /*!
     * Expected Information Gain, Normalized Entropy, 
     * Bayesian Active Learning by Disagreement, or random selection
     */
    typedef enum {EIG, NE, BALD, RAND} RetrainingScore; 

    /*! 
     * There are three modes how the retraining can be done:
     * - FULL re-trains kernel parameters and active set every time
     * - ONLY_AS re-trains only the active set, the kernel parameters
     *   are obtained from the very first training round
     * - PASSIVE training is only done in the very first round, no
     *   test points are added to the training set
     */
    typedef enum {FULL, ONLY_AS, PASSIVE} RetrainingMode; 

    /*!
     * Forgetting can be done either by
     * - using the smallest entropy differences as computed by the IVM
     * - using the smallest posterior variances
     */
    typedef enum {MIN_DELTA_H, MIN_VAR} ForgettingMode;

    double train_frac; //!< the fraction of the data used for training
    DataFormat data_format; //!< laser or camera data (see enum above)
    std::string data_file_name; //!< the file name for laser data (will be removed)
    std::string data_file_prefix1; //!< the prefix for class1 in GTSRB data (will be removed)
    std::string data_file_prefix2; //!< the prefix for class2 in GTSRB data (will be removed)
    std::string train_file_name; //!< the name of the training data file
    std::string test_file_name; //!< the name of the test data file
    std::string eval_file_name; //!< the file name for the evaluation set
    AlgorithmName algorithm_name; //!< the algorithm used (see enum above)
    uint64_t gtsrb_sep1, gtsrb_sep2; //!< the separator for each class in GTSRB data
    double init_sigma_f_sqr; //!< the initial kernel hyper parameters
    double init_length_scale; //!< the initial kernel hyper parameters
    double init_sigma_n_sqr; //!< the initial kernel hyper parameters
    double lambda; //!< the parameter of the sigmoid function (not learned)
    double active_set_frac; //!< the fraction of the active set in the training data
    uint64_t batch_size; //!< number of steps used for a batch update
    double max_entropy; //!< everything more than that gets re-asked
    uint64_t nb_questions; //!< number of questions asked in active learning
    uint64_t nb_iterations; //!< number of iterations for IVM training
    uint64_t label1, label2, label3; //!< labels wanted from the input data
    unsigned short do_optimization; //!< decides whether learning should be done 
                                    //!< (0 = never, 1 = once, 2 = always)
    RetrainingMode relearn; //!< see above
    bool forget; //!< if true, uninformative training points are removed
    ForgettingMode forget_mode; //!< see above
    uint max_train_data_size; //!< only used when 'forget' is true
    bool useEP; //!< decides to use Expectation Propagation in IVM
    uint64_t dim_red_pca; //!< dimensionality reduction, if 0 no reduction is done 
    std::vector<double> kparam_lower_bounds; //!< minimum required for kernel params
    std::vector<double> kparam_upper_bounds; //!< maximum allowed for kernel params
    RetrainingScore retraining_score;
    double feature_scale;
    bool shuffle;  //!< if true, test points are randomly re-ordered

    /*!
     * Reads the input parameters from a file
     */
    void Read(std::string const &filename);

    /*!
     * Writes the input parameters into a file
     */
    void Write(std::string filename) const;

    /*!
     * Returns the total number of lines in the data file
     */
    uint64_t GetNbLines() const;

    /*!
     * Returns the number of lines in the data file used for training
     */
    uint64_t GetNbLinesForTraining() const;

    /*!
     * Returns the intial values for the hyper parameters as loaded
     * from the config file
     */
    std::vector<double> GetHyperParamsInit() const;

    static std::vector<double> ReadVector(std::ifstream &ifile);
  };


}
#endif
