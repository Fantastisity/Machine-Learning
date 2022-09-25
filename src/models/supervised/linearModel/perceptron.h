#include "linearModel.h"

namespace MACHINE_LEARNING {
    class Perceptron : public LinearModel<Perceptron> {
            bool multiclass = 0;
            size_t nlabs;
            std::unordered_set<double> labels; // Unique labels
        public:
            template<typename T, typename R>
            void fit(T&& x, R&& y, const uint8_t verbose = 0) {
                init(std::forward<T>(x), std::forward<R>(y));
                labels = this->y.unique(), nlabs = labels.size();
                bool terminate = 0;
                if (nlabs <= 2) { // Binary classification
                    this->w = Matrix<double>(this->x.colNum(), 1);
                    for (size_t i = 0; i < this->iter && !terminate; ++i) {
                        terminate = 1;
                        for (size_t j = 0, n = this->x.rowNum(); j < n; ++j) {
                            auto x_tmp = this->x(rngSlicer(j, j + 1), rngSlicer(0, this->x.colNum()));
                            if ((this->y(j, 0) * (x_tmp * this->w))(0, 0) < 1) this->w += this->y(j, 0) * x_tmp.trans(), terminate = 0;
                        }
                    }
                } else { // Multiclass classification
                    multiclass = 1;
                    this->w = Matrix<double>(this->x.colNum(), nlabs);
                    for (size_t i = 0; i < this->iter && !terminate; ++i) {
                        terminate = 1;
                        for (size_t j = 0, n = this->x.rowNum(); j < n; ++j) {
                            auto x_tmp = this->x(rngSlicer(j, j + 1), rngSlicer(0, this->x.colNum())), prod = x_tmp * this->w;
                            double max_val = prod(0, 0);
                            size_t ypred_ind = 0, ytrue_ind = std::distance(labels.begin(), labels.find(this->y(j, 0)));
                            for (size_t lab = 1; lab < nlabs; ++lab) if (max_val < prod(0, lab)) max_val = prod(0, ypred_ind = lab);
                            if (ypred_ind != ytrue_ind) {
                                terminate = 0;
                                for (size_t k = 0, m = this->x.colNum(); k < m; ++k) 
                                    this->w(ypred_ind, k) -= this->eta * x_tmp(0, k),
                                    this->w(ytrue_ind, k) += this->eta * x_tmp(0, k);
                            }
                        }
                    }
                }
                if (verbose) {
                    if (verbose == 2) print_params();
                    print_weights();
                }
            }

            template<typename T>
            Matrix<double> predict(T&& xtest) {
                Matrix<double> tmp{0};
                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.template asType<double>();
                
                size_t nrow = tmp.rowNum();
                tmp.addCol(std::vector<double>(nrow, 1.0).data());
                tmp = tmp * this->w;

                if (!multiclass) for (size_t i = 0; i < nrow; ++i) tmp(i, 0) = tmp(i, 0) < 0 ? -1 : 1;
                else {
                    Matrix<double> res(nrow, 1);
                    for (size_t i = 0; i < nrow; ++i) {
                        double max_val = tmp(i, 0);
                        size_t ypred_ind = 0;
                        for (size_t lab = 1; lab < nlabs; ++lab) if (max_val < tmp(i, lab)) max_val = tmp(i, ypred_ind = lab);
                        res(i, 0) = *std::next(labels.begin(), ypred_ind);
                    }
                    return res;
                }
                return tmp;
            }
    };
}