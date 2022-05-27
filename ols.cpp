#ifndef LINEAR_MODEL_INCLUDED
#define LINEAR_MODEL_INCLUDED
#include "include/models/supervisedModel/ols.h"
#endif

namespace MACHINE_LEARNING {
    LinearRegression::LinearRegression() {
        #ifdef WRITE_TO_FILE
            output.open("test.dat");
        #endif
    }

    auto LinearRegression::loss() {
        Matrix<double> tmp = x * w - y, l = tmp.trans() * tmp * 0.5;
        switch (r) {
            case None:
                break;
            case L1:
                l += modUtil.sum(modUtil.abs(w)) * lamb;
                break;
            case L2:
                l += w.trans() * w * lamb * 0.5;
                break;
            case ENet:
                l += w.trans() * w * lamb * 0.5 * (1 - alpha) + 
                     modUtil.sum(modUtil.abs(w)) * lamb * alpha;
                break;
        }
        return l(0, 0);
    }

    auto LinearRegression::gradient(Matrix<double>& X, Matrix<double>& Y) {
        auto grad = X.trans() * (X * w - Y);
        switch (r) {
            case None:
                break;
            case L1:
                grad += modUtil.sign(w) * lamb;
                break;
            case L2:
                grad += (w.trans() * w * lamb)(0, 0);
                break;
            case ENet:
                grad += modUtil.sign(w) * lamb * alpha + (w.trans() * w * lamb)(0, 0) * (1 - alpha);
                break;
        }
        return grad;
    }

    void LinearRegression::gradient_descent() {
        switch (t) {
            case BATCH: {
                for (size_t i = 0; i < iter; ++i) {
                    if (loss() <= eps) break;
                    #ifdef WRITE_TO_FILE
                        output << i << '\t' << std::fixed << l;
                    #endif
                    w -= gradient(x, y) * eta;
                }
                #ifdef WRITE_TO_FILE
                    output.close();
                #endif
                return;
            }
            case STOCHASTIC: {
                size_t n = x.rowNum(), ind[n];
                for (size_t i = 0; i < n; ++i) ind[i] = i;
                #ifdef WRITE_TO_FILE
                    ll cnt = 0;
                #endif
                for (size_t i = 0, term = 0; i < iter && !term; ++i) {
                    std::shuffle(ind, ind + n, std::default_random_engine {});
                    for (auto& j : ind) {
                        if (loss() <= eps) {
                            term = 1;
                            break;
                        }
                        #ifdef WRITE_TO_FILE
                            output << cnt++ << '\t' << std::fixed << l;
                        #endif
                        auto x_t = x(rngSlicer(j, j + 1), rngSlicer(0, x.colNum())),
                             y_t = y(rngSlicer(j, j + 1), rngSlicer(0, y.colNum()));
                        w -= gradient(x_t, y_t) * eta;
                    }
                }
                #ifdef WRITE_TO_FILE
                    output.close();
                #endif
                return;
            }
            case MINI_BATCH: {
                size_t n = x.rowNum(), ind[n];
                while (n / batch_size < 2) batch_size >>= 1;
                if (!batch_size) {
                    logger("switching to BGD due to insufficient data");
                    t = BATCH, gradient_descent();
                }
                for (size_t i = 0; i < n; ++i) ind[i] = i;
                #ifdef WRITE_TO_FILE
                    ll cnt = 0;
                #endif
                for (size_t i = 0, term = 0; i < iter && !term; ++i) {
                    std::shuffle(ind, ind + n, std::default_random_engine {});
                    for (size_t j = 0, num; j < n; j += num) {
                        if (loss() <= eps) {
                            term = 1;
                            break;
                        }
                        #ifdef WRITE_TO_FILE
                            output << cnt++ << '\t' << std::fixed << l;
                        #endif
                        num = std::min(static_cast<size_t>(batch_size), n - j);
                        auto x_t = x(ptrSlicer(ind + j, num), rngSlicer(x.colNum())), 
                                y_t = y(ptrSlicer(ind + j,num), rngSlicer(y.colNum()));
                        w -= gradient(x_t, y_t) * eta;
                    }
                }
                #ifdef WRITE_TO_FILE
                    output.close();
                #endif
                return;
            }
        }
    }

    void LinearRegression::print_params() {
        pretty_print("", '*', 58, '*');
        pretty_print("", ' ', 38, "Parameter Settings");
        pretty_print("", '*', 58, '*');
        switch (t) {
            case BATCH: 
                pretty_print("gradient descent type:", ' ', 29, "BGD");
                break;
            case STOCHASTIC:
                pretty_print("gradient descent type:", ' ', 29, "SGD");
                break;
            case MINI_BATCH:
                pretty_print("gradient descent type:", ' ', 29, "MBGD"),
                pretty_print("batch size:", ' ', 40, batch_size);
                break;
        }
        if (r != None) {
            switch (r) {
                case L1:
                    pretty_print("regularizor:", ' ', 39, "Lasso");
                    break;
                case L2:
                    pretty_print("regularizor:", ' ', 39, "Ridge");
                    break;
                case ENet:
                    pretty_print("regularizor:", ' ', 39, "Elastic Net"),
                    pretty_print("alpha:", ' ', 45, alpha);
                    break;
            };
            pretty_print("lambda:", ' ', 44, lamb);
        }
        
        pretty_print("eta:", ' ', 47, eta);
        pretty_print("epsilon:", ' ', 43, eps);
        pretty_print("iterations:", ' ', 40, iter);
        pretty_print("", '*', 58, '*');
    }

    void LinearRegression::print_weights() {
        pretty_print("", '*', 58, '*');
        pretty_print("", ' ', 35, "Final Weights");
        pretty_print("", '*', 58, '*');
        for (size_t i = 0, r = w.rowNum(); i < r; ++i) pretty_print(i, ' ', 50, w(i, 0));
        pretty_print("", '*', 58, '*');
    }
}