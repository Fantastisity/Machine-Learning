#ifndef UTIL_BASE_INCLUDED
#define UTIL_BASE_INCLUDED
#include "utilityBase.h"
#endif
#ifndef MODEL_INCLUDED
#define MODEL_INCLUDED
#include "../models/model.h"
#endif

namespace MACHINE_LEARNING {
    class ModelUtil : public UtilBase {
        template <typename...> using Void = void;

        template<typename U, typename R = void>
        struct isModel {
            const static bool val = 0;
        };

        template<typename U>
        struct isModel<U, Void<decltype(&U::fit)>> {
            const static bool val = 1;
        };

        template<typename T>
        class EncoderBase {
            protected:
            std::unordered_map<T, ll> mapping;
            std::unordered_map<ll, T> reverse_mapping;
            ll cnt;
            bool inplace;
            EncoderBase(const bool inplace) : inplace(inplace) { cnt = 1; }
            public:
                void fit(const Matrix<T>& m, const size_t col) {
                    size_t n = m.rowNum();
                    for (size_t i = 0; i < n; ++i) 
                        if (!mapping[m(i, col)]) reverse_mapping[mapping[m(i, col)] = cnt++] = m(i, col);
                    --cnt;
                }
        };
        public:
            template<typename U, typename R = void>
            struct isDataframe {
                const static bool val = 0;
            };

            template<typename U>
            struct isDataframe<U, Void<decltype(&U::loc)>> {
                const static bool val = 1;
            };
            
            template<typename T>
            class LabelEncoder : public EncoderBase<T> {
                public:
                    LabelEncoder(const bool inplace = 0) : EncoderBase<T>(inplace) {}
                    void transform(Matrix<T>& m, const size_t col) {
                        size_t n = m.rowNum();
                        if (this->inplace) for (size_t i = 0; i < n; ++i) m(i, col) = this->mapping[m(i, col)];
                        else {
                            Matrix<T> tmp(n, 1);
                            for (size_t i = 0; i < n; ++i) tmp.insert(i, 0, this->mapping[m(i, col)]);
                            m.cbind(tmp);
                        }
                    }

                    void fit_transform(Matrix<T>& m, const size_t col) {
                        this->fit(m, col), transform(m, col);
                    }

                    void inverse_transform(Matrix<T>& m, const size_t col) {
                        size_t n = m.rowNum();
                        for (size_t i = 0; i < n; ++i) m(i, col) = this->reverse_mapping[m(i, col)];
                    }
            };

            template<typename T>
            class OnehotEncoder : public EncoderBase<T> {
                public:
                    OnehotEncoder(const bool inplace = 0) : EncoderBase<T>(inplace) {}
                    void transform(Matrix<T>& m, const size_t col) {
                        size_t n = m.rowNum(), c = m.colNum();
                        if (this->cnt <= 32) 
                            for (size_t i = 0; i < n; ++i) {
                                std::string bits = std::bitset<32>(this->mapping[m(i, col)]).to_string();
                                for (size_t j = c; j < c + 32; ++j) m.insert(i, j, bits[j - c] - '0');
                            }
                        else if (this->cnt <= 64) 
                            for (size_t i = 0; i < n; ++i) {
                                std::string bits = std::bitset<64>(this->mapping[m(i, col)]).to_string();
                                for (size_t j = c; j < c + 64; ++j) m.insert(i, j, bits[j - c] - '0');
                            }
                        else 
                            for (size_t i = 0; i < n; ++i) {
                                std::string bits = std::bitset<128>(this->mapping[m(i, col)]).to_string();
                                for (size_t j = c; j < c + 128; ++j) m.insert(i, j, bits[j - c] - '0');
                            }
                        
                        if (this->inplace) m.dropFeature(col);
                    }

                    void fit_transform(Matrix<T>& m, const size_t col) {
                        this->fit(m, col), transform(m, col);
                    }

                    void inverse_transform(Matrix<T>& m, const size_t col) {
                        size_t n = m.rowNum(), c = m.colNum();
                        if (this->cnt <= 32) this->cnt = 32;
                        else if (this->cnt <= 64) this->cnt = 64;
                        else this->cnt = 128;
                        size_t* remove_col = (size_t*) malloc(sizeof(size_t) * this->cnt);
                        for (size_t i = 0; i < n; ++i) {
                            ll val = 0;
                            for (size_t j = c - this->cnt; j < c; ++j) {
                                val |= static_cast<ll>(m(i, j));
                                val <<= 1;
                                if (!i) remove_col[j - (c - this->cnt)] = j;
                            }
                            m.insert(i, c, this->reverse_mapping[val >> 1]);
                        }
                        m.dropFeature(remove_col, this->cnt);
                    }
            };

            auto train_test_split(DataFrame<elem> X, DataFrame<elem> Y, const float test_size = 0.25, const bool shuffle = 0) {
                size_t row = X.rowNum(), testRow = row * test_size;
                if (shuffle) X.shuffle(), Y.shuffle();
                return std::make_tuple(X.iloc(rangeSlicer(row - testRow),      rangeSlicer(X.colNum())), 
                                       X.iloc(rangeSlicer(row - testRow, row), rangeSlicer(X.colNum())), 
                                       Y.iloc(rangeSlicer(row - testRow),      rangeSlicer(Y.colNum())), 
                                       Y.iloc(rangeSlicer(row - testRow, row), rangeSlicer(Y.colNum())));
            }

            template<typename T>
            auto abs(const Matrix<T>& m) 
            -> typename std::enable_if<isNumerical<T>::val, Matrix<T>>::type {
                size_t r = m.rowNum(), c = m.colNum();
                Matrix<T> mat(r * c);
                for (size_t i = 0; i < r; ++i)
                    for (size_t j = 0; j < c; ++j)
                        mat.insert(i, j, m(i, j) < 0 ? -m(i, j) : m(i, j));
                return mat;
            }

            template<typename T>
            auto sign(const Matrix<T>& m) 
            -> typename std::enable_if<isNumerical<T>::val, Matrix<T>>::type {
                size_t r = m.rowNum(), c = m.colNum();
                Matrix<T> mat(r * c);
                for (size_t i = 0; i < r; ++i)
                    for (size_t j = 0; j < c; ++j)
                        mat.insert(i, j, m(i, j) < 0 ? -1 : (!mat(i, j) ? 0 : 1));
                return mat;
            }

            template<typename T>
            auto sum(const Matrix<T>& m)
            -> typename std::enable_if<isNumerical<T>::val, T>::type {
                T res = 0.0;
                size_t r = m.rowNum(), c = m.colNum();
                for (size_t i = 0; i < r; ++i)
                    for (size_t j = 0; j < c; ++j)
                        res += m(i, j);
                return res;
            }

            template<typename T>
            double RMSE(const Matrix<T>& ypred, const Matrix<T>& ytest) {
                auto tmp = ypred - ytest;
                ll n = ypred.rowNum();
                return std::sqrt((tmp.trans() * tmp / n)(0, 0));
            }

            auto k_fold(const size_t k, const size_t sample_size) {
                size_t n = static_cast<size_t>(std::ceil(sample_size / (k * 1.0))),
                       * indTrain = (size_t*) malloc(sizeof(size_t) * k * n * (k - 1)),
                       * indTest = (size_t*) malloc(sizeof(size_t) * k * n),
                       * range = (size_t*) malloc(sizeof(size_t) * k * 2);
                if (!indTrain || !indTest || !range) {
                    std::cerr << "error realloc\n"; exit(1);
                }
                for (size_t i = 0, cntTrain, cntTest; i < k; ++i) {
                    cntTrain = 0, cntTest = 0;
                    for (size_t j = 0; j < sample_size; ++j) {
                        if (j >= n * i && cntTest < n) {
                            indTest[i * n + cntTest++] = j;
                            continue;
                        }
                        indTrain[i * n * (k - 1) + cntTrain++] = j;
                    }
                    range[i << 1] = cntTrain, range[(i << 1) + 1] = cntTest;
                }
                return std::make_tuple(indTrain, indTest, range, n);
            }

            template<typename U>
            auto cross_validation(U& estimator, DataFrame<elem>& X, DataFrame<elem>& Y, size_t k = 5) -> 
            typename std::enable_if<isModel<U>::val, double>::type {
                if (k > X.rowNum()) k = 1;
                auto [indTrain, indTest, range, n] = k_fold(k, X.rowNum()); 
                double score = 0;

                for (size_t i = 0; i < k; ++i) {
                    auto xtrain = X.iloc(ptrSlicer(indTrain + i * n * (k - 1), range[i << 1]), rangeSlicer(X.colNum())), 
                         ytrain = X.iloc(ptrSlicer(indTrain + i * n * (k - 1), range[i << 1]), rangeSlicer(Y.colNum())),
                         xtest = X.iloc(ptrSlicer(indTest + i * n, range[(i << 1) + 1]), rangeSlicer(X.colNum())), 
                         ytest = Y.iloc(ptrSlicer(indTest + i * n, range[(i << 1) + 1]), rangeSlicer(Y.colNum()));
                    estimator.fit(xtrain, ytrain);
                    estimator.predict(xtest);
                    score += RMSE(estimator.predict(xtest), ytest.values());
                }
                free(indTrain), free(indTest), free(range);
                return score / k;
            }

            template<typename U>
            auto grid_search(U& estimator, DataFrame<elem>& X, DataFrame<elem>& Y) -> 
            typename std::enable_if<isModel<U>::val>::type {
                std::vector<double> tmp;
                std::vector<std::vector<double>> param_comb, param_grid = estimator.p.gen_param_grid();
                auto gen_comb = [&](size_t pos, auto&& gen_comb) {
                    if (tmp.size() == param_grid.size()) {
                        param_comb.push_back(tmp);
                        return;
                    }
                    for (auto& i : param_comb[pos]) {
                        tmp.push_back(i);
                        gen_comb(pos + 1, gen_comb);
                        tmp.pop_back();
                    }
                };
                gen_comb(0, gen_comb);
            }
    };
    extern ModelUtil modUtil;
}