#ifndef MODEL_INCLUDED
#define MODEL_INCLUDED
#include "../models/model.h"
#endif

namespace MACHINE_LEARNING {
    using Param = std::vector<std::pair<char const *, std::initializer_list<elem>>>;
    namespace UTIL_BASE {
        namespace MODEL_UTIL {
            template<typename U, typename R, typename T = void>
            struct isModel {
                const static bool val = 0;
            };

            template<typename U, typename R>
            struct isModel<U, R, Void<decltype(&U::template fit<R, R>)>> {
                const static bool val = 1;
            };

            namespace {
                template<typename T>
                struct EncoderBase {
                    void fit(const DataFrame<T>& m, const size_t col) {
                        size_t n = m.rowNum();
                        for (size_t i = 0; i < n; ++i) 
                            if (!mapping.count(m(i, col))) reverse_mapping[mapping[m(i, col)] = cnt++] = m(i, col);
                        --cnt;
                    }
                    protected:
                        std::unordered_map<T, ll> mapping;
                        std::unordered_map<ll, T> reverse_mapping;
                        ll cnt;
                        bool inplace;
                        EncoderBase(const bool inplace, ll cnt) : inplace(inplace), cnt(cnt) {}
                };
            }

            template<typename T>
            struct LabelEncoder : public EncoderBase<T> {
                LabelEncoder(const bool inplace = 0, const ll cnt = 0) : EncoderBase<T>(inplace, cnt) {}
                void transform(DataFrame<T>& m, const size_t col) {
                    size_t n = m.rowNum();
                    if (this->inplace) for (size_t i = 0; i < n; ++i) m(i, col) = this->mapping[m(i, col)];
                    else {
                        DataFrame<T> tmp(n, 1);
                        for (size_t i = 0; i < n; ++i) tmp.insert(i, 0, this->mapping[m(i, col)]);
                        m.cbind(tmp);
                    }
                }

                void fit_transform(DataFrame<T>& m, const size_t col) {
                    this->fit(m, col), transform(m, col);
                }

                void inverse_transform(DataFrame<T>& m, const size_t col) {
                    size_t n = m.rowNum();
                    for (size_t i = 0; i < n; ++i) m(i, col) = this->reverse_mapping[m(i, col)];
                }
            };

            template<typename T>
            struct OnehotEncoder : public EncoderBase<T> {
                OnehotEncoder(const bool inplace = 0, const ll cnt = 1) : EncoderBase<T>(inplace, cnt) {}
                void transform(DataFrame<T>& m, const size_t col) {
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

                void inverse_transform(DataFrame<T>& m, const size_t col) {
                    size_t n = m.rowNum(), c = m.colNum();
                    if (this->cnt <= 32) this->cnt = 32;
                    else if (this->cnt <= 64) this->cnt = 64;
                    else this->cnt = 128;
                    size_t* remove_col = (size_t*) malloc(sizeof(size_t) * this->cnt);
                    if (!remove_col) {
                        std::cerr << "error malloc\n"; exit(1);
                    }
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
                    free(remove_col);
                }
            };

            namespace METRICS {
                template<typename T, typename R>
                inline auto minkowski(const T* const pointA, const T* const pointB, size_t n, size_t p = 1) 
                -> typename std::enable_if<isNumerical<T>::val && isNumerical<R>::val, double>::type {
                    assert(p);
                    double dist = 0;
                    for (--n; n >= 0; --n) 
                        dist += p > 1 ? 
                                std::pow(pointA[n] > pointB[n] ? pointA[n] - pointB[n] : pointB[n] - pointA[n], p) : 
                                pointA[n] > pointB[n] ? pointA[n] - pointB[n] : pointB[n] - pointA[n];
                    return std::pow(dist, 1 / p);
                }

                template<typename T, typename R>
                inline auto euclidean(const T* const pointA, const T* const pointB, size_t n) 
                -> typename std::enable_if<isNumerical<T>::val && isNumerical<R>::val, double>::type {
                    return minkowski(pointA, pointB, n, 2);
                }

                template<typename T, typename R>
                inline auto manhattan(const T* const pointA, const T* const pointB, size_t n) 
                -> typename std::enable_if<isNumerical<T>::val && isNumerical<R>::val, double>::type {
                    return minkowski(pointA, pointB, n);
                }

                template<typename T>
                inline auto abs(const Matrix<T>& m) 
                -> typename std::enable_if<isNumerical<T>::val, Matrix<T>>::type {
                    size_t r = m.rowNum(), c = m.colNum();
                    Matrix<T> mat(r, c);
                    for (size_t i = 0; i < r; ++i)
                        for (size_t j = 0; j < c; ++j)
                            mat(i, j) = m(i, j) < 0 ? -m(i, j) : m(i, j);
                    return mat;
                }

                template<typename T>
                inline auto sigmoid(const Matrix<T>& m) 
                -> typename std::enable_if<isNumerical<T>::val, Matrix<T>>::type {
                    size_t r = m.rowNum(), c = m.colNum();
                    Matrix<T> mat(r, c);
                    for (size_t i = 0; i < r; ++i)
                        for (size_t j = 0; j < c; ++j) mat(i, j) = 1.0 / (1 + exp(-m(i, j)));
                    return mat;
                }

                template<typename T>
                inline auto loge(const Matrix<T>& m) 
                -> typename std::enable_if<isNumerical<T>::val, Matrix<T>>::type {
                    size_t r = m.rowNum(), c = m.colNum();
                    Matrix<T> mat(r, c);
                    for (size_t i = 0; i < r; ++i)
                        for (size_t j = 0; j < c; ++j) 
                            mat(i, j) = log(m(i, j) > 0 ? m(i, j) : 1e-6);
                    return mat;
                }

                template<typename T>
                inline auto sign(const Matrix<T>& m) 
                -> typename std::enable_if<isNumerical<T>::val, Matrix<T>>::type {
                    size_t r = m.rowNum(), c = m.colNum();
                    Matrix<T> mat(r, c);
                    for (size_t i = 0; i < r; ++i)
                        for (size_t j = 0; j < c; ++j)
                            mat(i, j) = m(i, j) < 0 ? -1 : (!mat(i, j) ? 0 : 1);
                    return mat;
                }

                template<typename T>
                inline auto sum(const Matrix<T>& m)
                -> typename std::enable_if<isNumerical<T>::val, T>::type {
                    T res = 0.0;
                    size_t r = m.rowNum(), c = m.colNum();
                    for (size_t i = 0; i < r; ++i)
                        for (size_t j = 0; j < c; ++j)
                            res += m(i, j);
                    return res;
                }

                template<typename T, typename R>
                inline double RMSE(const Matrix<T>& ypred, const Matrix<R>& ytest) {
                    Matrix<double> tmp;
                    if constexpr (std::is_same<T, double>::value && std::is_same<R, double>::value) 
                        tmp = ypred - ytest;
                    else if constexpr (std::is_same<T, elem>::value && std::is_same<R, elem>::value) 
                        tmp = ypred.template asType<double>() - ytest.template asType<double>();
                    else if constexpr (std::is_same<T, elem>::value) tmp = ypred.template asType<double>() - ytest;
                    else tmp = ypred - ytest.template asType<double>();
                    return std::sqrt((tmp.trans() * tmp / ypred.rowNum())(0, 0));
                }

                template<typename T, typename R>
                inline double ACCURACY(const Matrix<T>& ypred, const Matrix<R>& ytest) {
                    size_t r = ytest.rowNum(), cnt = 0;
                    for (size_t i = 0; i < r; ++i) if (static_cast<double>(ypred(i, 0)) == static_cast<double>(ytest(i, 0))) ++cnt;
                    return 1.0 * cnt / r;
                }
            }

            template<typename T>
            struct BallTree {
                BallTree(const Matrix<T>& mat, const size_t leaf_size, const char* metric = "euclid", size_t p = 2) 
                : mat(mat), leaf_size(leaf_size), p(p) {
                    strcpy(this->metric, metric);
                    if (!(centroid = (T*) calloc(mat.colNum(), sizeof(T)))) {
                        std::cerr << "error calloc\n"; exit(1);
                    }
                }
                BallTree(const Matrix<T>&& mat, const size_t leaf_size, const char* metric = "euclid", size_t p = 2) 
                : mat(mat), leaf_size(leaf_size), p(p) {
                    strcpy(this->metric, metric);
                    if (!(centroid = (T*) calloc(mat.colNum(), sizeof(T)))) {
                        std::cerr << "error calloc\n"; exit(1);
                    }
                }
                BallTree(const BallTree&) = delete;
                BallTree(BallTree&&) = delete;
                BallTree& operator= (const BallTree&) = delete;
                BallTree& operator= (BallTree&&) = delete;
                ~BallTree() {
                    free(centroid);
                }
                private:
                    Matrix<T> mat;
                    size_t leaf_size, p;
                    char metric[7];
                    struct node {
                        node* left_child, * right_child;
                        T* centroid;
                        double radius;
                        ~node() {
                            if (left_child) {
                                delete left_child;
                                left_child = nullptr;
                            }
                            if (right_child) {
                                delete right_child; 
                                right_child = nullptr;
                            }
                            if (centroid) {
                                delete[] centroid;
                                centroid = nullptr;
                            }
                        }
                    };

                    node* init_node(size_t ncol) {
                        node* tmp = new node;
                        tmp->left_child = tmp->right_child = nullptr;
                        tmp->centroid = new T[ncol]{};
                        tmp->radius = 0;
                        return tmp;
                    }
                    
                    node* partition(size_t ind[], size_t nrow) {
                        size_t ncol = mat.colNum();
                        node* root = init_node(ncol);
                        // Find centroid
                        for (size_t i = 0; i < nrow; ++i) 
                            for (size_t j = 0; j < ncol; ++j) {
                                root->centroid[j] += root->centroid[j];
                                if (i == nrow - 1) root->centroid[j] /= nrow;
                            }
                        // Locate the first node that is the furtherest from centroid
                        size_t first_node_ind;
                        for (size_t i = 0; i < nrow; ++i) {
                            double dist;
                            if (!strcmp(metric, "euclid")) dist = METRICS::euclidean(mat(ind[i], 0), centroid, ncol);
                            else if (!strcmp(metric, "manhat")) dist = METRICS::manhattan(mat(ind[i], 0), centroid, ncol);
                            else dist = METRICS::minkowski(mat(ind[i], 0), centroid, ncol, p);

                            if (dist > root->radius) root->radius = dist, first_node_ind = ind[i];
                        }

                        // Locate the second node that is the furtherest from first node
                        size_t second_node_ind;
                        double max_dist = 0;
                        for (size_t i = 0; i < nrow; ++i) {
                            double dist;
                            if (!strcmp(metric, "euclid")) dist = METRICS::euclidean(mat(ind[i], 0), mat(first_node_ind, 0), ncol);
                            else if (!strcmp(metric, "manhat")) dist = METRICS::manhattan(mat(ind[i], 0), mat(first_node_ind, 0), ncol);
                            else dist = METRICS::minkowski(mat(ind[i], 0), mat(first_node_ind, 0), ncol, p);

                            if (dist > max_dist) max_dist = dist, second_node_ind = ind[i];
                        }

                        // Project data on to (first_node - second_node)
                        T diff[ncol], mid; memset(diff, 0, sizeof(diff));
                        std::vector<std::pair<T, size_t>> Z(nrow, std::make_pair(0, 0));
                        for (size_t i = 0; i < ncol; ++i) diff[i] = mat(first_node_ind, i) - mat(second_node_ind, i);
                        for (size_t i = 0; i < nrow; ++i) {
                            for (size_t j = 0; j < ncol; ++i) {
                                Z[i].first += diff[j] * mat(ind[i], j);
                            }
                        }

                        // Construct left and right child indices
                        sort(Z.begin(), Z.end());
                        size_t mid = nrow >> 1;
                        size_t left_child_indices[mid], right_child_indices[nrow - mid];
                        for (size_t i = 0; i < mid; ++i) 
                            left_child_indices[i] = Z[i].second, right_child_indices[i + mid] = Z[i + mid].second;

                        root->left = partition(left_child_indices, mid);
                        root->right = partition(right_child_indices, nrow - mid);

                        return root;
                    }
            };

            std::tuple<DataFrame<elem>, DataFrame<elem>, DataFrame<elem>, DataFrame<elem>>
            train_test_split(DataFrame<elem> X, DataFrame<elem> Y, const float test_size = 0.25, 
                             const bool shuffle = 0, size_t random_state = 0);

            

            std::tuple<size_t*, size_t*, size_t*, size_t> 
            k_fold(const size_t k, const size_t sample_size);

            template<typename M>
            inline double cross_validation(SupervisedModel<M>& estimator, DataFrame<elem>& X, DataFrame<elem>& Y, 
                                    const char* scoring = "RMSE", size_t k = 5) {
                if (k > X.rowNum()) k = 1;
                auto [indTrain, indTest, range, n] = k_fold(k, X.rowNum()); 
                double score = 0;
                auto x = X.values().template asType<double>(), y = Y.values().template asType<double>();
                for (size_t i = 0, xcol = X.colNum(), ycol = Y.colNum(); i < k; ++i) {
                    auto xtrain = x(ptrSlicer(indTrain + i * n * (k - 1), range[i << 1]), rngSlicer(xcol)), 
                         ytrain = y(ptrSlicer(indTrain + i * n * (k - 1), range[i << 1]), rngSlicer(ycol)),
                         xtest  = x(ptrSlicer(indTest + i * n, range[(i << 1) + 1]),      rngSlicer(xcol)), 
                         ytest  = y(ptrSlicer(indTest + i * n, range[(i << 1) + 1]),      rngSlicer(ycol));
                    estimator.fit(xtrain, ytrain);
                    if (!strcmp(scoring, "RMSE"))     score += METRICS::RMSE(estimator.predict(xtest), ytest);
                    else if (!strcmp(scoring, "ACC")) score += METRICS::ACCURACY(estimator.predict(xtest), ytest);
                }
                free(indTrain), free(indTest), free(range);
                return score / k;
            }

            template<typename M>
            inline std::pair<std::vector<std::pair<char const *, elem>>, double> 
            grid_search(SupervisedModel<M>& estimator, Param param_grid, DataFrame<elem>& X, DataFrame<elem>& Y) {
                std::vector<std::pair<char const *, elem>> tmp;
                std::vector<std::vector<std::pair<char const *, elem>>> param_comb;
                size_t len = param_grid.size();
                auto gen_comb = [&](size_t pos, auto&& gen_comb) {
                    if (pos == len) {
                        param_comb.push_back(tmp);
                        return;
                    }
                    for (auto& i : param_grid[pos].second) {
                        tmp.push_back(std::make_pair(param_grid[pos].first, i));
                        gen_comb(pos + 1, gen_comb);
                        tmp.pop_back();
                    }
                };
                gen_comb(0, gen_comb);

                double best_err = -1;
                size_t best_ind = -1;
                for (size_t i = 0, n = param_comb.size(); i < n; ++i) {
                    estimator.set_params(param_comb[i]);
                    double err = cross_validation(estimator, X, Y);
                    if (best_err == -1 || best_err > err) best_err = err, best_ind = i;
                }
                return std::make_pair(param_comb[best_ind], best_err);
            }
        }
    }
}