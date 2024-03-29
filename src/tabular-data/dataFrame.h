#ifndef MATRIX_INCLUDED
#define MATRIX_INCLUDED
#include "matrix.h"
#endif

namespace MACHINE_LEARNING {
    template<typename T = elem>
    class DataFrame {
        friend class Parser;
        Matrix<T> dt;
        std::unordered_map<const char*, size_t, CstrFunctor, CstrFunctor> name2ind; // Column name to index
        char** ind2name = nullptr; // Index to column name

        void init_ind2name(size_t n, bool assign = 1) {
            if (ind2name) {
                for (size_t i = 0; i < dt.col; ++i) free(ind2name[i]);
                free(ind2name);
            }
            ind2name = (char**) malloc(n * sizeof(char*));
            if (!ind2name) {
                std::cerr << "error malloc\n"; exit(1);
            }
            if (assign) for (auto& [k, v] : name2ind) ind2name[v] = const_cast<char*>(k);
        }

        void resize_ind2name(size_t size) {
            ind2name = (char**) realloc(ind2name, size);
            if (!ind2name) {
                std::cerr << "error realloc\n"; exit(1);
            }
        }

        void dealloc() {
            if (ind2name) {
                for (size_t i = 0; i < dt.col; ++i) {
                    if (ind2name[i]) free(ind2name[i]), ind2name[i] = nullptr;
                }
                free(ind2name);
                ind2name = nullptr;
            }
            dt.dealloc();
        }
        public:
            DataFrame(size_t cap = 1000, const std::vector<std::string>&& colnames = {}) : dt(cap) {
                if (!colnames.empty()) {
                    size_t ncol = colnames.size();
                    init_ind2name(ncol, 0);
                    for (size_t i = 0; i < ncol; ++i) {
                        char* tmp = strdup(colnames[i].c_str());
                        name2ind[ind2name[i] = tmp] = i;
                    }
                }
            }

            DataFrame(size_t row, size_t col, const std::vector<std::string>&& colnames = {}) : dt(row, col) {
                if (!colnames.empty()) {
                    size_t ncol = colnames.size();
                    init_ind2name(ncol, 0);
                    for (size_t i = 0; i < ncol; ++i) {
                        char* tmp = strdup(colnames[i].c_str());
                        name2ind[ind2name[i] = tmp] = i;
                    }
                }
            }

            DataFrame(T** Mat, const size_t nrow, const size_t ncol, const std::vector<std::string>&& colnames = {}) : dt(Mat, nrow, ncol) {
                if (!colnames.empty()) {
                    init_ind2name(ncol, 0);
                    for (size_t i = 0; i < ncol; ++i) {
                        const char* tmp = strdup(colnames[i].c_str());
                        name2ind[ind2name[i] = tmp] = i;
                    }
                }
            }

            DataFrame(const std::vector<std::vector<T>>&& Mat, const std::vector<std::string>&& colnames = {}) : dt(std::move(Mat)) {
                if (!colnames.empty()) {
                    size_t ncol = Mat[0].size();
                    init_ind2name(ncol, 0);
                    for (size_t i = 0; i < ncol; ++i) {
                        const char* tmp = strdup(colnames[i].c_str());
                        name2ind[ind2name[i] = tmp] = i;
                    }
                }
            }

            DataFrame(const std::initializer_list<std::initializer_list<T>>&& Mat, const std::vector<std::string>&& colnames = {}) 
            : dt(std::move(Mat)) {
                if (!colnames.empty()) {
                    size_t ncol = (*Mat.begin()).size();
                    init_ind2name(ncol, 0);
                    for (size_t i = 0; i < ncol; ++i) {
                        const char* tmp = strdup(colnames[i].c_str());
                        name2ind[ind2name[i] = tmp] = i;
                    }
                }
            }
            
            DataFrame(const Matrix<T>& m, const std::vector<std::string>&& colnames = {}) : dt(m) {
                if (!colnames.empty()) {
                    size_t ncol = dt.col;
                    init_ind2name(ncol, 0);
                    for (size_t i = 0; i < ncol; ++i) {
                        const char* tmp = strdup(colnames[i].c_str());
                        name2ind[ind2name[i] = tmp] = i;
                    }
                }
            }

            DataFrame(Matrix<T>&& m, const std::vector<std::string>&& colnames = {}) : dt(std::move(m)) {
                if (!colnames.empty()) {
                    size_t ncol = dt.col;
                    init_ind2name(ncol, 0);
                    for (size_t i = 0; i < ncol; ++i) {
                        char* tmp = strdup(colnames[i].c_str());
                        name2ind[ind2name[i] = tmp] = i;
                    }
                }
            }

            DataFrame(const std::unordered_map<std::string, std::vector<T>>& cols) {
                size_t cnt = 0;
                init_ind2name(cols.size(), 0);
                for (auto& [k, v] : cols) {
                    const char* tmp = strdup(k.c_str());
                    name2ind[tmp] = cnt, dt.addCol(v.data());
                    ind2name[cnt++] = tmp;
                }
            }

            DataFrame(const DataFrame& df) : dt(df.dt) {
                if (!df.name2ind.empty()) {
                    for (auto& [k, v] : df.name2ind) name2ind[strdup(k)] = v;
                    init_ind2name(df.colNum());
                }
            }

            DataFrame(DataFrame&& df) : dt(std::move(df.dt)) {
                if (!df.name2ind.empty()) {
                    name2ind = df.name2ind, df.name2ind.clear();
                    ind2name = df.ind2name, df.ind2name = nullptr;
                }
            }

            ~DataFrame() {
                dealloc();
            }

            auto& operator()(const size_t r, const size_t c) const {
                return dt(r, c);
            }

            void shuffle(size_t random_state) {
                dt.shuffle(random_state);
            }

            DataFrame sample(size_t n = 1, float frac = 0) {
                return DataFrame(dt.sample(n, frac));
            }

            template<typename R, typename U>
            DataFrame iloc(UTIL_BASE::MATRIX_UTIL::slice<R>&& s1, UTIL_BASE::MATRIX_UTIL::slice<U>&& s2, bool assign_colname = 0) {
                if (s1.size == rowNum() && s2.size == colNum()) return *this;

                if (assign_colname && s2.size < colNum()) {
                    std::vector<std::string> colnames(s2.size);
                    size_t cnt = 0;
                    for (auto i = s2.start; i < s2.end; ++i) {
                        size_t ind;
                        if constexpr (UTIL_BASE::MATRIX_UTIL::isPTR<U>::val) ind = *i;
                        else ind = i;

                        colnames[cnt++] = ind2name[ind];
                    }
                    return DataFrame(dt(std::forward<UTIL_BASE::MATRIX_UTIL::slice<R>>(s1), 
                                        std::forward<UTIL_BASE::MATRIX_UTIL::slice<U>>(s2)), 
                                        std::move(colnames));
                }

                return DataFrame(dt(std::forward<UTIL_BASE::MATRIX_UTIL::slice<R>>(s1), std::forward<UTIL_BASE::MATRIX_UTIL::slice<U>>(s2)));
            }

            template<typename R>
            DataFrame loc(UTIL_BASE::MATRIX_UTIL::slice<R>&& s1, std::vector<std::string> colnames, bool assign_colname = 0) {
                size_t n = colnames.size();
                assert(s1.size <= dt.row && n <= dt.col);
                if (s1.size == dt.row && n == dt.col) return *this;
                size_t inds[n];
                for (size_t i = 0; i < n; ++i) inds[i] = name2ind[colnames[i]];

                return assign_colname ? DataFrame(dt(std::forward<UTIL_BASE::MATRIX_UTIL::slice<R>>(s1), ptrSlicer(inds, n)), std::move(colnames)) :
                                        DataFrame(dt(std::forward<UTIL_BASE::MATRIX_UTIL::slice<R>>(s1), ptrSlicer(inds, n)));
            }

            auto& operator= (const DataFrame& df) {
                dealloc();
                dt = df.dt;
                if (df.ind2name) {
                    for (auto& [k, v] : df.name2ind) name2ind[strdup(k)] = v;
                    init_ind2name(df.colNum());
                }
                return *this;
            }
            
            auto& operator= (DataFrame&& df) {
                dealloc();
                dt = std::move(df.dt);
                if (df.ind2name) {
                    name2ind = df.name2ind, df.name2ind.clear();
                    ind2name = df.ind2name, df.ind2name = nullptr;
                }
                return *this;
            }

            // Returns the mean of a column
            double mean(const size_t c) const {
                assert(c < colNum());
                if constexpr (std::is_same<T, elem>::value) assert(dt(0, c).t != evType::STR);
                else assert(UTIL_BASE::isNumerical<T>::val);
                double sum = 0;
                size_t nrow = rowNum();
                for (size_t i = 0; i < nrow; ++i) sum += dt(i, c);
                return sum / nrow;
            }

            // Returns the standard deviation of a column
            double sd(const size_t c) const {
                assert(c < colNum());
                if constexpr (std::is_same<T, elem>::value) assert(dt(0, c).t != evType::STR);
                else assert(UTIL_BASE::isNumerical<T>::val);
                double sum = 0, mu = mean(c);
                size_t nrow = rowNum();
                for (size_t i = 0; i < nrow; ++i) sum += (dt(i, c) - mu) * (dt(i, c) - mu);
                return std::sqrt(sum / nrow);
            }

            // Returns the minimum value of a column
            T minVal(const size_t c) const {
                assert(c < colNum());
                if constexpr (std::is_same<T, elem>::value) assert(dt(0, c).t != evType::STR);
                else assert(UTIL_BASE::isNumerical<T>::val);
                T mine = dt(0, c);
                for (size_t i = 1, nrow = rowNum(); i < nrow; ++i) {
                    if (mine > dt(i, c)) mine = dt(i, c);
                }
                return mine;
            }

            // Returns the maximum value of a column
            T maxVal(const size_t c) const {
                assert(c < colNum());
                if constexpr (std::is_same<T, elem>::value) assert(dt(0, c).t != evType::STR);
                else assert(UTIL_BASE::isNumerical<T>::val);
                T maxe = dt(0, c);
                for (size_t i = 1, nrow = rowNum(); i < nrow; ++i) {
                    if (maxe < dt(i, c)) maxe = dt(i, c);
                }
                return maxe;
            }

            void insert(const size_t r, const size_t c, T&& val) {
                dt.insert(r, c, std::forward<T>(val));
            }

            void addRow(T* r) {
                dt.addRow(r);
            }

            void addFeature(T* feat, const char* colname = nullptr) {
                if (colname) {
                    resize_ind2name(dt.col + 1);
                    name2ind[ind2name[dt.col + 1] = strdup(colname)] = dt.col + 1;
                }
                dt.addCol(feat);
            }

            void dropFeature(size_t feat) {
                size_t n = colNum() - 1;
                name2ind.erase(ind2name[feat]);
                for (size_t i = feat; i < n; ++i) name2ind[ind2name[i] = strdup(ind2name[i + 1])] = i;
                resize_ind2name(dt.col - 1);
                dt.dropCol(feat);
            }

            void dropFeature(size_t* feat_set, size_t n) {
                for (size_t i = 0; i < n; ++i) name2ind.erase(ind2name[feat_set[i]]);
                init_ind2name(colNum() - n);
                dt.dropCol(feat_set, n);
            }

            // Row binding
            void rbind(DataFrame& df, bool reset_colname = 0, std::vector<std::string> new_colname = {}) {
                rbind(std::move(df), reset_colname, new_colname);
            }

            void rbind(DataFrame&& df, bool reset_colname = 0, std::vector<std::string> new_colname = {}) {
                size_t n = colNum();
                assert(n == df.colNum());
                if (!reset_colname) {
                    for (size_t i = 0; i < n; ++i) 
                        assert(!strcmp(df.ind2name[i], ind2name[i]));
                } else {
                    assert(new_colname.size() == n);
                    for (size_t i = 0; i < n; ++i) name2ind[ind2name[i] = strdup(new_colname[i].c_str())] = i;
                }
                dt.concat(df.dt, ROW);
            }

            // Column binding
            void cbind(DataFrame& df) {
                cbind(std::move(df));
            }

            void cbind(DataFrame&& df) {
                assert(rowNum() == df.rowNum());
                size_t i = colNum(), n = i + df.colNum();
                resize_ind2name(n);
                if (df.ind2name) {
                    for (size_t cnt = 0; i < n; ++cnt, ++i) name2ind[ind2name[i] = df.ind2name[cnt]] = i;
                    df.ind2name = nullptr;
                }
                dt.concat(df.dt, COL);
            }

            T sum(const size_t rStart, const size_t rEnd, const size_t cStart, const size_t cEnd) {
                return dt.sum(rStart, rEnd, cStart, cEnd);
            }

            void colNameMapping(const char* name, size_t ind) {
                name2ind[strdup(name)] = ind;
            }

            const size_t getColByName(const char* name) {
                return name2ind[name];
            }

            const char* getColNameByInd(const size_t ind) {
                return ind2name[ind];
            }

            const size_t rowNum() const {
                return dt.rowNum();
            }

            const size_t colNum() const {
                return dt.colNum();
            }

            void dim() const {
                dt.dim();
            }

            std::unordered_set<T> unique(const size_t ind = 0) const {
                return dt.unique(ind);
            }

            Matrix<T> values() const {
                return dt;
            }

            template<typename R>
            friend std::ostream& operator<< (std::ostream& os, const DataFrame<R>& Df);
    };

    template<typename R>
    inline std::ostream& operator<< (std::ostream& os, const DataFrame<R>& Df) {
        if (Df.ind2name) {
            for (size_t i = 0, ncol = Df.colNum(); i < ncol; ++i) {
                if (Df.ind2name[i]) os << Df.ind2name[i] << " ";
                else os << '\t ';
            }
            puts("");
        }
        os << Df.dt;
        return os;
    }
}
