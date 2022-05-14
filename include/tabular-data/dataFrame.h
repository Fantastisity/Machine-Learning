#ifndef MATRIX_INCLUDED
#define MATRIX_INCLUDED
#include "matrix.h"
#endif

namespace MACHINE_LEARNING {
    template<typename T = double>
    class DataFrame {
        struct CstrFunctor {
            size_t operator()(const char* str) const {
                size_t hash = 5381;
                for (; *str; ++str) hash = ((hash << 5) + hash) + *str;
                return hash;
            }
            bool operator()(const char* a, const char* b) const {
                return !strcmp(a, b);
            }
        };
        Matrix<T> dt;
        size_t cateVar_ind = 0;
        std::unordered_map<const char*, size_t, CstrFunctor, CstrFunctor> name2ind, cateVar_mapping;
        char** ind2name = nullptr, ***cateVar = nullptr;

        void resize_ind2name(size_t size) {
            ind2name = (char**) realloc(ind2name, size);
            if (!ind2name) {
                std::cerr << "error realloc\n"; exit(1);
            }
        }

        void resize_cateVar(size_t size, bool is_row) {
            if (!is_row) {
                cateVar = (char***) realloc(cateVar, sizeof(char**) * size);
                if (!cateVar) {
                    std::cerr << "error realloc\n"; exit(1);
                }
                for (size_t n = dt.row; cateVar_ind < size; ++cateVar_ind) {
                    cateVar[cateVar_ind] = (char**) malloc(sizeof(char*) * n);
                    if (!cateVar[cateVar_ind]) {
                        std::cerr << "error malloc\n"; exit(1);
                    }
                }
            } else {
                for (size_t i = 0; i < cateVar_ind; ++i) {
                    cateVar[i] = (char**) realloc(cateVar[i], sizeof(char*) * size);
                    if (!cateVar[i]) {
                        std::cerr << "error realloc\n"; exit(1);
                    }
                }
            }
        }

        void dealloc() {
            dt.dealloc();
            if (ind2name) {
                for (auto& [k, v] : name2ind) free(const_cast<char*>(k));
                for (size_t i = 0, ncol = dt.col; i < ncol; ++i) free(ind2name[i]);
                free(ind2name);
                ind2name = nullptr;
            }
            if (cateVar) {
                for (auto& [k, v] : cateVar_mapping) free(const_cast<char*>(k));
                for (size_t i = 0; i < cateVar_ind; ++i) {
                    for (size_t j = 0, ncol = dt.col; j < ncol; ++j) free(cateVar[i][j]);
                    free(cateVar[j]);
                }
                free(cateVar); 
                cateVar = nullptr;
            }
        }
        public:
            DataFrame(size_t nrow = 50, size_t ncol = 50, const std::vector<std::string>&& colnames = {}) : dt(nrow, ncol) {
                if (!colnames.empty()) {
                    init_ind2name(ncol, 0);
                    for (size_t i = 0; i < ncol; ++i) {
                        const char* tmp = strdup(colnames[i].c_str();
                        name2ind[ind2name[i] = tmp] = i;
                    }
                }
            }
            DataFrame(T** Mat, const size_t nrow, const size_t ncol, const std::vector<std::string>&& colnames = {}) : dt(Mat, nrow, ncol) {
                if (!colnames.empty()) {
                    init_ind2name(ncol, 0);
                    for (size_t i = 0; i < ncol; ++i) {
                        const char* tmp = strdup(colnames[i].c_str();
                        name2ind[ind2name[i] = tmp] = i;
                    }
                }
            }
            DataFrame(const std::vector<std::vector<T>>&& Mat, const std::vector<std::string>&& colnames = {}) : dt(std::move(Mat)) {
                if (!colnames.empty()) {
                    size_t ncol = Mat[0].size();
                    init_ind2name(ncol, 0);
                    for (size_t i = 0; i < ncol; ++i) {
                        const char* tmp = strdup(colnames[i].c_str();
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
                        const char* tmp = strdup(colnames[i].c_str();
                        name2ind[ind2name[i] = tmp] = i;
                    }
                }
            }
            DataFrame(const Matrix<T>& m, const std::vector<std::string>&& colnames = {}) : dt(m) {
                if (!colnames.empty()) {
                    size_t ncol = dt.col;
                    init_ind2name(ncol, 0);
                    for (size_t i = 0; i < ncol; ++i) {
                        const char* tmp = strdup(colnames[i].c_str();
                        name2ind[ind2name[i] = tmp] = i;
                    }
                }
            }
            DataFrame(Matrix<T>&& m, const std::vector<std::string>&& colnames = {}) : dt(std::move(m)) {
                if (!colnames.empty()) {
                    size_t ncol = dt.col;
                    init_ind2name(ncol, 0);
                    for (size_t i = 0; i < ncol; ++i) {
                        const char* tmp = strdup(colnames[i].c_str();
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
                    name2ind = df.name2ind;
                    init_ind2name(dt.col, 0);
                    std::copy(df.ind2name, df.ind2name + df.col, )
                }
                if (!df.cateVar.empty()) {
                    cateVar = df.cateVar;
                    cateVar_mapping = df.cateVar_mapping;
                }
            }
            DataFrame(DataFrame&& df) : dt(std::move(df.dt)) {
                if (!df.ind2name.empty()) {
                    name2ind = df.name2ind;
                    ind2name = df.ind2name;
                }
                if (!df.cateVar.empty()) {
                    cateVar = df.cateVar;
                    cateVar_mapping = df.cateVar_mapping;
                }
            }

            auto operator()(const size_t r, const size_t c) const {
                return dt(r, c);
            }

            void shuffle() {
                dt.shuffle();
            }

            template<typename R, typename U>
            DataFrame iloc(slice<R>&& s1, slice<U>&& s2, bool assign_colname = 0) {
                if (s1.size == rowNum() && s2.size == colNum()) return *this;

                if (assign_colname && s2.size < colNum()) {
                    std::vector<std::string> colnames(s2.size);
                    size_t cnt = 0;
                    for (auto i = s2.start; i < s2.end; ++i) {
                        size_t ind;
                        if constexpr (MatrixUtil::isPTR<U>::val) ind = *i;
                        else ind = i;

                        colnames[cnt++] = ind2name[ind];
                    }
                    return DataFrame(dt(std::forward<slice<R>>(s1), std::forward<slice<U>>(s2)), std::move(colnames));
                }

                return DataFrame(dt(std::forward<slice<R>>(s1), std::forward<slice<U>>(s2)));
            }

            template<typename R>
            DataFrame loc(slice<R>&& s1, std::vector<std::string> colnames, bool assign_colname = 0) {
                size_t n = colnames.size();
                if (s1.size == rowNum() && n == colNum()) return *this;
                size_t inds[n];
                for (size_t i = 0; i < n; ++i) inds[i] = name2ind[colnames[i]];

                return assign_colname ? DataFrame(dt(std::forward<slice<R>>(s1), ptrSlicer(inds, n)), std::move(colnames)) :
                                        DataFrame(dt(std::forward<slice<R>>(s1), ptrSlicer(inds, n)));
            }

            auto& operator= (const DataFrame& df) {
                dealloc();
                dt = df.dt;
                if (!df.ind2name.empty()) {
                    name2ind = df.name2ind;
                    init_ind2name(df.colNum());
                }
                if (!df.cateVar.empty()) {
                    cateVar = df.cateVar;
                    cateVar_mapping = df.cateVar_mapping;
                }
                return *this;
            }
            
            auto& operator= (DataFrame&& df) {
                dealloc();
                dt = std::move(df.dt);
                if (!df.ind2name.empty()) {
                    name2ind = df.name2ind;
                    ind2name = df.ind2name;
                }
                if (!df.cateVar.empty()) {
                    cateVar = df.cateVar;
                    cateVar_mapping = df.cateVar_mapping;
                }
                return *this;
            }

            void insert(const size_t r, const size_t c, T&& val) {
                dt.insert(r, c, std::forward<T>(val));
            }

            void insert(const size_t c, std::string&& val) {
                if (!cateVar_mapping.count(c)) cateVar_mapping[c] = cateVar_ind++;
                cateVar[cateVar_mapping[c]].push_back(std::move(val));
            }

            void addFeature(T* feat, std::string&& colname = {}) {
                if (colname != "") {
                    ind2name.push_back(colname);
                    name2ind[ind2name.back()] = 1 + colNum();
                }
                dt.addCol(feat);
            }

            void dropFeature(size_t feat) {
                size_t n = colNum() - 1;
                name2ind.erase(ind2name[feat]);
                if (cateVar_mapping.count(feat)) cateVar_mapping.erase(feat);
                for (size_t i = feat; i < n; ++i) name2ind[ind2name[i] = ind2name[i + 1]] = i;
                ind2name.pop_back();
                dt.dropCol(feat);
            }

            void dropFeature(size_t* feat_set, size_t n) {
                for (size_t i = 0; i < n; ++i) {
                    name2ind.erase(ind2name[feat_set[i]]);
                    if (cateVar_mapping.count(feat_set[i])) cateVar_mapping.erase(feat_set[i]);
                }
                init_ind2name(colNum() - n);
                dt.dropCol(feat_set, n);
            }

            void rbind(DataFrame&& df, bool reset_colname = 0, std::vector<std::string> new_colname = {}) {
                size_t n = colNum();
                assert(n == df.colNum());
                if (!reset_colname) {
                    for (size_t i = 0; i < n; ++i) 
                        assert(df.ind2name[i] == ind2name[i]);
                } else {
                    assert(new_colname.size() == n);
                    for (size_t i = 0; i < n; ++i) name2ind[ind2name[i] = new_colname[i]] = i;
                }

                if (!cateVar.empty()) {
                    assert(!df.cateVar.empty());
                    for (size_t i = 0; i < cateVar_ind; ++i) std::copy(df.cateVal[i].begin(), df.cateVal[i].end(), std::back_inserter(cateVal[i]));
                }

                dt.concat(df.dt, ROW);
            }

            void cbind(DataFrame&& df) {
                assert(rowNum() == df.rowNum());
                size_t i = colNum(), n = i + df.colNum();
                ind2name.resize(n);
                for (size_t cnt = 0; i < n; ++cnt, ++i) name2ind[ind2name[i] = df.ind2name[cnt]] = i;
                if (!df.cateVar.empty()) {
                    for (size_t j = 0; j < df.cateVar_ind; ++j) cateVar.push_back(df.cateVar[j]);
                    
                }

                dt.concat(df.dt, COL);
            }

            void colNameMapping(const char* name, size_t ind) {
                if (!ind2name) {
                    ind2name = (char**) malloc(colNum() * sizeof(char*));
                    if (!ind2name) {
                        std::cerr << "error malloc\n"; exit(1);
                    }
                } else if (ind >= colNum()) resize_ind2name(ind << 1);
                const char* tmp = strdup(name);
                name2ind[ind2name[ind] = tmp] = ind;
            }

            void init_ind2name(size_t n, bool assign = 1) {
                ind2name = (char**) malloc(n * sizeof(char*));
                if (!ind2name) {
                    std::cerr << "error malloc\n"; exit(1);
                }
                if (assign) for (auto& [k, v] : name2ind) ind2name[v] = k;
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

            auto values() const {
                return dt;
            }

            ~DataFrame() {
                dealloc();
            }

            template<typename R>
            friend std::ostream& operator<< (std::ostream& os, const DataFrame<R>& Df);
    };

    template<typename R>
    std::ostream& operator<< (std::ostream& os, const DataFrame<R>& Df) {
        os << Df.dt;
        return os;
    }
}