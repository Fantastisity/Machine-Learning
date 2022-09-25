#ifndef MATRIXUTIL_INCLUDED
#define MATRIXUTIL_INCLUDED
#include "../utils/matrixUtil.h"
#endif

namespace MACHINE_LEARNING {
    /* 
        extracts built-in type from the template parameter, so that type conversion can be done implicitly;
        for instance, it allows to pass an int as argument given T as double
    */
    template<typename R>
    using Deduce = typename std::common_type<R>::type;
    template<typename T>
    class Matrix {
        template<typename R>
        friend class DataFrame;
        
        template<typename>
        friend class Matrix;

        T* mat = nullptr;
        size_t row = 0, col = 0, cap = 0;

            template<typename F>
            Matrix& apply(F f) { // Apply a function to each matrix element
                for (size_t i = 0; i < row; ++i)
                    for (size_t j = 0; j < col; ++j)
                        f(i, j);
                return *this;
            }

            void resize(const size_t size) {
                cap = size;
                mat = (T*) realloc(mat, sizeof(T) * cap);
                if (!mat) {
                    std::cerr << "error realloc\n"; exit(1);
                }
            }

            void dealloc() {
                if (mat) {
                    free(mat);
                    mat = nullptr;
                }
            }

            /* Copy util functions */
            void deepCopy(const std::vector<std::vector<T>>&& Mat) {
                mat = (T*) malloc(sizeof(T) * cap);
                if (!mat) {
                    std::cerr << "error malloc\n"; exit(1);
                }
                for (size_t i = 0; i < row; ++i) 
                    std::copy(Mat[i].data(), Mat[i].data() + col, mat + i * col);
            }

            void deepCopy(const std::vector<T>&& Mat) {
                mat = (T*) malloc(sizeof(T) * cap);
                if (!mat) {
                    std::cerr << "error malloc\n"; exit(1);
                }
                std::copy(Mat.data(), Mat.data() + Mat.size(), mat);
            }

            void deepCopy(std::initializer_list<T>&& matIter) {
                mat = (T*) malloc(sizeof(T) * cap);
                if (!mat) {
                    std::cerr << "error malloc\n"; exit(1);
                }
                std::copy(matIter.begin(), matIter.end(), mat);
            }

            void deepCopy(typename std::initializer_list<std::initializer_list<T>>::iterator matIter) {
                mat = (T*) malloc(sizeof(T) * cap);
                if (!mat) {
                    std::cerr << "error malloc\n"; exit(1);
                }
                for (size_t i = 0; i < row; ++i, ++matIter) 
                    std::copy(matIter->begin(), matIter->end(), mat + i * col);
            }

            void deepCopy(T* Mat) {
                mat = (T*) malloc(sizeof(T) * cap);
                if (!mat) {
                    std::cerr << "error malloc\n"; exit(1);
                }
                std::copy(Mat, Mat + row * col, mat);
            }
        public:
            explicit Matrix(size_t cap = 200) : cap(cap) {
                if (cap) {
                    mat = (T*) calloc(cap, sizeof(T));
                    if (!mat) {
                        std::cerr << "error calloc\n"; exit(1);
                    }
                }
            }

            Matrix(size_t r, size_t c) : row(r), col(c), cap(r * c) {
                if (cap) {
                    mat = (T*) calloc(cap, sizeof(T));
                    if (!mat) {
                        std::cerr << "error calloc\n"; exit(1);
                    }
                }
            }

            Matrix(T* Mat, const size_t r, const size_t c) {
                assert(Mat);
                row = r, col = c, cap = r * c;
                if (cap) deepCopy(Mat);
            }

            Matrix(const std::vector<std::vector<T>>&& Mat) : row(Mat.size()), col(Mat[0].size()), cap(row * col) {
                if (cap) deepCopy(std::forward<const std::vector<std::vector<T>>>(Mat));
            }

            Matrix(const std::vector<T>&& Mat, const size_t nrow = 0, const size_t ncol = 0) : row(nrow), col(ncol), cap(row * col) {
                if (cap) deepCopy(std::forward<const std::vector<T>>(Mat));
            }

            Matrix(const std::initializer_list<std::initializer_list<T>>&& Mat) : row(Mat.size()), col((*Mat.begin()).size()), cap(row * col) {
                if (cap) deepCopy(Mat.begin());
            }

            Matrix(const std::initializer_list<T>&& Mat, const size_t nrow = 0, const size_t ncol = 0) : row(nrow), col(ncol), cap(row * col) {
                if (cap) deepCopy(std::forward<const std::initializer_list<T>>(Mat));
            }

            Matrix(const Matrix& m) {
                assert(m.mat);
                row = m.row, col = m.col, cap = m.cap;
                deepCopy(m.mat);
            }

            Matrix(Matrix&& m) {
                assert(m.mat);
                row = m.row, col = m.col, cap = m.cap;
                mat = m.mat, m.mat = nullptr;
            }

            Matrix<T>& operator= (const Matrix& m) {
                assert(m.mat);
                if (mat == m.mat) return *this;
                dealloc();
                row = m.row, col = m.col, cap = row * col;
                deepCopy(m.mat);
                return *this;
            }
            
            Matrix<T>& operator= (Matrix&& m) {
                assert(m.mat);
                if (mat == m.mat) return *this;
                dealloc();
                row = m.row, col = m.col, cap = row * col;
                mat = m.mat, m.mat = nullptr;
                return *this;
            }

            bool operator== (const Matrix& m) const {
                for (size_t i = 0; i < row; ++i)
                    for (size_t j = 0; j < col; ++j)
                        if (mat[i * col + j] != m.mat[i * col + j]) return 0;
                return 1;
            }

            bool operator!= (const Matrix& m) const {
                return !(*this == m);
            }

            const size_t rowNum() const {
                return row;
            }

            const size_t colNum() const {
                return col;
            }

            void dim() const { // Prints matrix dimensions
                std::cout << "[" << row << ", " << col << "]\n";
            }

            template<typename R>
            void insert(const size_t r, const size_t c, R&& val) {
                if (row <= r) row = r + 1;
                if (col <= c) col = c + 1;
                if (row * col >= cap) resize((row * col) << 1);
                mat[r * col + c] = std::forward<R>(val);
            }

            void addRow(T* r) {
                size_t new_row = row + 1;
                for (size_t i = 0; i < col; ++i) insert(new_row, i, r[i]);
            }

            void addCol(T* c) {
                T* m = (T*) malloc(sizeof(T) * (cap = row * ++col));
                for (size_t i = 0; i < row; ++i) {
                    std::copy(mat + i * (col - 1), mat + i * (col - 1) + col - 1, m + i * col);
                    m[i * col + col - 1] = c[i];
                }
                free(mat);
                mat = m;
                m = nullptr;
            }

            void dropCol(size_t c) { // Single column
                assert(c < col);
                size_t* ind = (size_t*) malloc(sizeof(size_t) * (col - 1));
                if (!ind) {
                    std::cerr << "error malloc\n"; exit(1);
                }
                for (size_t i = 0, p = 0; i < col; ++i) {
                    if (i == c) continue;
                    ind[p++] = i;
                }
                *this = (*this)(rngSlicer(row), ptrSlicer(ind, col - 1));
            }

            void dropCol(size_t* c, size_t n) { // Multiple columns
                assert(n <= col);
                size_t* ind = (size_t*) malloc(sizeof(size_t) * (col - n));
                if (!ind) {
                    std::cerr << "error malloc\n"; exit(1);
                }
                for (size_t i = 0, p = 0, j; i < col; ++i) {
                    j = 0;
                    for (; j < n; ++j) if (c[j] == i) break;
                    if (j == n) ind[p++] = i;
                }
                *this = (*this)(rngSlicer(row), ptrSlicer(ind, col - n));
            }

            void concat(const Matrix& Mat, bool is_row) {
                if (is_row) {
                    assert(col == Mat.col);
                    if ((row + Mat.row) * col >= cap) resize((row + Mat.row) * col);
                    std::copy(Mat.mat, Mat.mat + Mat.row * Mat.col, mat + row * col);
                    row += Mat.row;
                } else {
                    assert(row == Mat.row);
                    T* m = (T*) malloc(sizeof(T) * row * (col + Mat.col));
                    for (size_t i = 0; i < row; ++i) {
                        std::copy(mat + i * col, mat + i * col + col, m + i * (col + Mat.col));
                        std::copy(Mat.mat + i * Mat.col, Mat.mat + i * Mat.col + Mat.col, m + i * (col + Mat.col) + col);
                    }
                    cap = (col += Mat.col) * row;
                    free(mat);
                    mat = m;
                    m = nullptr;
                }
            }

            template<typename R>
            Matrix<R> asType() const { // Type conversion (T -> R)
                if constexpr (std::is_same<T, R>::value) return *this;
                Matrix<R> m(row, col);
                UTIL_BASE::MATRIX_UTIL::conv_type(m.mat, mat, row * col);
                return m;
            }

            static Matrix eye(size_t n) { // Identity matrix
                Matrix m(n, n);
                for (int i = 0; i < n; ++i) m(i, i) = 1;
                return m;
            }

            Matrix inverse() {
                assert(row == col);
                Matrix idm = eye(row);
                assert(UTIL_BASE::MATRIX_UTIL::gauss_jordan_elimination(Matrix(*this).mat, idm.mat, row, col, col, 1));
                return idm;
            }

            Matrix trans() const { // Leveraged caching
                Matrix m(col, row);
                for (size_t i = 0; i < row; i += 6) 
                    for (size_t j = 0; j < col; j += 4) 
                        for (size_t k = i, end = i + 6 < row ? i + 6 : row; k < end; ++k) {
                            T c1, c2, c3, c4;
                            if (j + 3 < col) {
                                c1 = mat[k * col + j], c2 = mat[k * col + j + 1], c3 = mat[k * col + j + 2], c4 = mat[k * col + j + 3];
                                m(j, k) = c1, m(j + 1, k) = c2, m(j + 2, k) = c3, m(j + 3, k) = c4;
                            } else if (j + 2 < col) {
                                c1 = mat[k * col + j], c2 = mat[k * col + j + 1], c3 = mat[k * col + j + 2];
                                m(j, k) = c1, m(j + 1, k) = c2, m(j + 2, k) = c3;
                            } else if (j + 1 < col) {
                                c1 = mat[k * col + j], c2 = mat[k * col + j + 1];
                                m(j, k) = c1, m(j + 1, k) = c2;
                            } else c1 = mat[k * col + j], m(j, k) = c1;
                        }
                return m;
            }

            Matrix multiply(Matrix& rht) const {
                return multiply(std::move(rht));
            }

            Matrix multiply(Matrix&& rht) const {
                assert(row == rht.row && col == rht.col);
                Matrix m(row, col);
                auto batch_mult = [&](size_t from, size_t to) {
                    for (size_t i = from; i < to; ++i) m.mat[i] = mat[i] * rht.mat[i];
                };
                const size_t pcnt = std::thread::hardware_concurrency(); // Number of cores
                std::thread threads[pcnt - 1]; // Ignore the main thread
                size_t batch = cap / pcnt; // Determine the batch(no. of rows) size
                batch_mult(0, batch); // Main thread
                for (size_t i = 0, cnt = batch; i < pcnt - 1; ++i, cnt += batch) { // Remaining threads
                    threads[i] = std::thread(batch_mult, cnt, cnt + batch < cap ? cnt + batch : cap);
                    threads[i].join();
                }
                return m;
            }

            void shuffle(size_t random_state) { // Randomly shuffle the matrix
                srand(random_state);
                if (row > 1) 
                    for (size_t i = 0, j; i < row - 1; ++i) {
                        j = i + rand() / (RAND_MAX / (row - i) + 1);
                        std::swap_ranges(mat + i * col, mat + i * col + col, mat + j * col); // Swap two rows
                    }
            }

            /*
                Return a random sample of rows
                @param
                    n: number of rows to return
                    frac: fraction of rows to return, cannot be used with n  
            */
            Matrix sample(size_t n = 1, float frac = 0) {
                assert((n > 0 && n <= row && !frac) || (!n && frac > 0));
                if (frac) n = row * frac;
                std::random_device rd;
                std::mt19937 gen(rd());
                std::unordered_set<size_t> res; // Stores unique row indices
                for (size_t i = row - n; i < row; ++i) {
                    size_t v = std::uniform_int_distribution<unsigned long long>(0, i)(gen);
                    if (!res.insert(v).second) res.insert(i);
                }
                return (*this)(ptrSlicer(std::vector<size_t>(res.begin(), res.end()).data(), n), rngSlicer(col));
            }

            std::unordered_set<T> unique(const size_t col_ind = 0) const { // Unique elements in a specific column
                std::unordered_set<T> unique_val;
                for (size_t i = 0; i < row; ++i) unique_val.insert(mat[i * col + col_ind]);
                return unique_val;
            }

            // Get sum of elements within row and column ranges
            T sum(const size_t rStart, const size_t rEnd, const size_t cStart, const size_t cEnd) {
                assert(rEnd <= row && cEnd <= col);
                T res = 0;
                for (size_t r = rStart; r < rEnd; ++r)
                    for (size_t c = cStart; c < cEnd; ++c) res += mat[r * col + c];
                return res;
            }

            T* value() {
                return mat;
            }

            ~Matrix() {
                dealloc();
            }

            T& operator()(const size_t r, const size_t c) const {
                return mat[r * col + c];
            }

            // Returns a sub matrix
            template<typename Q, typename U>
            auto operator()(UTIL_BASE::MATRIX_UTIL::slice<Q>&& row_slice, UTIL_BASE::MATRIX_UTIL::slice<U>&& col_slice) {
                Matrix<T> subMat(row_slice.size, col_slice.size);
                size_t r = 0, c, r_ind, c_ind;
                for (auto i = row_slice.start; i < row_slice.end; ++i, ++r) {
                    c = 0;
                    for (auto j = col_slice.start; j < col_slice.end; ++j, ++c) {
                        if constexpr (UTIL_BASE::MATRIX_UTIL::isPTR<Q>::val) r_ind = *i; // Pointer slicer
                        else r_ind = i; // Range slicer

                        if constexpr (UTIL_BASE::MATRIX_UTIL::isPTR<U>::val) c_ind = *j;
                        else c_ind = j;

                        subMat.insert(r, c, mat[r_ind * col + c_ind]);
                    }
                }
                return subMat;
            }

            template<typename R>
            friend std::ostream& operator<< (std::ostream& os, const Matrix<R>& m);

            template<typename R>
            auto operator* (Matrix<R>& rht) const {
                // Deduce to scalar multiplication if either matrix has only a single entry
                if (rht.cap == 1) return *this * rht(0, 0);
                else if (cap == 1) return rht * mat[0];
                assert(col == rht.row);
                using tp = decltype(std::declval<T>() * std::declval<R>());
                Matrix<tp> m(row, rht.col);
                UTIL_BASE::MATRIX_UTIL::Mult<T, R, tp>::mult(mat, rht.mat, m.mat, row, col, rht.col);
                return m;
            }

            template<typename R>
            auto operator* (Matrix<R>&& rht) const {
                if (rht.cap == 1) return *this * rht(0, 0);
                else if (cap == 1) return rht * mat[0];
                assert(col == rht.row);
                using tp = decltype(std::declval<T>() * std::declval<R>());
                Matrix<tp> m(row, rht.col);
                UTIL_BASE::MATRIX_UTIL::Mult<T, R, tp>::mult(mat, rht.mat, m.mat, row, col, rht.col);
                return m;
            }

            Matrix operator* (const T val) const {
                Matrix m = *this;
                m *= std::move(val);
                return m;
            }

            template<typename R>
            friend Matrix<R> operator* (const Deduce<R> val, Matrix<R>& Mat);

            template<typename R>
            friend Matrix<R> operator* (const Deduce<R> val, Matrix<R>&& Mat);

            Matrix& operator*= (const T& val) {
                return apply([&](const size_t i, const size_t j){mat[i * col + j] *= val;});
            }

            Matrix& operator*= (const T&& val) {
                return apply([&](const size_t i, const size_t j){mat[i * col + j] *= val;});
            }

            template<typename R>
            Matrix operator+ (R&& rht) const {
                Matrix m = *this;
                m += std::forward<R>(rht);
                return m;
            }

            template<typename R>
            friend Matrix<R> operator+ (const Deduce<R> val, Matrix<R>& Mat);

            template<typename R>
            friend Matrix<R> operator+ (const Deduce<R> val, Matrix<R>&& Mat);

            template<typename R>
            Matrix& operator+= (const Matrix<R>& rht) {
                assert(row == rht.row && col == rht.col);
                return apply([&](const size_t i, const size_t j){mat[i * col + j] += rht(i, j);});
            }

            template<typename R>
            Matrix& operator+= (const Matrix<R>&& rht) {
                assert(row == rht.row && col == rht.col);
                return apply([&](const size_t i, const size_t j){mat[i * col + j] += rht(i, j);});
            }

            Matrix& operator+= (const T& val) {
                return apply([&](const size_t i, const size_t j){mat[i * col + j] += val;});
            }

            Matrix& operator+= (const T&& val) {
                return apply([&](const size_t i, const size_t j){mat[i * col + j] += val;});
            }

            Matrix& operator-() {
                for (size_t i = 0; i < row * col; ++i) mat[i] = -mat[i];
                return *this;
            }

            template<typename R>
            Matrix operator- (R&& rht) const {
                Matrix m = *this;
                m -= std::forward<R>(rht);
                return m;
            }

            template<typename R>
            friend Matrix<R> operator- (const Deduce<R> val, Matrix<R>& Mat);

            template<typename R>
            friend Matrix<R> operator- (const Deduce<R> val, Matrix<R>&& Mat);

            template<typename R>
            Matrix& operator-= (const Matrix<R>& rht) {
                assert(row == rht.row && col == rht.col);
                return apply([&](const size_t i, const size_t j){mat[i * col + j] -= rht(i, j);});
            }

            template<typename R>
            Matrix& operator-= (const Matrix<R>&& rht) {
                assert(row == rht.row && col == rht.col);
                return apply([&](const size_t i, const size_t j){mat[i * col + j] -= rht(i, j);});
            }

            Matrix& operator-= (const T& val) {
                return apply([&](const size_t i, const size_t j){mat[i * col + j] -= val;});
            }

            Matrix& operator-= (const T&& val) {
                return apply([&](const size_t i, const size_t j){mat[i * col + j] -= val;});
            }

            Matrix operator/ (T val) const {
                Matrix m = *this;
                m /= std::move(val);
                return m;
            }

            template<typename R>
            friend Matrix<R> operator/ (const Deduce<R> val, Matrix<R>& Mat);

            template<typename R>
            friend Matrix<R> operator/ (const Deduce<R> val, Matrix<R>&& Mat);

            Matrix& operator/= (const T&& val) {
                return apply([&](const size_t i, const size_t j){mat[i * col + j] /= val;});
            }
    };

    template<typename R>
    inline std::ostream& operator<< (std::ostream& os, const Matrix<R>& m) {
        for (size_t i = 0; i < m.row; ++i) {
            for (size_t j = 0; j < m.col; ++j) 
                os << m(i, j) << " ";
            puts("");
        }
        return os;
    }

    template<typename R>
    inline Matrix<R> operator+ (const Deduce<R> val, Matrix<R>& Mat) {
        return Matrix(Mat) += val;
    }

    template<typename R>
    inline Matrix<R> operator+ (const Deduce<R> val, Matrix<R>&& Mat) {
        return Matrix(std::move(Mat)) += val;
    }

    template<typename R>
    inline Matrix<R> operator- (const Deduce<R> val, Matrix<R>& Mat) {
        Matrix<R> tmp(Mat);
        tmp = -tmp;
        return tmp += val;
    }

    template<typename R>
    inline Matrix<R> operator- (const Deduce<R> val, Matrix<R>&& Mat) {
        Matrix<R> tmp(std::move(Mat));
        tmp = -tmp;
        return tmp += val;
    }

    template<typename R>
    inline Matrix<R> operator* (const Deduce<R> val, Matrix<R>& Mat) {
        return Matrix(Mat) *= val;
    }

    template<typename R>
    inline Matrix<R> operator* (const Deduce<R> val, Matrix<R>&& Mat) {
        return Matrix(std::move(Mat)) *= val;
    }

    template<typename R>
    inline Matrix<R> operator/ (const Deduce<R> val, Matrix<R>& Mat) {
        return Matrix(Mat) /= val;
    }

    template<typename R>
    inline Matrix<R> operator/ (const Deduce<R> val, Matrix<R>&& Mat) {
        return Matrix(std::move(Mat)) *= 1 / val;
    }
}