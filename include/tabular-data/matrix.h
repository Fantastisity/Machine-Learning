#include "../utils/matrixUtil.h"

namespace MACHINE_LEARNING {
    /* 
        used for extracting the built-in type from template parameter, so that type conversion can be done implicitly
        for instance, it allows to pass an int as argument given T is double 
    */
    // template<typename R>
    // using Deduce = typename std::common_type<R>::type;
    template<typename T>
    class Matrix {
        template<typename R>
        friend class DataFrame;
        template <typename>
        friend class Matrix;
        
        T* mat = nullptr;
        size_t row = 0, col = 0, cap;
            template<typename F>
            Matrix& apply(F f) {
                for (size_t i = 0; i < row; ++i)
                    for (size_t j = 0; j < col; ++j)
                        f(i, j);
                return *this;
            }

            template<typename R>
            void deepCopy(const std::vector<std::vector<R>>&& Mat) {
                mat = (T*) malloc(sizeof(T) * cap);
                if (!mat) {
                    std::cerr << "error malloc\n"; exit(1);
                }
                for (size_t i = 0; i < row; ++i) 
                    std::copy(Mat[i].data(), Mat[i].data() + col, mat + i * col);
            }

            template<typename R>
            void deepCopy(typename std::initializer_list<std::initializer_list<R>>::iterator matIter) {
                mat = (T*) malloc(sizeof(T) * cap);
                if (!mat) {
                    std::cerr << "error malloc\n"; exit(1);
                }
                for (size_t i = 0; i < row; ++i, ++matIter) 
                    std::copy(matIter->begin(), matIter->end(), mat + i * col);
            }

            template<typename R>
            void deepCopy(R* Mat) {
                mat = (T*) malloc(sizeof(T) * cap);
                if (!mat) {
                    std::cerr << "error malloc\n"; exit(1);
                }
                std::copy(Mat, Mat + row * col, mat);
            }

            void resize(const size_t size) {
                cap = size;
                mat = (T*)realloc(mat, sizeof(T) * cap);
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
        public:
            explicit Matrix(size_t cap = 1000) : cap(cap) {
                if (cap) {
                    mat = (T*) calloc(cap, sizeof(T));
                    if (!mat) {
                        std::cerr << "error malloc\n"; exit(1);
                    }
                }
            }

            template<typename R>
            Matrix(R** Mat, const size_t r, const size_t c) {
                row = r, col = c, cap = r * c;
                deepCopy(Mat);
            }

            template<typename R>
            Matrix(const std::vector<std::vector<R>>&& Mat) {
                row = Mat.size(), col = Mat[0].size(), cap = row * col;
                deepCopy(std::forward<const std::vector<std::vector<R>>>(Mat));
            }

            template<typename R>
            Matrix(const std::initializer_list<std::initializer_list<R>>&& Mat) {
                row = Mat.size(), col = (*Mat.begin()).size(), cap = row * col;
                deepCopy(Mat.begin());
            }

            template<typename R>
            Matrix(const Matrix<R>& m) {
                row = m.row, col = m.col, cap = row * col;
                deepCopy(m.mat);
            }

            template<typename R>
            Matrix(Matrix<R>&& m) {
                row = m.row, col = m.col, cap = row * col;
                deepCopy(m.mat);
                if (m.mat) free(m.mat), m.mat = nullptr;
            }

            template<typename R>
            auto& operator= (const Matrix<R>& m) {
                dealloc();
                row = m.row, col = m.col, cap = row * col;
                deepCopy(m.mat);
                return *this;
            }
            
            template<typename R>
            auto& operator= (Matrix<R>&& m) {
                dealloc();
                row = m.row, col = m.col, cap = row * col;
                deepCopy(m.mat);
                if (m.mat) free(m.mat), m.mat = nullptr;
                return *this;
            }

            template<typename R>
            bool operator== (const Matrix<R>& m) const {
                for (size_t i = 0; i < row; ++i)
                    for (size_t j = 0; j < col; ++j)
                        if (mat[i * col + j] != m.mat[i * col + j]) return 0;
                return 1;
            }

            template<typename R>
            bool operator!= (const Matrix<R>& m) const {
                return !(*this == m);
            }

            const size_t rowNum() const {
                return row;
            }

            const size_t colNum() const {
                return col;
            }

            void dim() const {
                std::cout << row << " * " << col << '\n';
            }

            template<typename R>
            void insert(const size_t r, const size_t c, R&& val) {
                if (row <= r) row = r + 1;
                if (col <= c) col = c + 1;
                if (row * col >= cap) resize((row * col) << 1);
                mat[r * col + c] = std::forward<R>(val);
            }

            template<typename R>
            void addCol(R* c) {
                if (row * ++col >= cap) resize(row * col);
                T* m = (T*) malloc(sizeof(T) * row * col);
                for (size_t i = 0, cnt = 0; i < row; ++i) {
                    std::copy(mat + i * (col - 1), mat + i * (col - 1) + col - 1, m + i * col);
                    m[i * col + col - 1] = c[cnt++];
                }
                free(mat);
                mat = m;
                m = nullptr;
            }

            void dropCol(size_t c) {
                assert(c < col);
                size_t* ind = (size_t*) malloc(sizeof(size_t) * (col - 1));
                if (!ind) {
                    std::cerr << "error malloc\n"; exit(1);
                }
                for (size_t i = 0, p = 0; i < col; ++i) {
                    if (i == c) continue;
                    ind[p++] = i;
                }
                *this = (*this)(rangeSlicer(row), ptrSlicer(ind, col - 1));
            }

            void dropCol(size_t* c, size_t n) {
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
                *this = (*this)(rangeSlicer(row), ptrSlicer(ind, col - n));
            }

            template<typename R>
            void concat(const Matrix<R>& Mat, bool is_row) {
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
            Matrix<R> asType() {
                Matrix<R> m(row * col);
                for (size_t i = 0; i < row; ++i)
                    for (size_t j = 0; j < col; ++j) {
                        m(i, j) = static_cast<R>(mat[i * col + j]);
                    }
                return m;
            }

            Matrix gen_identity(size_t n) {
                Matrix m(n * n);
                for (int i = 0; i < n; ++i) m(i, i) = 1;
                return m;
            }

            Matrix inverse() {
                assert(row == col);
                Matrix<T> m = *this, idm = gen_identity(row);
                return NULL;
            }

            Matrix trans() {
                Matrix m(col * row);
                for (size_t i = 0; i < row; ++i)
                    for (size_t j = 0; j < col; ++j) m.insert(j, i, mat[i * col + j]);
                return m;
            }

            void shuffle() {
                if (row > 1) 
                    for (size_t i = 0, j; i < row - 1; ++i) {
                        j = i + rand() / (RAND_MAX / (row - i) + 1);
                        std::swap_ranges(mat + i * col, mat + i * col + col, mat + j * col);
                    }
            }

            ~Matrix() {
                dealloc();
            }

            T& operator()(const size_t r, const size_t c) const {
                return mat[r * col + c];
            }

            template<typename Q, typename U>
            auto operator()(slice<Q>&& s1, slice<U>&& s2) {
                Matrix<T> subMat(s1.size * s2.size);
                size_t r = 0, c, tmp_r, tmp_c;
                for (auto i = s1.start; i < s1.end; ++i, ++r) {
                    c = 0;
                    for (auto j = s2.start; j < s2.end; ++j, ++c) {
                        if constexpr (MatrixUtil::isPTR<Q>::val) tmp_r = *i;
                        else tmp_r = i;

                        if constexpr (MatrixUtil::isPTR<U>::val) tmp_c = *j;
                        else tmp_c = j;

                        subMat.insert(r, c, mat[tmp_r * col + tmp_c]);
                    }
                }
                return subMat;
            }

            template<typename R>
            friend std::ostream& operator<< (std::ostream& os, const Matrix<R>& m);

            template<typename R>
            auto operator* (Matrix<R>& rht) const {
                assert(col == rht.row);
                using tp = decltype(std::declval<T>() * std::declval<R>());
                Matrix<tp> m(row * rht.col);
                m.row = row, m.col = rht.col;
                MatrixUtil::Mult<T, R, tp>::mult(mat, rht.mat, m.mat, row, col, rht.col);
                return m;
            }

            template<typename R>
            auto operator* (Matrix<R>&& rht) const {
                assert(col == rht.row);
                using tp = decltype(std::declval<T>() * std::declval<R>());
                Matrix<tp> m(row * rht.col);
                m.row = row, m.col = rht.col;
                MatrixUtil::Mult<T, R, tp>::mult(mat, rht.mat, m.mat, row, col, rht.col);
                return m;
            }

            Matrix operator* (T val) {
                Matrix m = *this;
                m *= std::move(val);
                return m;
            }

            template<typename R>
            auto operator*= (Matrix<R>&& rht) const {
                *this = *this * std::move(rht);
                return *this;
            }

            template<typename R>
            auto operator*= (Matrix<R>& rht) const {
                *this = *this * rht;
                return *this;
            }

            Matrix& operator*= (const T&& val) {
                return apply([&](const size_t i, const size_t j){mat[i * col + j] *= val;});
            }

            template<typename R>
            Matrix operator+ (R&& rht) const {
                Matrix m = *this;
                m += std::forward<T>(rht);
                return m;
            }

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

            template<typename R>
            Matrix operator- (R&& rht) const {
                Matrix m = *this;
                m -= std::forward<R>(rht);
                return m;
            }

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

            Matrix& operator/= (const T&& val) {
                return apply([&](const size_t i, const size_t j){mat[i * col + j] /= val;});
            }
    };

    template<typename R>
    std::ostream& operator<< (std::ostream& os, const Matrix<R>& m) {
        for (size_t i = 0; i < m.row; ++i) {
            for (size_t j = 0; j < m.col; ++j) 
                os << m(i, j) << " ";
            puts("");
        }
        return os;
    }
}