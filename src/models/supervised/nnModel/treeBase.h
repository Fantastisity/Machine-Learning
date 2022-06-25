namespace MACHINE_LEARNING {
    namespace NNTREE {
        template<typename T>
        struct TreeBase {
                std::priority_queue<std::pair<double, size_t>> query(T* point, size_t n_neighbors) {
                    assert(point && n_neighbors);
                    if (!root) {
                        size_t nrow = mat.rowNum(), ind[nrow];
                        for (size_t i = 0; i < nrow; ++i) ind[i] = i;
                        root = partition(ind, nrow);
                    }
                    std::priority_queue<std::pair<double, size_t>> q;
                    query(root, point, n_neighbors, q);
                    return q;
                }
                virtual ~TreeBase() {
                    if (root) {
                        delete root;
                        root = nullptr;
                    }
                }
            protected:
                Matrix<T> mat;
                size_t leaf_size, p;
                Metric m;
                TreeBase(const Matrix<T>& mat, const size_t leaf_size, const Metric m = Metric::EUCLIDEAN, const size_t p = 2)
                : mat(mat), leaf_size(leaf_size), m(m), p(p) {}

                TreeBase(const Matrix<T>&& mat, const size_t leaf_size, const Metric m = Metric::EUCLIDEAN, const size_t p = 2)
                : mat(std::move(mat)), leaf_size(leaf_size), m(m), p(p) {}
                struct node {
                    node* left_child, * right_child;
                    T* split_node, * indices;
                    size_t size;
                    double radius;
                    bool is_leaf;
                    ~node() {
                        if (left_child) {
                            delete left_child;
                            left_child = nullptr;
                        }
                        if (right_child) {
                            delete right_child; 
                            right_child = nullptr;
                        }
                        if (split_node) {
                            delete[] split_node;
                            split_node = nullptr;
                        }
                        if (indices) {
                            delete[] indices;
                            indices = nullptr;
                        }
                    }
                };
                node* init_node(size_t nrow, size_t ncol) {
                    node* nd = new node;
                    nd->left_child = nd->right_child = nullptr;
                    nd->split_node = new T[ncol];
                    nd->indices = new T[nd->size = nrow];
                    nd->is_leaf = 0, nd->radius = 0;
                    return nd;
                }
                node* root = nullptr;
                virtual node* partition(size_t ind[], size_t nrow) = 0;
                virtual void query(node* root, const T const * point, const size_t n_neighbors, 
                        std::priority_queue<std::pair<double, size_t>>& point_set, double dist_min = -1) = 0;
        };
    }
}