#ifndef NNTREEBASE_INCLUDED
#define NNTREEBASE_INCLUDED
#include "treeBase.h"
#endif

namespace MACHINE_LEARNING {
    namespace NNTREE {
        template<typename T>
        struct BallTree : TreeBase<T> {
            BallTree(const Matrix<T>& mat, const size_t leaf_size, const Metric m = Metric::EUCLIDEAN, const size_t p = 2) 
            : TreeBase<T>(mat, leaf_size, m, p) {}
            BallTree(const Matrix<T>&& mat, const size_t leaf_size, const Metric m = Metric::EUCLIDEAN, const size_t p = 2) 
            : TreeBase<T>(std::move(mat), leaf_size, m, p) {}
            BallTree(const BallTree& tree) : TreeBase<T>(tree.mat, tree.leaf_size, tree.m, tree.p) {
                this->root = tree.root;
            }
            BallTree(BallTree&& tree) : TreeBase<T>(std::move(tree.mat), tree.leaf_size, tree.m, tree.p) {
                this->root = tree.root;
                tree.root = nullptr;
            }
            BallTree& operator= (const BallTree& tree) {
                this->mat = tree.mat;
                this->leaf_size = tree.leaf_size;
                this->m = tree.m;
                this->p = tree.p;
                this->root = tree.root;
                return *this;
            }
            BallTree& operator= (BallTree&& tree) {
                this->mat = std::move(tree.mat);
                this->leaf_size = tree.leaf_size;
                this->m = tree.m;
                this->p = tree.p;
                this->root = tree.root, tree.root = nullptr;
                return *this;
            }

            private:
                Matrix<T> mat;
                size_t leaf_size, p;
                Metric m;
                
                // Ref. https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote16.html
                typename TreeBase<T>::node* partition(size_t ind[], size_t nrow) {
                    size_t ncol = mat.colNum();
                    typename TreeBase<T>::node* root = this->init_node(nrow, ncol);
                    // Find centroid
                    for (size_t i = 0; i < nrow; ++i) 
                        for (size_t j = 0; j < ncol; ++j) {
                            root->split_node[j] += mat(i, j);
                            if (i == nrow - 1) root->split_node[j] /= nrow;
                        }
                    // Single node case
                    if (nrow == 1) { 
                        root->radius = 0, root->is_leaf = 1, root->indices[0] = ind[0], root->size = 1;
                        return root;
                    }
                    // Store indices for current subtree
                    std::copy(ind, ind + nrow, root->indices);
                    root->size = nrow;
                    // Locate the first node that is the furthest from centroid
                    size_t first_node_ind;
                    for (size_t i = 0; i < nrow; ++i) {
                        double dist;
                        switch (m) { 
                            case Metric::EUCLIDEAN:
                                dist = UTIL_BASE::MODEL_UTIL::METRICS::euclidean(&mat(ind[i], 0), root->split_node, ncol);
                                break;
                            case Metric::MANHATTAN:
                                dist = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(&mat(ind[i], 0), root->split_node, ncol);
                                break;
                            case Metric::MINKOWSKI:
                                dist = UTIL_BASE::MODEL_UTIL::METRICS::minkowski(&mat(ind[i], 0), root->split_node, ncol, p);
                                break;
                        }
                        if (dist > root->radius) root->radius = dist, first_node_ind = ind[i];
                    }
                    if (nrow <= leaf_size) { // Leaf nodes; no further splits are needed
                        root->is_leaf = 1;
                        return root;
                    }
                    // Locate the second node that is the furthest from first node
                    size_t second_node_ind;
                    double max_dist = 0;
                    for (size_t i = 0; i < nrow; ++i) {
                        double dist;
                        switch (m) { 
                            case Metric::EUCLIDEAN: 
                                dist = UTIL_BASE::MODEL_UTIL::METRICS::euclidean(&mat(ind[i], 0), &mat(first_node_ind, 0), ncol);
                                break;
                            case Metric::MANHATTAN:
                                dist = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(&mat(ind[i], 0), &mat(first_node_ind, 0), ncol);
                                break;
                            case Metric::MINKOWSKI:
                                dist = UTIL_BASE::MODEL_UTIL::METRICS::minkowski(&mat(ind[i], 0), &mat(first_node_ind, 0), ncol, p);
                                break;
                        }
                        if (dist > max_dist) max_dist = dist, second_node_ind = ind[i];
                    }

                    // Project data on to (first_node - second_node)
                    T diff[ncol]; memset(diff, 0, sizeof(diff));
                    std::vector<std::pair<T, size_t>> Z(nrow, std::make_pair(0, 0));
                    for (size_t i = 0; i < ncol; ++i) diff[i] = mat(first_node_ind, i) - mat(second_node_ind, i);
                    for (size_t i = 0; i < nrow; ++i) {
                        Z[i].second = i;
                        for (size_t j = 0; j < ncol; ++j) {
                            Z[i].first += diff[j] * mat(ind[i], j);
                        }
                    }

                    // Construct left and right child indices
                    sort(Z.begin(), Z.end());
                    size_t mid = nrow >> 1;
                    size_t left_child_indices[mid], right_child_indices[nrow - mid];
                    for (size_t i = 0; i < mid; ++i) 
                        left_child_indices[i] = Z[i].second, right_child_indices[i] = Z[i + mid].second;
                    for (size_t i = mid; i < nrow - mid; ++i) right_child_indices[i] = Z[i + mid].second;
                    root->left_child = partition(left_child_indices, mid);
                    root->right_child = partition(right_child_indices, nrow - mid);

                    return root;
                }

                // Ref. https://www.cs.cmu.edu/~agray/clsfnn.pdf
                void query(typename TreeBase<T>::node* root, const T const * point, const size_t n_neighbors, 
                        std::priority_queue<std::pair<double, size_t>>& point_set, double dist_min) {
                    if (!root) return;
                    if (dist_min == -1) dist_min = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(point, root->split_node, mat.colNum());
                    if (point_set.size() >= n_neighbors && dist_min >= point_set.top().first) return;
                    if (root->is_leaf) 
                        for (size_t i = 0, n = root->size; i < n; ++i) {
                            double cur_dist = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(point, &mat(root->indices[i], 0), mat.colNum());
                            if (point_set.size() < n_neighbors || point_set.top().first > cur_dist) {
                                point_set.push(std::make_pair(cur_dist, root->indices[i]));
                                if (point_set.size() == n_neighbors + 1) point_set.pop();
                            }
                        }
                    else {
                        double left_dist  = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(point, root->left_child->split_node, mat.colNum()),
                               right_dist = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(point, root->right_child->split_node, mat.colNum());
                        // Child closest to point -> Child furthest to point
                        if (left_dist < right_dist) {
                            query(root->left_child, point, n_neighbors, point_set, left_dist);
                            query(root->right_child, point, n_neighbors, point_set, right_dist);
                        } else {
                            query(root->right_child, point, n_neighbors, point_set, right_dist);
                            query(root->left_child, point, n_neighbors, point_set, left_dist);
                        }
                    }
                }
        };
    }
}