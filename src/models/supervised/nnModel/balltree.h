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
                // Ref. https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote16.html
                typename TreeBase<T>::node* partition(size_t ind[], size_t nrow) {
                    size_t ncol = this->mat.colNum();
                    typename TreeBase<T>::node* root = this->init_node(nrow, ncol);
                    // Find centroid
                    for (size_t i = 0; i < nrow; ++i) 
                        for (size_t j = 0; j < ncol; ++j) {
                            root->split_node[j] += this->mat(i, j);
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
                        switch (this->m) { 
                            case Metric::EUCLIDEAN:
                                dist = UTIL_BASE::MODEL_UTIL::METRICS::euclidean(&this->mat(ind[i], 0), root->split_node, ncol);
                                break;
                            case Metric::MANHATTAN:
                                dist = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(&this->mat(ind[i], 0), root->split_node, ncol);
                                break;
                            case Metric::MINKOWSKI:
                                dist = UTIL_BASE::MODEL_UTIL::METRICS::minkowski(&this->mat(ind[i], 0), root->split_node, ncol, this->p);
                                break;
                        }
                        if (dist > root->radius) root->radius = dist, first_node_ind = ind[i];
                    }
                    if (nrow <= this->leaf_size) { // Leaf nodes; no further splits are needed
                        root->is_leaf = 1;
                        return root;
                    }
                    // Locate the second node that is the furthest from first node
                    size_t second_node_ind;
                    double max_dist = 0;
                    for (size_t i = 0; i < nrow; ++i) {
                        double dist;
                        switch (this->m) { 
                            case Metric::EUCLIDEAN: 
                                dist = UTIL_BASE::MODEL_UTIL::METRICS::euclidean(&this->mat(ind[i], 0), &this->mat(first_node_ind, 0), ncol);
                                break;
                            case Metric::MANHATTAN:
                                dist = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(&this->mat(ind[i], 0), &this->mat(first_node_ind, 0), ncol);
                                break;
                            case Metric::MINKOWSKI:
                                dist = UTIL_BASE::MODEL_UTIL::METRICS::minkowski(&this->mat(ind[i], 0), &this->mat(first_node_ind, 0), ncol, this->p);
                                break;
                        }
                        if (dist > max_dist) max_dist = dist, second_node_ind = ind[i];
                    }

                    // Project data on to (first_node - second_node)
                    T diff[ncol]; memset(diff, 0, sizeof(diff));
                    std::vector<std::pair<T, size_t>> Z(nrow, std::make_pair(0, 0));
                    for (size_t i = 0; i < ncol; ++i) diff[i] = this->mat(first_node_ind, i) - this->mat(second_node_ind, i);
                    for (size_t i = 0; i < nrow; ++i) {
                        Z[i].second = i;
                        for (size_t j = 0; j < ncol; ++j) Z[i].first += diff[j] * this->mat(ind[i], j);
                    }

                    // Construct left and right child indices
                    sort(Z.begin(), Z.end());
                    size_t mid = nrow >> 1;
                    size_t left_child_indices[mid], right_child_indices[nrow - mid];
                    for (size_t i = 0; i < mid; ++i) left_child_indices[i] = Z[i].second, right_child_indices[i] = Z[i + mid].second;
                    for (size_t i = mid; i < nrow - mid; ++i) right_child_indices[i] = Z[i + mid].second;
                    root->left_child = partition(left_child_indices, mid), root->right_child = partition(right_child_indices, nrow - mid);
                    return root;
                }
        };
    }
}