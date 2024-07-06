#include <iostream>
#include <cstring>

template <typename dtype> struct matrix{
    // the matrix is considered to be small
    // so we would use single pointer
    int rows, cols;
    dtype* mat = nullptr;

    matrix(){
        this->rows = 0;
        this->cols = 0;
        mat = nullptr;
    }

    matrix(int r, int c){
        // if we use single pointer then we require r*c continuous blocks of memory
        // but if we use double pointer then we require r*c+r with r and c continuous blocks of memory
        // time complexity in case of allocation in single pointer is O(1) considering malloc takes O(1) time
        // in case of double pointer it is O(r) because of the for loop
        this->mat = (dtype*)calloc(r*c,sizeof(dtype));
        if (this->mat == nullptr) {
            throw std::bad_alloc(); // Handle allocation failure
        }
        this->rows = r;
        this->cols = c;
    }

    matrix(int r, int c, const dtype& initValue){
        this->rows = r;
        this->cols = c;
        this->mat = (dtype*)calloc(r*c,sizeof(dtype));
        if (this->mat == nullptr) {
            throw std::bad_alloc(); // Handle allocation failure
        }
        for (int i = 0; i < rows * cols; ++i) {
            mat[i] = initValue;
        }
    }

    matrix(matrix& other){
        // in case of copying the object how to perform the copy
        // it is the copy constructor
        this->rows = other.rows;
        this->cols= other.cols;
        this->mat = (dtype*)malloc(other.rows*other.cols*sizeof(dtype));
        if (this->mat == nullptr) {
            throw std::bad_alloc(); // Handle allocation failure
        }
        // if this constructor was not specified then this->mat = other.mat
        // the array would not have been copied rather both of them would point to the same array
        // freeing one object would free the other too and may cause double free error and segmentation fault
        std::memcpy(this->mat, other.mat, other.rows*other.cols*sizeof(dtype)); // memcpy would be faster
    }

    matrix(matrix&& other) noexcept : rows(other.rows), cols(other.cols), mat(other.mat) {
        other.rows = 0;
        other.cols = 0;
        other.mat = nullptr;
    }

    ~matrix(){
        free(mat);
    }

    void resize(int r, int c, const dtype& value){
        this->rows = r;
        this->cols = c;
        this->mat = (dtype*)realloc(this->mat, r*c*sizeof(dtype));
        if (this->mat == nullptr) {
            throw std::bad_alloc(); // Handle allocation failure
        }
        for (int i = 0; i < rows * cols; ++i) {
            mat[i] = value;
        }
    }

    void print() const {
        for (int i=0; i < this->rows * this->cols; i=i+this->cols){
            for (int j=0; j<this->cols; ++j){
                std::cout << this->mat[i+j] << " ";
            }
            std::cout << std::endl;
        }
    }

    void print(const matrix& mat) const {
        for (int i=0; i < mat.rows * mat.cols; i=i+mat.cols){
            for (int j=0; j<mat.cols; ++j){
                std::cout << mat.mat[i+j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // to access the elements we can use matrix[rows][column]
    // but cpp does not support overloading of chaining of [] operators
    // we can make a temp struct in order to achieve this

    struct proxy{
        int cols;
        dtype* row_val;

        proxy(dtype* arr, int c){
            row_val = arr;
            cols = c;
        }

        // instead of returning a value instead return a reference
        // so that assign statement can also work
        dtype& operator[](int c) const {
            if (c < 0 || c >= cols){
                // proxy class cannot directly access the cols variable but it can directly access dtype variable
                // don't know why :(
                throw std::out_of_range("Column Index out of range");
            }
            return row_val[c];
        }
    };

    proxy operator[](int r) const {
        if (r < 0 || r >= rows){
            throw std::out_of_range("Row Index out of range");
        }
        return proxy(this->mat + r*cols, cols);
    }

    // now overload the operator + - * for matrix
    // before overloading the operator + - * we need to overload for the assignment operator
    // the assignment operator should return the lvalue because if it returns void then a = b = c .. would not work
    // it should return the reference to that matrix

    // this is basically a pointer to the current object
    matrix& operator=(const matrix<dtype>& matB){
        if (this == &matB){
            return *this;
        }

        if (this->rows != matB.rows || this->cols != matB.cols){
            // free currently assigned memory
            free(this->mat);
            this->mat = (dtype*)malloc(matB.rows * matB.cols * sizeof(dtype));
            this->rows = matB.rows;
            this->cols = matB.cols;
        }
        memcpy(this->mat, matB.mat, this->rows*this->cols*sizeof(dtype));
        return *this;
    }

    matrix& operator=(matrix&& other) noexcept {
        if (this == &other) return *this;

        free(this->mat);
        this->rows = other.rows;
        this->cols = other.cols;
        this->mat = other.mat;

        other.rows = 0;
        other.cols = 0;
        other.mat = nullptr;
        return *this;
    }

    // the return type of operator+ should not be a reference because
    // if a reference is returned it points to a local variable matC which was destroyed when
    // the function ended and hence it would use cause segmentation fault
    // mat3 = mat1 + mat2. In this case a copy of mat2 is created to matB using the copy constructor
    // we have defined above
    // mat3 = mat1 + mat2 takes overall of O(4*r*c) (4 times as in to include the time taken to copy the object) whereas
    // mat1 += mat2 takes overall of O(2*r*c) 2x faster!!
    matrix operator+(const matrix<dtype>& matB){
        if (this->rows == matB.rows && this->cols == matB.cols){
            matrix matC(this->rows, this->cols);
            for (int i=0; i < this->rows * this->cols; i=i+this->cols){
                for (int j=0; j<this->cols; ++j){
                    matC.mat[i+j] = this->mat[i+j] + matB.mat[i+j];
                }
            }
            return matC;
        } else {
            throw std::invalid_argument("Matrix dimension should match for matrix addition");
        }
    }

    matrix& operator+=(const matrix<dtype>& matB){
        if (this->rows == matB.rows && this->cols == matB.cols){
            for (int i=0; i < this->rows * this->cols; i=i+this->cols){
                for (int j=0; j<this->cols; ++j){
                    this->mat[i+j] = this->mat[i+j] + matB.mat[i+j];
                }
            }
            return *this;
        } else {
            throw std::invalid_argument("Matrix dimension should match for matrix addition");
        }
    }

    matrix operator-(const matrix<dtype>& matB){
        if (this->rows == matB.rows && this->cols == matB.cols){
            matrix matC(this->rows, this->cols);
            for (int i=0; i < this->rows * this->cols; i=i+this->cols){
                for (int j=0; j<this->cols; ++j){
                    matC.mat[i+j] = this->mat[i+j] - matB.mat[i+j];
                }
            }
            return matC;
        } else {
            throw std::invalid_argument("Matrix dimension should match for matrix subtraction");
        }
    }

    matrix& operator-=(const matrix<dtype>& matB){
        if (this->rows == matB.rows && this->cols == matB.cols){
            for (int i=0; i < this->rows * this->cols; i=i+this->cols){
                for (int j=0; j<this->cols; ++j){
                    this->mat[i+j] = this->mat[i+j] - matB.mat[i+j];
                }
            }
            return *this;
        } else {
            throw std::invalid_argument("Matrix dimension should match for matrix subtraction");
        }
    }

    // let's say matA -> (m,n) and matB -> (n,p)
    // matA * matB -> (m,p)
    // Time complexity O(m*n*p) and in general we can say that O(n^3)
    // Matrix multiplication can be improved considering computer caches
    matrix operator*(const matrix<dtype>& matB){
        if (this->cols == matB.rows){
            matrix matC(this->rows, matB.cols);
            for (int i=0; i<this->rows; ++i){
                for (int k=0; k<matB.rows; ++k){
                    for (int j=0; j<matB.cols; ++j){
                        matC.mat[i*matB.cols+j] = (matC.mat[i*matB.cols+j] + this->mat[i*this->cols+k] * matB.mat[k*matB.cols+j]);
                    }
                }
            }
            return matC;
        } else {
            throw std::invalid_argument("Matrix dimension should match for matrix multiplication");
        }
    }

    matrix& operator*=(const matrix<dtype>& matB){
        if (this->cols == matB.rows){
            for (int i=0; i<this->rows; ++i){
                for (int k=0; k<matB.rows; ++k){
                    for (int j=0; j<matB.cols; ++j){
                        this->mat[i*matB.cols+j] = (this->mat[i*matB.cols+j] + this->mat[i*this->cols+k] * matB.mat[k*matB.cols+j]);
                    }
                }
            }
            return *this;
        } else {
            throw std::invalid_argument("Matrix dimension should match for matrix multiplication");
        }
    }
};