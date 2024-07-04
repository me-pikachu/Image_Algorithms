#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

// these are the open modes
#define READ_COLOR 0
#define READ_GRAYSCALE 1
#define READ_UNCHANGED 2

// these are the PixelDepth
#define _64F double
#define _32F float
#define _32S int32_t
#define _16S int16_t
#define _8S char
#define _8U uchar

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
        memcpy(this->mat, other.mat, other.rows*other.cols*sizeof(dtype)); // memcpy would be faster
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

template <typename PixelDepth = _8U>
class Image{
private:
    // we have an random number of channels currently
    // we can have 1 or 3 or 4 no of channels
    matrix<std::vector<PixelDepth>> img_mat; // it would hold the image data

public:

    Image(){
        img_mat = matrix<std::vector<PixelDepth>>(0, 0, std::vector<PixelDepth>(0, 0));
    }

    Image(const matrix<std::vector<PixelDepth>>& other_mat){
        if (other_mat.rows == 0 || other_mat.cols == 0){
            std::cerr << "Error reading the matrix" << std::endl;
            return;
        }

        const int C = other_mat[0][0].size();
        img_mat = matrix<std::vector<PixelDepth>>(other_mat.rows, other_mat.cols, std::vector<PixelDepth>(C, 0));
        for (int x=0; x<other_mat.rows; ++x){
            for (int y=0; y<other_mat.cols; ++y){
                for (int c=0; c<C; ++c){
                    img_mat[x][y][c] = other_mat[x][y][c];
                }
            }
        }
    }

    Image(const cv::Mat& cv_mat){
        if (cv_mat.empty()){
            std::cerr << "Error reading the cv matrix" << std::endl;
            return;
        }

        if (cv_mat.channels() == 1){
            img_mat = matrix<std::vector<PixelDepth>>(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(1, 0));
            for (int x=0; x<cv_mat.rows; ++x){
                for (int y=0; y<cv_mat.cols; ++y){
                    img_mat[x][y][0] = cv_mat.at<uchar>(x,y);
                }
            }
        } else if (cv_mat.channels() == 3){
            img_mat = matrix<std::vector<PixelDepth>>(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(3, 0));
            for (int x=0; x<cv_mat.rows; ++x){
                for (int y=0; y<cv_mat.cols; ++y){
                    img_mat[x][y][0] = cv_mat.at<cv::Vec3b>(x,y)[0];
                    img_mat[x][y][1] = cv_mat.at<cv::Vec3b>(x,y)[1];
                    img_mat[x][y][2] = cv_mat.at<cv::Vec3b>(x,y)[2];
                }
            }
        } else if (cv_mat.channels() == 4){
            img_mat = matrix<std::vector<PixelDepth>>(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(4, 0));
            for (int x=0; x<cv_mat.rows; ++x){
                for (int y=0; y<cv_mat.cols; ++y){
                    img_mat[x][y][0] = cv_mat.at<cv::Vec4b>(x,y)[0];
                    img_mat[x][y][1] = cv_mat.at<cv::Vec4b>(x,y)[1];
                    img_mat[x][y][2] = cv_mat.at<cv::Vec4b>(x,y)[2];
                    img_mat[x][y][3] = cv_mat.at<cv::Vec4b>(x,y)[3];
                }
            }
        } else {
            std::cerr << "Invalid number of channels" << std::endl;
            return;
        }
    }

    Image(std::string file_path, int openmode = READ_COLOR){

        if (openmode == READ_UNCHANGED){
            // the original format is preserved
            cv::Mat cv_mat = cv::imread(file_path, cv::IMREAD_UNCHANGED);
            if (cv_mat.empty()){
                std::cerr << "Error opening the image" << std::endl;
                return;
            }

            if (cv_mat.channels() == 1){
                img_mat = matrix<std::vector<PixelDepth>>(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(1, 0));
                for (int x=0; x<cv_mat.rows; ++x){
                    for (int y=0; y<cv_mat.cols; ++y){
                        img_mat[x][y][0] = cv_mat.at<uchar>(x,y);
                    }
                }
            } else if (cv_mat.channels() == 3){
                img_mat = matrix<std::vector<PixelDepth>>(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(3, 0));
                for (int x=0; x<cv_mat.rows; ++x){
                    for (int y=0; y<cv_mat.cols; ++y){
                        img_mat[x][y][0] = cv_mat.at<cv::Vec3b>(x,y)[0];
                        img_mat[x][y][1] = cv_mat.at<cv::Vec3b>(x,y)[1];
                        img_mat[x][y][2] = cv_mat.at<cv::Vec3b>(x,y)[2];
                    }
                }
            } else if (cv_mat.channels() == 4){
                img_mat = matrix<std::vector<PixelDepth>>(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(4, 0));
                for (int x=0; x<cv_mat.rows; ++x){
                    for (int y=0; y<cv_mat.cols; ++y){
                        img_mat[x][y][0] = cv_mat.at<cv::Vec4b>(x,y)[0];
                        img_mat[x][y][1] = cv_mat.at<cv::Vec4b>(x,y)[1];
                        img_mat[x][y][2] = cv_mat.at<cv::Vec4b>(x,y)[2];
                        img_mat[x][y][3] = cv_mat.at<cv::Vec4b>(x,y)[3];
                    }
                }
            } else {
                std::cerr << "Invalid number of channels" << std::endl;
                return;
            }

        } else if (openmode == READ_COLOR){
            // 3 channels RGB mode
            cv::Mat cv_mat = cv::imread(file_path, cv::IMREAD_COLOR);
            if (cv_mat.empty()){
                std::cerr << "Error opening the image" << std::endl;
                return;
            }

            img_mat = matrix<std::vector<PixelDepth>>(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(3, 0));
            for (int x=0; x<cv_mat.rows; ++x){
                for (int y=0; y<cv_mat.cols; ++y){
                    img_mat[x][y][0] = cv_mat.at<cv::Vec3b>(x,y)[0];
                    img_mat[x][y][1] = cv_mat.at<cv::Vec3b>(x,y)[1];
                    img_mat[x][y][2] = cv_mat.at<cv::Vec3b>(x,y)[2];
                }
            }

        } else if (openmode == READ_GRAYSCALE){
            cv::Mat cv_mat = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
            if (cv_mat.empty()){
                std::cerr << "Error opening the image" << std::endl;
                return;
            }

            img_mat = matrix<std::vector<PixelDepth>>(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(1, 0));
            for (int x=0; x<cv_mat.rows; ++x){
                for (int y=0; y<cv_mat.cols; ++y){
                    img_mat[x][y][0] = cv_mat.at<uchar>(x,y);
                }
            }

        } else {
            std::cerr << "Invalid openmode" << std::endl;
            return;
        }
    }

    void read(const matrix<std::vector<PixelDepth>>& other_mat){
        if (other_mat.rows == 0 || other_mat.cols == 0){
            std::cerr << "Error reading the matrix" << std::endl;
            return;
        }

        const int C = other_mat[0][0].size();
        img_mat.resize(other_mat.rows, other_mat.cols, std::vector<PixelDepth>(C, 0));
        for (int x=0; x<other_mat.rows; ++x){
            for (int y=0; y<other_mat.cols; ++y){
                for (int c=0; c<C; ++c){
                    img_mat[x][y][c] = other_mat[x][y][c];
                }
            }
        }
    }

    void read(const cv::Mat& cv_mat){
        if (cv_mat.empty()){
            std::cerr << "Error reading the cv matrix" << std::endl;
            return;
        }

        if (cv_mat.channels() == 1){
            img_mat.resize(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(1, 0));
            for (int x=0; x<cv_mat.rows; ++x){
                for (int y=0; y<cv_mat.cols; ++y){
                    img_mat[x][y][0] = cv_mat.at<uchar>(x,y);
                }
            }
        } else if (cv_mat.channels() == 3){
            img_mat.resize(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(3, 0));
            for (int x=0; x<cv_mat.rows; ++x){
                for (int y=0; y<cv_mat.cols; ++y){
                    img_mat[x][y][0] = cv_mat.at<cv::Vec3b>(x,y)[0];
                    img_mat[x][y][1] = cv_mat.at<cv::Vec3b>(x,y)[1];
                    img_mat[x][y][2] = cv_mat.at<cv::Vec3b>(x,y)[2];
                }
            }
        } else if (cv_mat.channels() == 4){
            img_mat.resize(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(4, 0));
            for (int x=0; x<cv_mat.rows; ++x){
                for (int y=0; y<cv_mat.cols; ++y){
                    img_mat[x][y][0] = cv_mat.at<cv::Vec4b>(x,y)[0];
                    img_mat[x][y][1] = cv_mat.at<cv::Vec4b>(x,y)[1];
                    img_mat[x][y][2] = cv_mat.at<cv::Vec4b>(x,y)[2];
                    img_mat[x][y][3] = cv_mat.at<cv::Vec4b>(x,y)[3];
                }
            }
        } else {
            std::cerr << "Invalid number of channels" << std::endl;
            return;
        }
    }

    void read(std::string file_path, int openmode = READ_COLOR){
        if (openmode == READ_UNCHANGED){
            // the original format is preserved
            cv::Mat cv_mat = cv::imread(file_path, cv::IMREAD_UNCHANGED);
            if (cv_mat.empty()){
                std::cerr << "Error opening the image" << std::endl;
                return;
            }

            if (cv_mat.channels() == 1){
                img_mat.resize(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(1, 0));
                for (int x=0; x<cv_mat.rows; ++x){
                    for (int y=0; y<cv_mat.cols; ++y){
                        img_mat[x][y][0] = cv_mat.at<uchar>(x,y);
                    }
                }
            } else if (cv_mat.channels() == 3){
                img_mat.resize(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(3, 0));
                for (int x=0; x<cv_mat.rows; ++x){
                    for (int y=0; y<cv_mat.cols; ++y){
                        img_mat[x][y][0] = cv_mat.at<cv::Vec3b>(x,y)[0];
                        img_mat[x][y][1] = cv_mat.at<cv::Vec3b>(x,y)[1];
                        img_mat[x][y][2] = cv_mat.at<cv::Vec3b>(x,y)[2];
                    }
                }
            } else if (cv_mat.channels() == 4){
                img_mat.resize(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(4, 0));
                for (int x=0; x<cv_mat.rows; ++x){
                    for (int y=0; y<cv_mat.cols; ++y){
                        img_mat[x][y][0] = cv_mat.at<cv::Vec4b>(x,y)[0];
                        img_mat[x][y][1] = cv_mat.at<cv::Vec4b>(x,y)[1];
                        img_mat[x][y][2] = cv_mat.at<cv::Vec4b>(x,y)[2];
                        img_mat[x][y][3] = cv_mat.at<cv::Vec4b>(x,y)[3];
                    }
                }
            } else {
                std::cerr << "Invalid number of channels" << std::endl;
                return;
            }

        } else if (openmode == READ_COLOR){
            // 3 channels RGB mode
            cv::Mat cv_mat = cv::imread(file_path, cv::IMREAD_COLOR);
            if (cv_mat.empty()){
                std::cerr << "Error opening the image" << std::endl;
                return;
            }

            img_mat.resize(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(3, 0));
            for (int x=0; x<cv_mat.rows; ++x){
                for (int y=0; y<cv_mat.cols; ++y){
                    img_mat[x][y][0] = cv_mat.at<cv::Vec3b>(x,y)[0];
                    img_mat[x][y][1] = cv_mat.at<cv::Vec3b>(x,y)[1];
                    img_mat[x][y][2] = cv_mat.at<cv::Vec3b>(x,y)[2];
                }
            }

        } else if (openmode == READ_GRAYSCALE){
            cv::Mat cv_mat = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
            if (cv_mat.empty()){
                std::cerr << "Error opening the image" << std::endl;
                return;
            }

            img_mat.resize(cv_mat.rows, cv_mat.cols, std::vector<PixelDepth>(1, 0));
            for (int x=0; x<cv_mat.rows; ++x){
                for (int y=0; y<cv_mat.cols; ++y){
                    img_mat[x][y][0] = cv_mat.at<uchar>(x,y);
                }
            }

        } else {
            std::cerr << "Invalid openmode" << std::endl;
            return;
        }
    }

    cv::Mat getCVMat() const {
        int cv_type;
        if constexpr (std::is_same_v<PixelDepth, double>) {
            cv_type = CV_64F;
        } else if constexpr (std::is_same_v<PixelDepth, float>) {
            cv_type = CV_32F;
        } else if constexpr (std::is_same_v<PixelDepth, int32_t>) {
            cv_type = CV_32S;
        } else if constexpr (std::is_same_v<PixelDepth, int16_t>) {
            cv_type = CV_16S;
        } else if constexpr (std::is_same_v<PixelDepth, uint16_t>) {
            cv_type = CV_16U;
        } else if constexpr (std::is_same_v<PixelDepth, char>) {
            cv_type = CV_8S;
        } else if constexpr (std::is_same_v<PixelDepth, uchar>) {
            cv_type = CV_8U;
        } else {
            std::cerr << "Invalid PixelDepth" << std::endl;
            cv::Mat cv_mat(0, 0, CV_MAKETYPE(CV_8U, 1));
            return cv_mat;
        }

        cv::Mat cv_mat(img_mat.rows, img_mat.cols, CV_MAKETYPE(cv_type, channels()));
        if (channels() == 1){
            for (int x=0; x<img_mat.rows; ++x){
                for (int y=0; y<img_mat.cols; ++y){
                    cv_mat.at<uchar>(x, y) = img_mat[x][y][0];
                }
            }
        } else if (channels() == 3){
            for (int x=0; x<img_mat.rows; ++x){
                for (int y=0; y<img_mat.cols; ++y){
                    cv_mat.at<cv::Vec3b>(x, y)[0] = img_mat[x][y][0];
                    cv_mat.at<cv::Vec3b>(x, y)[1] = img_mat[x][y][1];
                    cv_mat.at<cv::Vec3b>(x, y)[2] = img_mat[x][y][2];
                }
            }
        } else if (channels() == 4){
            for (int x=0; x<img_mat.rows; ++x){
                for (int y=0; y<img_mat.cols; ++y){
                    cv_mat.at<cv::Vec4b>(x, y)[0] = img_mat[x][y][0];
                    cv_mat.at<cv::Vec4b>(x, y)[1] = img_mat[x][y][1];
                    cv_mat.at<cv::Vec4b>(x, y)[2] = img_mat[x][y][2];
                    cv_mat.at<cv::Vec4b>(x, y)[3] = img_mat[x][y][3];
                }
            }
        } else {
            std::cerr << "Invalid number of channels" << std::endl;
            cv::Mat cv_mat(0, 0, CV_MAKETYPE(CV_8U, 1));
            return cv_mat;
        }

        return cv_mat;
    }

    cv::Mat getCVMat(matrix<std::vector<PixelDepth>> other_mat) const {
        int cv_type;
        if constexpr (std::is_same_v<PixelDepth, double>) {
            cv_type = CV_64F;
        } else if constexpr (std::is_same_v<PixelDepth, float>) {
            cv_type = CV_32F;
        } else if constexpr (std::is_same_v<PixelDepth, int32_t>) {
            cv_type = CV_32S;
        } else if constexpr (std::is_same_v<PixelDepth, int16_t>) {
            cv_type = CV_16S;
        } else if constexpr (std::is_same_v<PixelDepth, uint16_t>) {
            cv_type = CV_16U;
        } else if constexpr (std::is_same_v<PixelDepth, char>) {
            cv_type = CV_8S;
        } else if constexpr (std::is_same_v<PixelDepth, uchar>) {
            cv_type = CV_8U;
        } else {
            std::cerr << "Invalid PixelDepth" << std::endl;
            cv::Mat cv_mat(0, 0, CV_MAKETYPE(CV_8U, 1));
            return cv_mat;
        }

        if (other_mat.rows == 0 || other_mat.cols == 0){
            std::cerr << "Error reading the matrix" << std::endl;
            cv::Mat cv_mat(0, 0, CV_MAKETYPE(CV_8U, 1));
            return cv_mat;
        }
        const int C = other_mat[0][0].size();

        cv::Mat cv_mat(other_mat.rows, other_mat.cols, CV_MAKETYPE(cv_type, C));
        if (C == 1){
            for (int x=0; x<other_mat.rows; ++x){
                for (int y=0; y<other_mat.cols; ++y){
                    cv_mat.at<uchar>(x, y) = other_mat[x][y][0];
                }
            }
        } else if (C == 3){
            for (int x=0; x<other_mat.rows; ++x){
                for (int y=0; y<other_mat.cols; ++y){
                    cv_mat.at<cv::Vec3b>(x, y)[0] = other_mat[x][y][0];
                    cv_mat.at<cv::Vec3b>(x, y)[1] = other_mat[x][y][1];
                    cv_mat.at<cv::Vec3b>(x, y)[2] = other_mat[x][y][2];
                }
            }
        } else if (C == 4){
            for (int x=0; x<other_mat.rows; ++x){
                for (int y=0; y<other_mat.cols; ++y){
                    cv_mat.at<cv::Vec4b>(x, y)[0] = other_mat[x][y][0];
                    cv_mat.at<cv::Vec4b>(x, y)[1] = other_mat[x][y][1];
                    cv_mat.at<cv::Vec4b>(x, y)[2] = other_mat[x][y][2];
                    cv_mat.at<cv::Vec4b>(x, y)[3] = other_mat[x][y][3];
                }
            }
        } else {
            std::cerr << "Invalid number of channels" << std::endl;
            cv::Mat cv_mat(0, 0, CV_MAKETYPE(CV_8U, 1));
            return cv_mat;
        }
        return cv_mat;
    }

    void write(std::string file_path) const {
        cv::Mat cv_mat = getCVMat();
        bool success = cv::imwrite(file_path, cv_mat);
        if (!success){
            std::cerr << "Error saving the image!" << std::endl;
            assert(0);
        }
    }

    void write(matrix<std::vector<PixelDepth>> other_mat, std::string file_path) const {
        cv::Mat cv_mat = getCVMat(other_mat);
        bool success = cv::imwrite(file_path, cv_mat);
        if (!success){
            std::cerr << "Error saving the image!" << std::endl;
            assert(0);
        }
    }

    void show(std::string name = "Img Window", int delay = 0) const {
        // name is the windows name
        cv::Mat cv_mat = getCVMat();
        cv::imshow(name, cv_mat);
        cv::waitKey(delay); // if delay is 0 then the img would be forever until
        // a key stroke is pressed
    }

    void show(matrix<std::vector<PixelDepth>> other_mat, std::string name = "Img Window", int delay = 0) const {
        // name is the windows name
        cv::Mat cv_mat = getCVMat(other_mat);
        cv::imshow(name, cv_mat);
        cv::waitKey(delay); // if delay is 0 then the img would be forever until
        // a key stroke is pressed
    }

    PixelDepth depth(){
        // returns the current depth of the image
        return PixelDepth();
    }

    int channels() const {
        if (img_mat.rows == 0 || img_mat.cols == 0){
            return 0;
        } else {
            return img_mat[0][0].size();
        }
    }

    std::vector<PixelDepth>& at(int x, int y) const {
        if (img_mat.rows == 0 || img_mat.cols == 0){
            std::cerr << "The image is empty" << std::endl;
        } else {
            return img_mat[x][y];
        }
    }

    PixelDepth& at(int x, int y, int c) const {
        if (img_mat.rows == 0 || img_mat.cols == 0){
            std::cerr << "The image is empty" << std::endl;
        } else {
            if (channels() <= c){
                std::cerr << "Channel out of range" << std::endl;
            } else {
                return img_mat[x][y][c];
            }
        }
    }

    matrix<std::vector<PixelDepth>> getGrayImg(std::vector<float> mult = {0.299, 0.587, 0.114}) const {
        if (mult.empty()){
            if (channels() == 1){
                // already in grayscale mode
                mult = {1};
            } else if (channels() == 3){
                // rgb mode
                mult = {0.299, 0.587, 0.114};
            } else if (channels() == 4){
                // rgba mode
                mult = {0.299, 0.587, 0.114, 0};
            } else {
                std::cerr << "Invalid number of channels" << std::endl;
                return matrix<std::vector<PixelDepth>>(); // Empty matrix
            }
        }

        if (mult.size() != channels()){
            std::cerr << "The size of the multiplier must the same as the number of channels" << std::endl;
             return matrix<std::vector<PixelDepth>>(); // Empty matrix
        }

        float sum = 0;
        for (int i=0; i<mult.size(); ++i){
            sum += mult[i];
        }
        if (sum != 1.0){
            std::cerr << "The sum of the elements of the multiplier should be equal to 1" << std::endl;
             return matrix<std::vector<PixelDepth>>(); // Empty matrix
        }

        matrix<std::vector<PixelDepth>> gray_mat(img_mat.rows, img_mat.cols, std::vector<PixelDepth>(1, 0));

        for (int x=0; x < img_mat.rows; ++x){
            for (int y=0; y < img_mat.cols; ++y){
                // we need to convert to 0 to 255 value
                double pixelval = 0;
                for (int c=0; c < channels(); ++c){
                    pixelval += img_mat[x][y][c] * mult[c];
                }
                gray_mat[x][y][0] = (PixelDepth)pixelval;
            }
        }
        
        return gray_mat;
    }

    matrix<std::vector<PixelDepth>> getBlurImg(int kernel = 5) const {
        // kernel * kernel elements
        int half_kernel = kernel / 2;
        // kernel is the size of the square whose average pixel value we are gonna assign to out current pixel

        matrix<std::vector<PixelDepth>> blur_mat(img_mat.rows, img_mat.cols, std::vector<PixelDepth>(channels(), 0)); // image type is preserved

        for (int x=0; x<img_mat.rows; ++x){
            for (int y=0; y<img_mat.cols; ++y){

                int count = 0;
                std::vector<double> sum(channels(), 0); // to avoid any overflow
                for (int dx = -half_kernel; dx <= half_kernel; ++dx){
                    for (int dy = -half_kernel; dy <= half_kernel; ++dy){
                        if ((x+dx) >= 0 && (x+dx) < img_mat.rows && (y+dy) >= 0 && (y+dy) < img_mat.cols){
                            // the dimensions are correct
                            for (int c=0; c < channels(); ++c){
                                sum[c] += img_mat[x+dx][y+dy][c];
                            }
                            count++;
                        }
                    }
                }

                for (int c=0; c < channels(); ++c){
                    blur_mat[x][y][c] = (PixelDepth)(sum[c]/count);
                }
            }
        }

        return blur_mat;
    }
};

int main(){
    Image<_8U> img;
    img.read("sample_images/img1.jpeg", READ_COLOR);
    img.show("Original Image");
    img.show(img.getGrayImg(), "Grey Image");
    img.show(img.getBlurImg(0), "Blur Image");
    img.write(img.getGrayImg(), "output_images/greyimg1.jpeg");
    img.write(img.getBlurImg(), "output_images/blurimg1.jpeg");
    return 0;
}