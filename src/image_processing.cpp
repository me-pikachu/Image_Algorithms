#include "Image.cpp"
#include "NeuralNetwork.cpp"

void reverseInt(int& num) {
    uchar c1, c2, c3, c4;
    c1 = num & 255;
    c2 = (num >> 8) & 255;
    c3 = (num >> 16) & 255;
    c4 = (num >> 24) & 255;
    num = ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

uchar valInt(uint32_t& val, int index, bool rev){
    if (rev){
        uchar* c = (uchar*)&val;
        if (index == 0) return *(c+3);
        else if (index == 1) return *(c+2);
        else if (index == 2) return *(c+1);
        else if (index == 3) return *(c);
        else assert(0);
    } else {
        uchar* c = (uchar*)&val;
        if (index == 0) return *(c);
        else if (index == 1) return *(c+1);
        else if (index == 2) return *(c+2);
        else if (index == 3) return *(c+3);
        else assert(0);
    }
}

std::vector<matrix<std::vector<uchar>>> MNISTreadImg(const std::string& file_path){
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()){
        std::cerr << "Error while opening the file" << std::endl;
        std::vector<matrix<std::vector<uchar>>> data;
        return data;
    }

    int magicNumber, numImages, rows, cols;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    reverseInt(magicNumber);
    if (magicNumber != 2051){
        std::cerr << "Either the file is invalid or it's corrupted" << std::endl;
    }

    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    reverseInt(numImages);
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    reverseInt(rows);
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    reverseInt(cols);

    // all the images are grayscale images
    std::vector<matrix<std::vector<uchar>>> data(numImages);
    // initialising the data
    for (int curImg=0; curImg < numImages; ++curImg){
        data[curImg] = matrix<std::vector<uchar>>(rows, cols, std::vector<uchar>(1, 0));
    }

    for (int curImg=0; curImg < numImages; ++curImg){
        for (int i=0; i<28; ++i){
            for (int j=0; j<28; ++j){
                // the image is in big endian format for reading it we should use little endian
                char pixel;
                file.read(&pixel, sizeof(pixel));
                data[curImg][i][j][0] = (uchar)pixel;
            }
        }
    }

    return data;
}

std::vector<uchar> MNISTreadLabel(const std::string& file_path){
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()){
        std::cerr << "Error while opening the file" << std::endl;
        std::vector<uchar> data(0);
        return data;
    }

    int magicNumber, numLabels;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    reverseInt(magicNumber);
    if (magicNumber != 2049){
        std::cerr << "Either the file is invalid or it's corrupted" << std::endl;
    }
    
    file.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
    reverseInt(numLabels);

    // all the images are grayscale images
    std::vector<uchar> data(numLabels, -1);
    
    for (int i=0; i < numLabels; ++i){
        char label;
        file.read(&label, sizeof(label));
        data[i] = (uchar)label;
    }

    return data;
}

int main(){
    auto train_images = MNISTreadImg("mnist_data/train_images/train-images.idx3-ubyte");
    auto train_labels = MNISTreadLabel("mnist_data/train_labels/train-labels.idx1-ubyte");

    size_t _size = 10;
    int rows = train_images[0].rows;
    int cols = train_images[0].cols;
    std::vector<std::vector<double>> input(_size);
    std::vector<std::vector<double>> output(_size);
    for (int i=0; i<_size; ++i){
        for (int r=0; r<rows; ++r){
            for (int c=0; c<cols; ++c){
                input[i].push_back(train_images[i][r][c][0]);
            }
        }
        output[i].push_back(train_labels[i]);
    }

    NeuralNetwork<double, double> nn(rows*cols, 1, 20, train_images[0].rows*cols, NeuralNetwork<double, double>::reluFunc);
    nn.train(input, output, 1000, 0.1, 0.01, 0.9);
    return 0;
}