#include <fstream>
#include <random>
#include <iostream>
#include <vector>

template <typename inDtype, typename outDtype>
// inDtype means the data type for the input values
// outDtype means the data type for output values
class NeuralNetwork {
private:
    int inputsize;
    int outputsize;
    int depth;
    int width;

    std::vector<std::vector<double>> inputweights;
    std::vector<std::vector<std::vector<double>>> hiddenweights;
    std::vector<std::vector<double>> outputweights;
    
public:
    NeuralNetwork(int insize, int outsize, int hiddenDepth, int hiddenWidth, double minWeightVal = -1.0, double maxWeightVal = 1.0){
        // depth represent number of hidden layer
        // width represent number of neurons in each hidden layer
        if (insize == 0 || outsize == 0){
            std::cerr << "Input or Output size 0 not allowed" << std::endl;
        }

        if (hiddenDepth == 0 || hiddenWidth == 0){
            std::cerr << "Depth or Width of hidden layer 0 not allowed" << std::endl;
        }

        this->inputsize = insize;
        this->outputsize = outsize;
        this->depth = hiddenDepth;
        this->width = hiddenWidth;

        std::random_device rd;
        std::mt19937 random(rd());
        std::uniform_real_distribution<> dis(minWeightVal, maxWeightVal);

        // assigning the required space
        inputweights = std::vector<std::vector<double>>(width + 1, std::vector<double>(insize + 1, 0));
        hiddenweights = std::vector<std::vector<std::vector<double>>>(depth-1, std::vector<std::vector<double>>(width+1, std::vector<double>(width+1, 0)));
        outputweights = std::vector<std::vector<double>>(outsize, std::vector<double>(width + 1, 0));

        // now initialising with random weights
        for (int i=0; i < width+1; ++i){
            for (int j=0; j < insize+1; ++j){
                inputweights[i][j] = dis(random); // a random weight from negWeightRange to posWeightRange
            }
        }

        // now initialising with random weights
        for (int i=0; i < depth-1; ++i){
            for (int j=0; j < width+1; ++j){
                for (int k=0; k < width+1; ++k){
                    hiddenweights[i][j][k] = dis(random); // a random weight
                }
            }
        }

        // now initialising with random weights
        for (int i=0; i < outsize; ++i){
            for (int j=0; j < width+1; ++j){
                outputweights[i][j] = dis(random); // a random weight
            }
        }    
    }

    double sigmoid(double z){
        // returns a value between 0 and 1
        return 1/(1 + exp(-z));
    }

    double sigmoidDerivative(double z){
        // returns the derivative of the sigmoid function
        double sig = sigmoid(z);
        return sig*(1-sig);
    }

    double relu(double z){
        // rectified linear unit
        if (z < 0) return 0;
        else return z;
    }

    double reluDerivative(double z){
        // returns the derivative of the relu function
        if (z < 0) return 0;
        else return 1;
    }

    std::vector<std::vector<double>> feedForwardValues(const std::vector<inDtype>& inputVector){
        // it returns all the values in the Neural Network including the hidden values
        std::vector<std::vector<double>> layerVal(depth+1, std::vector<double>(width+1, 0));

        if (inputVector.size() != inputsize){
            std::cerr << "Input size is not matching" << std::endl;
        }

        // getting the first layerval from the input layer
        for (int i=0; i < width; ++i){
            for (int j=0; j < inputsize; ++j){
                layerVal[0][i] += inputVector[j] * inputweights[i][j];
            }
            layerVal[0][i] += inputweights[i][inputsize]; // bias value
        }
        layerVal[0][width] = 1; // for next bias value

        // for the hidden layers
        for (int k=1; k < depth; ++k){
            for (int i=0; i < width; ++i){
                for (int j=0; j < width+1; ++j){
                    layerVal[k][i] += sigmoid(layerVal[k-1][j]) * hiddenweights[k-1][i][j];
                }
            }
            layerVal[k][width] = 1; // for next bias value
        }

        // getting the output
        for (int i=0; i < outputsize; ++i){
            for (int j=0; j < width+1; ++j){
                layerVal[depth][i] += sigmoid(layerVal[depth-1][j]) * outputweights[i][j];
            }
        }

        return layerVal;
    }

    std::vector<double> feedForward(const std::vector<inDtype>& inputVector){
        // returns the outputVector based upon the current Neural Network
        auto layerVal = feedForwardValues(inputVector);

        std::vector<double> outputVector(outputsize, 0);
        for (int i=0; i < outputsize; ++i){
            outputVector[i] = sigmoid(layerVal[depth][i]);
        }

        return outputVector;
    }

    double RMSError(const std::vector<inDtype>& inputVector, const std::vector<outDtype>& outputVector){
        // calculates the root mean square error for a particular input vector
        std::vector<double> curOutputVector = feedForward(inputVector);
        double error = 0;
        for (int i=0; i < outputsize; ++i){
            error += (curOutputVector[i] - outputVector[i]) * (curOutputVector[i] - outputVector[i]) / outputsize;
        }
        return sqrt(error);
    }

    std::vector<std::vector<double>> getGradient(const std::vector<std::vector<double>>& layerVal, const std::vector<std::vector<double>>& gradient, int curLayer){
        std::vector<std::vector<double>> newGradient(width+1, std::vector<double>(width+1, 0));

        for (int i=0; i < width+1; ++i){
            for (int j=0; j < width+1; ++j){
                if (curLayer == depth-2){
                    for (int k=0; k < outputsize; ++k){
                        newGradient[i][j] += gradient[k][width] * outputweights[k][i];
                    }
                } else {
                    for (int k=0; k < width+1; ++k){
                        newGradient[i][j] += gradient[k][width] * hiddenweights[curLayer+1][k][i];
                    }
                }
                newGradient[i][j] *= sigmoidDerivative(layerVal[curLayer+1][i]) * layerVal[curLayer][j];
            }
        }

        return newGradient;
    }

    void backpropagate(const std::vector<inDtype>& inputVector, const std::vector<outDtype>& outputVector, double LearningRate){
        // updates the weights based upon the outputVector
        auto layerVal = feedForwardValues(inputVector);
        
        std::vector<std::vector<double>> gradient(width+1, std::vector<double>(width+1, 0));

        // calculating the gradient for the output layer
        for (int i=0; i < outputsize; ++i){
            for (int j=0; j < width+1; ++j){
                gradient[i][j] = 2 * (sigmoid(layerVal[depth][i]) - outputVector[i]) * sigmoidDerivative(layerVal[depth][i]) * layerVal[depth-1][j] / outputsize;
            }
        }

        // updating the output weights
        for (int i=0; i < outputsize; ++i){
            for (int j=0; j < width+1; ++j){
                outputweights[i][j] -= LearningRate * gradient[i][j];
            }
        }

        // for the hidden layers
        for (int k=depth-2; k >= 0; --k){
            gradient = getGradient(layerVal, gradient, k);
            for (int i=0; i < width+1; ++i){
                for (int j=0; j < width+1; ++j){
                    hiddenweights[k][i][j] -= LearningRate * gradient[i][j];
                }
            }
        }

        // for the input layer
        std::vector<std::vector<double>> newGradient(width+1, std::vector<double>(inputsize+1, 0));
        for (int i=0; i < width+1; ++i){
            for (int j=0; j < inputsize; ++j){
                for (int k=0; k < width+1; ++k){
                    newGradient[i][j] += gradient[k][width] * hiddenweights[0][k][i];
                }
                newGradient[i][j] *= sigmoidDerivative(layerVal[0][i]) * inputVector[j];
            }

            // now since the input vector does not have any bias term
            for (int k=0; k < width+1; ++k){
                newGradient[i][inputsize] += gradient[k][width] * hiddenweights[0][k][i];
            }
            newGradient[i][inputsize] *= sigmoidDerivative(layerVal[0][i]);
        }

        for (int i=0; i < width+1; ++i){
            for (int j=0; j < inputsize+1; ++j){
                inputweights[i][j] -= LearningRate * newGradient[i][j];
            }
        }
    }

    void train(const std::vector<std::vector<inDtype>>& inVectors, const std::vector<std::vector<outDtype>>& outVectors, int epoch, double LearningRate){

        if (inVectors.size() != outVectors.size() || inVectors.size() == 0 || outVectors.size() == 0){
            std::cerr << "Incomplete trainning data" << std::endl;
            return;
        }
        
        int trainingSize = inVectors.size();

        for (int i=0; i < epoch; ++i){
            for (int j=0; j < trainingSize; ++j){
                backpropagate(inVectors[j], outVectors[j], LearningRate);
            }

            std::cout << "Epoch " << i+1 << "/" << epoch << " completed. ";
            double error = 0;
            for (int j=0; j < trainingSize; ++j){
                error += RMSError(inVectors[j], outVectors[j]) * RMSError(inVectors[j], outVectors[j]) / trainingSize;
            }

            std::cout << "RMS Error: " << sqrt(error) << std::endl;
        }
    }

};

int main(){
    /*
    NeuralNetwork<double, double> nn(3, 1, 10, 20);
    std::cout << nn.feedForward({-1, -1, -1})[0] << std::endl;
    nn.train({{-1, -1, -1}, {0, 0, 0}, {5, 5, 5}, {2.5, 2.5, 2.5}, {3, 1, -1}, {2, -1, 0}, {0, 0, 0.1}, {-1, 0, 1}}, {{1}, {1}, {1}, {1}, {0}, {0}, {0}, {0}}, 10000, 0.001);
    std::cout << nn.feedForward({-1, -1, -1})[0] << std::endl;
    while (1){
        double x, y, z;
        std::cin >> x >> y >> z;
        std::cout << nn.feedForward({x, y, z})[0] << std::endl;
    }
    */

    std::random_device rd;
    std::mt19937 random(rd());
    std::uniform_real_distribution<> dis(-100, 100);

    NeuralNetwork<double, double> nn(1, 1, 20, 20);
    std::vector<std::vector<double>> inputvector;
    std::vector<std::vector<double>> outputvector;
    for (int i=0; i<1000; ++i){
        std::vector<double> in; std::vector<double> out;
        double num = dis(random);
        in.push_back(num);
        out.push_back(sin(num));
        inputvector.push_back(in);
        outputvector.push_back(out);
    }

    nn.train(inputvector, outputvector, 1000, 0.07);

}