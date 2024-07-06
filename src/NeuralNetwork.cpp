#include "matrix.cpp"

#include <vector>
#include <thread>
#include <omp.h>
#include <fstream>
#include <random>


template <typename inDtype = double, typename outDtype = double>
// inDtype means the data type for the input values
// outDtype means the data type for output values
class NeuralNetwork {
public:
    enum ActivationFunc{
        sigmoidFunc,
        reluFunc
    };

private:
    int inSize = 0;
    int outSize = 0;
    int depth = 0; // no of hidden layers
    int width = 0; // no of neurons in a hidden layer
    // all the hidden layer has same no of neurons
    ActivationFunc actFunc;
    
    std::vector<matrix<double>> weights; // weigth[k][i][j] represents ...
    // ... the (k-1)th hidden layer's jth neuron to (k)th hidden layer's ith neuron

    // for optimisers
    matrix<double> momentum;
    matrix<double> RMSProp;
    
public:
    NeuralNetwork(const int& inputSize, const int& outputSize, const int& hiddenDepth, const int& hiddenWidth,  ActivationFunc actiFunc = NeuralNetwork::sigmoidFunc, const double& minWeightVal = -1.0, const double& maxWeightVal = 1.0){
        // depth represent number of hidden layer
        // width represent number of neurons in each hidden layer
        if (inputSize == 0 || outputSize == 0){
            std::cerr << "Input or Output size 0 not allowed" << std::endl;
            return;
        }

        if (hiddenDepth == 0 || hiddenWidth == 0){
            std::cerr << "Depth or Width of hidden layer 0 not allowed" << std::endl;
            return;
        }

        if (hiddenWidth < std::max(inputSize, outputSize)){
            std::cerr << "Width of hidden layer cannot be less than inputSize or outputSize" << std::endl;
            return;
        }

        this->inSize = inputSize;
        this->outSize = outputSize;
        this->depth = hiddenDepth;
        this->width = hiddenWidth;
        this->actFunc = actiFunc;

        std::random_device rd;
        std::mt19937 random(rd());
        std::uniform_real_distribution<> dis(minWeightVal, maxWeightVal);

        // assigning the required space
        double zero = 0.0;
        weights = std::vector(depth+1, matrix<double>(width+1, width+1));
        momentum = matrix<double>(depth+1, width+1, zero);
        RMSProp = matrix<double>(depth+1, width+1, zero);
        
        // now initialising with random weights
        #pragma omp parallel for schedule(dynamic)
        for (int k=0; k<depth+1; ++k){
            for (int i=0; i<width+1; ++i){
                for (int j=0; j<width+1; ++j){
                    weights[k][i][j] = dis(random);
                }
            }
        }  

        // the output layer weights needs to be initialised to 0 which don't form the output
        for (int i=outSize; i<width+1; ++i){
            for (int j=0; j<width+1; ++j){
                weights[depth][i][j] = 0;
            }
        }

        // the input layer weights needs to be initialised to 0 which don't connect from the input
        for (int i=0; i<width+1; ++i){
            for (int j=inputSize+1; j<width+1; ++j){
                weights[depth][i][j] = 0;
            }
        }
    }

    double absolute(double z){
        return z >= 0 ? z : -z;
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

    double activationFunc(double z){
        if (actFunc == NeuralNetwork::sigmoidFunc){
            return sigmoid(z);
        } else if (actFunc == NeuralNetwork::reluFunc){
            return relu(z);
        } else {
            std::cerr << "Invalid activation function" << std::endl;
            return 0;
        }
    }

    double activationFuncDerivate(double z){
        if (actFunc == NeuralNetwork::sigmoidFunc){
            return sigmoidDerivative(z);
        } else if (actFunc == NeuralNetwork::reluFunc){
            return reluDerivative(z);
        } else {
            std::cerr << "Invalid activation function" << std::endl;
            return 0;
        }
    }

    double RMSError(const std::vector<inDtype>& inputVector, const std::vector<outDtype>& outputVector){
        // calculates the root mean square error for a particular input vector and it's corresponding expected outputVector
        std::vector<double> curOutputVector = feedForward(inputVector);
        double error = 0;
        for (int i=0; i<outSize; ++i){
            error += (curOutputVector[i] - outputVector[i]) * (curOutputVector[i] - outputVector[i]) / outSize;
        }
        return sqrt(error);
    }

    double MAPError(const std::vector<inDtype>& inputVector, const std::vector<outDtype>& outputVector){
        // calculates the mean absolute percentage error for a particular input vector and it's corresponding expected outputVector
        std::vector<double> curOutputVector = feedForward(inputVector);
        double error = 0;
        for (int i=0; i<outSize; ++i){
            if (outputVector[i] == 0) error += absolute(curOutputVector[i] - outputVector[i]) * 100 / outSize;
            else error += absolute(curOutputVector[i] - outputVector[i]) * 100 / (outSize * outputVector[i]);
        }
        return error; // returns error out of 100 percent
    }

    double MAError(const std::vector<inDtype>& inputVector, const std::vector<outDtype>& outputVector, int p){
        // calculates the mean absolute error for a particular input vector and it's corresponding expected outputVector
        std::vector<double> curOutputVector = feedForward(inputVector);
        double error = 0;
        for (int i=0; i<outSize; ++i){
            error += absolute(curOutputVector[i] - outputVector[i]) / outSize;
        }
        return sqrt(error);
    }

    matrix<double> feedForwardValues(const std::vector<inDtype>& inputVector){
        // it returns all the values in the Neural Network including the hidden values
        double zero = 0.0;
        matrix<double> layerVal(depth+2, width+1, zero); // these are the values before appyling the activation function

        if (inputVector.size() != inSize){
            std::cerr << "Input size is not matching" << std::endl;
            return matrix<double>(0, 0);
        }

        // feeding the input values in layerVal
        for (int i=0; i<inSize; ++i){
            layerVal[0][i] = inputVector[i];
        }
        layerVal[0][inSize] = 1; // for the bias value


        for (int k=1; k<depth+2; ++k){
            #pragma omp parallel for schedule(dynamic)
            for (int i=0; i<width; ++i){
                for (int j=0; j<width+1; ++j){
                    layerVal[k][i] += activationFunc(layerVal[k-1][j]) * weights[k-1][i][j];
                }
            }
            layerVal[k][width] = 1; // for the bias value
        }

        layerVal[depth+1][width] = 0;
        return layerVal;
    }

    std::vector<double> feedForward(const std::vector<inDtype>& inputVector){
        // returns the outputVector based upon the current Neural Network
        matrix<double> layerVal;
        //std::cout << layerVal.rows << layerVal.cols << std::endl;
        layerVal = feedForwardValues(inputVector);

        std::vector<double> outputVector(outSize, 0);
        for (int i=0; i<outSize; ++i){
            outputVector[i] = activationFunc(layerVal[depth+1][i]);
        }

        return outputVector;
    }

    matrix<double> getUpdatedParameters(const std::vector<matrix<double>>& layerVals, const matrix<double>& gradient, const int& cur, const int& _size, const int& beta){
        // updates momentum and RMSProp and returns the new gradient
        double zero = 0.0;
        matrix<double> newGradient(_size+1, width+1, zero);

        #pragma omp parallel for schedule(dynamic)
        for (int k=0; k<_size; ++k){
            for (int i=0; i<width+1; ++i){
                for (int j=0; j<width+1; ++j){
                    newGradient[k][j] += gradient[k][i] * weights[cur][i][j];
                }
            }
        }

        #pragma omp parallel for schedule(dynamic)
        for (int k=0; k<_size; ++k){
            for (int i=0; i<width+1; ++i){
                newGradient[k][i] *= activationFuncDerivate(layerVals[k][cur][i]);
                newGradient[_size][i] += newGradient[k][i] / _size;
            }
        }

        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<width+1; ++i){
            double grad = newGradient[_size][i];
            momentum[cur][i] = beta * momentum[cur][i] + (1 - beta) * grad; // updating the momentum
            RMSProp[cur][i] = beta * RMSProp[cur][i] + (1 - beta) * grad * grad; // updating the RMSprop
        }        

        return newGradient;
    }

    void backpropagate(const std::vector<std::vector<inDtype>>& inputVectors, const std::vector<std::vector<outDtype>>& outputVectors, const double& LearningRate, const double& beta){
        // updates the weights based upon the outputVectors
        if (inputVectors.size() != outputVectors.size()){
            std::cerr << "The Input Vector batch size is not equal to Output Vector batch size" << std::endl;
            return;
        }

        const size_t _size = inputVectors.size(); // this _size is the batch size
        std::vector<matrix<double>> layerVals(_size);

        #pragma omp parallel for schedule(dynamic)
        for (int k=0; k<_size; ++k){
            layerVals[k] = feedForwardValues(inputVectors[k]);
        }

        double zero = 0.0;
        matrix<double> gradient(_size+1, width+1, zero);

        // calculating the gradient for the output layer
        #pragma omp parallel for schedule(dynamic)
        for (int k=0; k<_size; ++k){
            for (int i=0; i<width+1; ++i){
                for (int j=0; j<width+1; ++j){
                    gradient[k][i] += 2 * (activationFunc(layerVals[k][depth+1][j]) - outputVectors[k][j]) * activationFuncDerivate(layerVals[k][depth+1][j]) * activationFunc(layerVals[k][depth][i]);
                }
            }
        }

        #pragma omp parallel for schedule(dynamic)
        for (int k=0; k<_size; ++k){
            for (int i=0; i<width+1; ++i){
                gradient[_size][i] += gradient[k][i] / _size;
            }
        }

        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<outSize; ++i){
            double grad = gradient[_size][i];
            momentum[depth][i] = beta * momentum[depth][i] + (1 - beta) * grad; // updating the momentum
            RMSProp[depth][i] = beta * RMSProp[depth][i] + (1 - beta) * grad * grad; // updating the RMSProp
        }        

        // updating the output weights
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<outSize; ++i){
            for (int j=0; j<width+1; ++j){
                // using both momentum and RMSProp approach
                if (RMSProp[depth][j] == 0) weights[depth][i][j] -= LearningRate * momentum[depth][j] / 0.001;
                else weights[depth][i][j] -= LearningRate * momentum[depth][j] / sqrt(RMSProp[depth][j]);
            }
        }

        // for the rest of the layers
        for (int k=depth-1; k>=0; --k){
            //std::cout << "Hidden Layer " << depth-k << std::endl; 
            gradient = getUpdatedParameters(layerVals, gradient, k, _size, beta);
            #pragma omp parallel for schedule(dynamic)
            for (int i=0; i<width+1; ++i){
                for (int j=0; j<width+1; ++j){
                    if (RMSProp[k][j] == 0) weights[k][i][j] -= LearningRate * momentum[k][j] / 0.001;
                    else weights[k][i][j] -= LearningRate * momentum[k][j] / sqrt(RMSProp[k][j]);
                }
            }
        }
    }

    void train(const std::vector<std::vector<inDtype>>& inVectors, const std::vector<std::vector<outDtype>>& outVectors, const int& epoch, const double& batchSize, const double& LearningRate, const double& momentumParameter){
        // train the neural network in batches
        if (inVectors.size() != outVectors.size() || inVectors.size() == 0 || outVectors.size() == 0){
            std::cerr << "Incomplete or Invalid trainning data" << std::endl;
            return;
        }
        
        int trainingSize = inVectors.size();
        int batch = trainingSize * batchSize;
        int totalBatches = trainingSize / batch;

        std::vector<std::vector<std::vector<inDtype>>> inBatches(totalBatches);
        std::vector<std::vector<std::vector<inDtype>>> outBatches(totalBatches);

        for (int i=0; i<totalBatches; ++i){
            for (int j=0; j<batch; ++j){
                inBatches[i].push_back(inVectors[i*batch + j]);
            }
        }

        for (int i=0; i<totalBatches; ++i){
            for (int j=0; j<batch; ++j){
                outBatches[i].push_back(outVectors[i*batch + j]);
            }
        }

        for (int i=0; i<epoch; ++i){
            for (int j=0; j<totalBatches; ++j){
                //std::cout << "Batch: " << j << std::endl;
                backpropagate(inBatches[j], outBatches[j], LearningRate, momentumParameter);
            }

            // our current Epoch completed
            std::cout << "Epoch " << i+1 << "/" << epoch << " completed. ";
            double error = 0;
            for (int j=0; j<trainingSize; ++j){
                error += MAPError(inVectors[j], outVectors[j]) / trainingSize;
            }
            std::cout << "Mean Absolute Percentage Error: " << error << std::endl;
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

    NeuralNetwork<double, double> nn(1, 1, 10, 5, NeuralNetwork<double, double>::sigmoidFunc);
    std::vector<std::vector<double>> inputvector;
    std::vector<std::vector<double>> outputvector;
    for (int i=0; i<10000; ++i){
        std::vector<double> in; std::vector<double> out;
        double num = dis(random);
        in.push_back(num);
        sin(num) >= 0 ? out.push_back(sin(num)) : out.push_back(-sin(num));
        inputvector.push_back(in);
        outputvector.push_back(out);
    }
    nn.train(inputvector, outputvector, 1000, 0.1, 0.01, 0.1);

    while (true){
        double x;
        std::cin >> x;
        std::cout << "The actual value: " << nn.absolute(sin(x)) << std::endl;
        std::cout << "Neural Network's prediction is: " << nn.feedForward({x})[0] << std::endl;
    }

    return 0;
}

