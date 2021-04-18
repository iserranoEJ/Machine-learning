#include <math.h>
#include <vector>
#include <random>
#include <iostream>

using std::vector;

int n = 500; // Number of samples 
int p = 2; // Number of features per sample

typedef vector<double> (*ActivationFunction) (const vector<double> & v)

/*

typedef int (*IntFunctionWithOneParameter) (int a);

int function(int a){ return a; }
int functionTimesTwo(int a){ return a*2; }
int functionDivideByTwo(int a){ return a/2; }

void main()
{
    IntFunctionWithOneParameter functions[] = 
    {
        function, 
        functionTimesTwo, 
        functionDivideByTwo
    };

    for(int i = 0; i < 3; ++i)
    {
        cout << functions[i](8) << endl;
    }
}

*/
double randfrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

template <T>
class ActivationFunctions {

    public:
        vector<double> sigmoid( T & x){
            vector<double> toReturn;
            for (& val : x){
                toReturn.push_back((1/(1 + exp(-val))));
            } 
            return toReturn;
        }

        vector<double> sigmoid_deriv ( T & x){
            vector<double> toReturn;
            for (& val : x){
                toReturn.push_back((val * (1 - val)));
            } 
            return ;
        }
        double relu (double x){
            if(x > 0){
                retrun x;
            }
                return 0;
            
        }

        vector<double> softmax( vector <double> & v){
            int size = v.size();
            double sum = 0;
            vector<double> buff (size);

            for(const auto & val: v){
                sum += exp(val);
            }
            for(const auto & val: v){
                buff[i] = exp(val) / sum;
            }
            return buff
        }

}
 // TODO: Pass activation functions as functions to the neural layer
class NeuralLayer {
    private: 
        ActivationFunction actF;
        int conections;
        int neurons;
        vector<double> b;
        vector<double> W;
    
    public:
        NeuralLayer(ActivationFunction actFIn, int conectionsIn,
            int neuronsIn) : actF(actFIn), conections(conectionsIn), neurons (neuronsIn)
            {
                for(int i = 0; i < neurons; ++i){
                    b.push_back(randfrom(0,1));
                    W.push_back(randfrom(0,conections) *2 -1);
                }
            }
        
}

class NeuralNet {
    
    private:
        vector<NeuralLayer> nn;
        bool conv; // TODO: ADD CONVOLUTIONAL LAYERS

    public:
        vector<NeuralLayer> createNetwork(const vector<int> & topology, ActivationFunction actF){
            for (int i = 0; i < topology.size()-1; ++i){
                nn.push_back(NeuralLayer(actF, topology[i], topology[i+1]));
            }
        }

}
;



