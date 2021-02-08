/**
    Defines the Layer class
    @file Layer.cpp
    @author Gautam Sharma
    @version 1.1 01/31/2021
*/


#include "../include/Layer.hpp"
#include "../src/Tensor.cpp"
using std::cout;
using std::endl;
using std::vector;

using namespace FastNet::Layer;
using namespace FastNet::Tensor;
using Tensorf = FastNet::Tensor::Tensor<float>;


class Tensor;

Layer::Layer(int rows, int cols) : _Rows{rows}, _Cols{cols}
{
    for (int i = 0; i < this->_Rows; i++)
    {
        vector<float> tmp;
        for (int j = 0; j < this->_Cols; j++)
        {
            tmp.push_back((float)rand() / (RAND_MAX));
        }
        Tensorf _T{tmp}; // constructs a Tensor
        _Weights.push_back(_T);                 // Pushes back the Tensor to make a single row
    }

}; // random number between 0 and 1
const void Layer::Print() const
{
    cout << "Layer( " << endl;

    for (int i = 0; i < this->_Rows; i++)
    {
        _Weights[i].Print();
    }
    cout << ")" << endl;
}

const Tensorf &Layer::operator()(const int &position) const
{
    return this->_Weights.at(position);
}

size_t Layer::GetRows() const
{
    return this->_Rows;
}

constexpr size_t Layer::GetCols() const
{
    return this->_Cols;
}

std::vector<Tensorf> Layer::GetWeights() const
{
    return this->_Weights;
}


vector<Tensorf> Layer::dot(std::vector<Tensorf> t1) // // returns a tensor

{
    int cols_input = t1[0].GetTensor().size(); // columns of input tensor
    int num_examples = t1.size(); // number of examples in our input or number of rows 
    cout<<num_examples<<endl;
    cout<<cols_input<<endl;
    //cout<<sz<<endl;
    assert(cols_input == this->_Cols); // chech if column of input = rows of second matrix

    vector<Tensorf> result;
    
    //for (int i = 0; i<num_examples; i++){
        // auto row = t1.back();
        // t1.pop_back();
        for (int j=this->_Rows-1; j >=0; j--)
        {    std::vector<float> tensor;  // temperory vector that holds the tensor
            
            for (int i=0;i<num_examples;i++){
            auto row  = t1.at(i); 

            
            //t1.pop_back();
            row.Print();
            this->_Weights[j].Print();
            auto tmp = (this->_Weights[j]).dot(row);
            tensor.push_back(tmp);
            }

        

    //std::reverse(tensor.begin(), tensor.end());
    tensor.resize(num_examples);
    Tensorf t{tensor};
    result.push_back(t);
    }
    //std::reverse(result.begin(), result.end());
    //result.resize(num_examples);
    return result;

}
