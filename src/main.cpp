/**
    Main file to execute the library
    @file main.cpp
    @author Gautam Sharma
    @version 1.1 01/31/2021
*/

#include "../include/Tensor.hpp"
#include "../src/Tensor.cpp"
#include "../include/Functions.hpp"
#include "../include/Layer.hpp"

#include <chrono>

using Vectorf = std::vector<float>;
using Vectori = std::vector<int>;
using Tensorf = Tensor<float>;
using Tensori = Tensor<int>;
using namespace FastNet::Tensor;
using namespace FastNet::Layer;
using namespace std;
int main()
{
    auto start = chrono::high_resolution_clock::now();
    Vectorf input = {1.5, 2, 3, 4, 3, 4, 5, 6, 43, 3, 35, 21, -3.4, 0.2, 99, 100, 22, 3, 3, 2, 4, 3, 7.8, 8.99, 5.4, 3.5, 5.4, 4.5, 43, 55, 44};
    Tensorf t{input};
    // t.Print();

    Vectorf weights = {1.5, 2, 3, 4, 3, 4, 5, 6, 43, 3, 35, 21, -3.4, 0.2, 99, 100, 22, 3, 3, 2, 4, 3, 7.8, 8.99, 5.4, 3.5, 5.4, 4.5, 43, 55, 44};
    Tensorf t2{weights};
    // t.Print();

    auto dot = t.dot(t2); //dot product
                          //     auto end = chrono::high_resolution_clock::now();
                          // double time_taken =
                          //     chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                          // time_taken *= 1e-9;
                          //     cout << "Time taken by program is : " << fixed
                          //         << time_taken;
                          // cout << " sec" << endl;
    cout << dot << endl;
    t.Print();

    auto z = sigmoid(t);
    z.Print();

    auto x = Range(0, 100);
    x.Print();

    Tensorf y = sigmoid(x);
    auto y2 = sigmoid(y);
    auto y3 = sigmoid(y2);
    auto y4 = sigmoid(y3);
    y4.Print();
    cout << y4.Sum() << endl;

    // assuming a matrix
    Vectorf input1 = {1, 2, 3};
    Vectorf input2 = {4, 5, 6};
    Vectorf input3 = {7, 8, 9};

    vector<Tensorf> t1{input1, input2, input3}; // 3 by 3
    Layer l4{5, 3};                             //  5 by 3

    auto layer1 = l4.dot(t1);
    cout << "output:" << endl;
    (layer1.at(0)).Print(); //vector of tensors
    (layer1.at(1)).Print();
    (layer1.at(2)).Print();
    (layer1.at(3)).Print();
    (layer1.at(4)).Print();

    Layer l2{10, 3}; //  5 by 3
    auto tt = l2.dot(layer1);
    (tt.at(0)).Print();
    (tt.at(1)).Print();
    (tt.at(2)).Print();
    (tt.at(3)).Print();
    (tt.at(4)).Print();
    //auto x1 = (tt.at(0));
    Tensorf temp{tt.at(0)};
    temp.Print();
    Tensorf temp2 = sigmoid(temp);
    temp2.Print();

    Tensorf temp3 = relu(temp);
    temp3.Print();

    Vectorf temp4 = {-1, -1, -2, -3, 5, 6};
    Tensorf ttt{temp4};
    relu(ttt).Print();

}
