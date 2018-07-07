//g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename testandoKPPrybil.cpp .cpp` testandoKPPrybil.cpp `pkg-config --libs opencv`

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

//Incluindo todas as bibliotecas std do c/c++
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

int main(){
	Mat input = imread("../dataset/2D/distance/100/100.LDR.jpg", IMREAD_UNCHANGED);
	
	double x, y;
	
	while(cin>>x>>y){
		circle(input, Point (x, y), 4, Scalar(0, 0, 255), 1, 8, 0);
	}
	
	imwrite("100.R.prybil.jpg", input);
	
	return 0;
}
