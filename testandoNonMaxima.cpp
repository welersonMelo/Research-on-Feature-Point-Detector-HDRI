//g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename surf.cpp .cpp` surf.cpp `pkg-config --libs opencv`

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

//Incluindo todas as bibliotecas std do c/c++
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

int main(){
	
	String nome = "response1.png";
	
	Mat input = imread(nome, IMREAD_UNCHANGED);
	
	Mat aux = input; //Mat::zeros(Size(input.cols, input.rows), CV_8UC1);
	
	int cont = 0;
	
	for(int x = 22; x < input.cols-22; x++){
		for(int y = 22; y < input.rows-22; y++){
			uchar mid = input.at<uchar>(y, x);
	
			bool cond = true, diferente = false;
			
			for(int i = y - 10; i <= y + 10; i++){
				for(int j = x - 10; j <= x + 10; j++){
					if(i == y && j == x) continue;
					
					if(mid != input.at<uchar>(i, j))
						diferente = true;
					
					if(mid < input.at<uchar>(i, j)){
						cond = false;
						break;
					}
				}
				if(!cond) break;
			}

			if(cond && diferente){
				aux.at<Vec3b>(y, x) = (0,0,255);
				cont++;
			}else aux.at<double>(y, x) = (0, 0, 0);
		}
	}
	
	cout<<cont<<endl;
	imwrite("teste.jpg", aux);
	
	return 0;
}
