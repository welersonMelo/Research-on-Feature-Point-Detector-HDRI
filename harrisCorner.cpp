#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
// incluindo todas as bibliotecas std do c/c++
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

const int INF = (int) 1e9;
const float k = 0.04;//Constante calculo response 

//Criando imagens do tipo Mat
Mat input, inputGray, Ix, Iy, Ix2, Iy2, Ixy, response;
bool isHDR = false;
int quantKeyPoints = 0;
int gaussianSize = 3;
float thresholdValue = 0;
vector<pair<int, int> > keyPoint;

//Abrindo imagem no argumento da linha de comando
void read(char *name){	
	input = imread(name, IMREAD_UNCHANGED);
	//Gerando imagem grayscale
	cvtColor(input, inputGray, COLOR_BGR2GRAY);
	//Conferindo se é HDR
	if(input.depth() == CV_32F) {
		isHDR = true;
		printf("Imagem HDR\n");
	}
	else isHDR = false;
}

//Pegando maior e menor valor numa imagem
float getMaxValue(Mat src1){
  float maior = -INF;
  for(int row = 0; row < src1.rows; row++){
	for(int col = 0; col < src1.cols; col++){
		//printf("%.3f ", src1.at<float>(row, col));
		maior = max(maior, src1.at<float>(row, col));
	}
	//printf("\n");
  }
  
  return maior;
}

//Calcula o Response map para obter os Keypoints retornando o maior valor de resposta encontrado
float responseCalc(){
	float maior = -INF;
	response = Mat::zeros(cv::Size(input.rows, input.cols), CV_32F);
	for(int row = 0; row < input.rows; row++){
		for(int col = 0; col < input.cols; col++){
			float fx2 = Ix2.at<float>(row, col);
			float fy2 = Iy2.at<float>(row, col);
			float fxy = Ixy.at<float>(row, col);
			float det = (fx2 * fy2) - (fxy * fxy);
			float trace = (fx2 + fy2);
			response.at<float>(row, col) = det - k*(trace*trace);
			maior = max(maior, response.at<float>(row, col));
		}
	}
	return maior;
}

//Passando o Limiar na imagem resultante response
void thresholdR(){
	//Atualizando threshold
	thresholdValue = thresholdValue * 0.01;
	for(int row = 0; row < input.rows; row++){
		for(int col = 0; col < input.cols; col++){
			float val = response.at<float>(row, col);
			if(val >= thresholdValue){
				keyPoint.push_back(make_pair(row, col));
			}else response.at<float>(row, col) = 0;
		}
	}
	quantKeyPoints = (int)keyPoint.size();
}


//Conferindo se o valor a ser acessado está dentro dos limites da imagem
bool outOfBounds(int i, int j){
	return (i < 0 || j < 0 || i >= input.rows || j >= input.cols);
}

//Selecionando os Keypoinst com a Non Maxima supression
void nonMaximaSupression(){
	//Selecionando os vizinhos
	int maskSize = 3;
	int quantVizinhos = maskSize*maskSize;
	int dir1[] = {0, -1, -1, -1, 0, 1,  1, 1};
	int dir2[] = {1, 1 , 0, -1, -1, -1, 0, 1};
	//Criando vector auxiliar
	vector<pair<int, int> > aux;
	
	for(int i = 0; i < (int)keyPoint.size(); i++){
		bool isMax = true;
		int x = keyPoint[i].first;
		int y = keyPoint[i].second;
		float mid = response.at<float>(x, y);
		for(int k = 0; k < quantVizinhos; k++){
			int I = dir1[k] + x;
			int J = dir2[k] + y;
			if(!outOfBounds(I, J)){
				if(mid < response.at<float>(I, J)){ //Se ele não for o maior dentre os vizinhos
					isMax = false;
					break;
				}
			}
		}
		if(isMax){
			aux.push_back(make_pair(x, y));
		}
	}
	
	//Passando o vector atualizado de volta 
	keyPoint.clear();
	keyPoint = aux;
	quantKeyPoints = keyPoint.size();
	
}//Fim função

void showKeyPoints(){
	for(int i = 0; i < (int)keyPoint.size(); i++){
		int x = keyPoint[i].first;
		int y = keyPoint[i].second;
		//printf("%d %d\n", x, y);
		circle(input, Point (y, x), 2, Scalar(0, 0, 255), 1, 8, 0);
	}
}

int main(int, char** argv ){
	read(argv[1]);
	
	//Inicalizando com a gaussiana
	GaussianBlur(inputGray, inputGray, Size(gaussianSize,gaussianSize), 0, 0, BORDER_DEFAULT);
	
	//Computando sobel operator (derivada da gaussiana) no eixo x e y
	Sobel(inputGray, Ix, CV_32F, 1, 0, gaussianSize, 1, 0, BORDER_DEFAULT);
	Sobel(inputGray, Iy, CV_32F, 0, 1, gaussianSize, 1, 0, BORDER_DEFAULT);
	
	Ix2 = Ix.mul(Ix); // Ix^2
	Iy2 = Iy.mul(Iy);// Iy^2
	Ixy = Ix.mul(Iy);// Ix * Iy
	
	//Aplicamos a Gaussiana em cada duma das imagens anteriores
	GaussianBlur(Ix2, Ix2, Size(gaussianSize,gaussianSize), 0, 0, BORDER_DEFAULT);
	GaussianBlur(Iy2, Iy2, Size(gaussianSize,gaussianSize), 0, 0, BORDER_DEFAULT);
	GaussianBlur(Ixy, Ixy, Size(gaussianSize,gaussianSize), 0, 0, BORDER_DEFAULT);
	
	thresholdValue = responseCalc();
	imshow("r", response);
	
	thresholdR();
	
	imshow("r2", response);
	
	printf("quantidade de KeyPoints depois threshold: %d\n", quantKeyPoints);
	
	nonMaximaSupression();
	
	printf("quantidade final KeyPoints: %d\n", quantKeyPoints);
	
	showKeyPoints();
	
	imshow("Result", input);
	
	waitKey(0);
	
	return 0;
}
