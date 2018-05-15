//g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename surf.cpp .cpp` surf.cpp `pkg-config --libs opencv`

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

//Incluindo todas as bibliotecas std do c/c++
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

//Criando imagens do tipo Mat
FILE *in, *out0, *out1, *out2, *out3;

Mat Dxx, Dyy, Dxy, responseBlob[5][5];
Mat input, inputGray, response;
Mat roi[4];

vector<unsigned int> integralImg[5000];

bool isHDR = false;

int quantKeyPoints = 0;

vector<pair<int, int> > keyPoint;	
	
//Abrindo imagem no argumento da linha de comando
void read(char *name, char *argv2){
	input = imread(name, IMREAD_UNCHANGED);
	
	//Gerando imagem grayscale
	cvtColor(input, inputGray, COLOR_BGR2GRAY);
	
	//Conferindo se é HDR
	if(input.depth() == CV_32F) {
		isHDR = true;
		normalize(inputGray, inputGray, 0.0, 256.0, NORM_MINMAX, CV_32FC1, Mat());
		printf("Imagem HDR\n");
	}else isHDR = false;
	
	//Lendo arquivo com os pontos (x, y) da ROI
	if(argv2 != NULL){
		string path(argv2); //../dataset/2D/distance/100/100
		string num = "";
		while(true){
			if(path.back() == '/'){
				break;
			}
			num += path.back();
			path.pop_back();
		}
		reverse(num.begin(), num.end());
		
		roi[0] = imread(path+"ROI."+num+".png", IMREAD_UNCHANGED);
		roi[1] = imread(path+"ROIh."+num+".png", IMREAD_UNCHANGED);
		roi[2] = imread(path+"ROIm."+num+".png", IMREAD_UNCHANGED);
		roi[3] = imread(path+"ROIs."+num+".png", IMREAD_UNCHANGED);
		
	}else{
		roi[0] = Mat::zeros(cv::Size(input.cols, input.rows), CV_8U);
		roi[1] = Mat::zeros(cv::Size(input.cols, input.rows), CV_8U);
		roi[2] = Mat::zeros(cv::Size(input.cols, input.rows), CV_8U);
		roi[3] = Mat::zeros(cv::Size(input.cols, input.rows), CV_8U);
		
		for(int x = 0; x < input.cols; x++)
			for(int y = 0; y < input.rows; y++)
				roi[0].at<uchar>(y, x) = 1;
	}
}

//Conferindo se o valor a ser acessado está dentro dos limites da ROI
bool outOfBounds(int i, int j){
	return (i < 0 || j < 0 || i >= input.rows || j >= input.cols);
}

void showKeyPoints(){
	
	for(int i = 0; i < (int)keyPoint.size(); i++){
		int x = keyPoint[i].first;
		int y = keyPoint[i].second;
		//printf("%d %d\n", x, y);
		circle(input, Point (y, x), 4, Scalar(0, 0, 255), 1, 8, 0);
		circle(input, Point (y, x), 3, Scalar(0, 0, 255), 1, 8, 0);
		circle(input, Point (y, x), 2, Scalar(0, 0, 255), 1, 8, 0);
		circle(input, Point (y, x), 1, Scalar(0, 0, 255), 1, 8, 0);
	}
}

void saveKeypoints(){
	printf("Salvando keypoints ROIs no arquivo...\n");
	
	vector<pair<float, pair<int, int> > > aux1, aux2, aux3;
	vector<pair<float, pair<int, int> > > aux;
	
    for(int i = 0; i < (int)keyPoint.size(); i++){
		int y = keyPoint[i].first, x = keyPoint[i].second;
		aux.push_back({-response.at<float>(y, x), {y, x}});
	}
    sort(aux.begin(), aux.end());
    
    int quantMaxKP = 400;
    
    for(int i = 0; i < quantMaxKP && i < aux.size(); i++){
	 	int y = aux[i].second.first, x = aux[i].second.second;
	 	if(roi[1].at<uchar>(y, x) != 0) aux1.push_back({-response.at<float>(y, x), {y, x}});
	 	else if(roi[2].at<uchar>(y, x) != 0) aux2.push_back({-response.at<float>(y, x), {y, x}});
	 	else if(roi[3].at<uchar>(y, x) != 0) aux3.push_back({-response.at<float>(y, x), {y, x}});
    }
    
    double T = aux1.size() + aux2.size() + aux3.size();
	
	double minFp = min(aux1.size()/T, min(aux2.size()/T, aux3.size()/T));
	double maxFp = max(aux1.size()/T, max(aux2.size()/T, aux3.size()/T));
	double D = 1 - (maxFp - minFp);
	
	//Salvando Uniformity Rate
	fprintf(out0, "%.4f\n", D);
    
	keyPoint.clear();
	for(int i = 0; i < aux1.size(); i++)
		keyPoint.push_back({aux1[i].second.first, aux1[i].second.second});
	for(int i = 0; i < aux2.size(); i++)
		keyPoint.push_back({aux2[i].second.first, aux2[i].second.second});
	for(int i = 0; i < aux3.size(); i++)
		keyPoint.push_back({aux3[i].second.first, aux3[i].second.second});
	
	//Salvando pontos ROI 1
	fprintf(out1, "%d\n", (int)aux1.size());
	for(int i = 0; i < (int)aux1.size(); i++)
		fprintf(out1, "%d %d %.4f\n", aux1[i].second.first, aux1[i].second.second, -aux1[i].first);
	fclose(out1);
	//Salvando pontos ROI 2
	fprintf(out2, "%d\n", (int)aux2.size());
	for(int i = 0; i < (int)aux2.size(); i++)
		fprintf(out2, "%d %d %.4f\n", aux2[i].second.first, aux2[i].second.second, -aux2[i].first);
	fclose(out2);
	//Salvando pontos ROI 3
	fprintf(out3, "%d\n", (int)aux3.size());
	for(int i = 0; i < (int)aux3.size(); i++)
		fprintf(out3, "%d %d %.4f\n", aux3[i].second.first, aux3[i].second.second, -aux3[i].first);
	fclose(out3);
}

//NOVAS FUNCOES DO SURF A PARTIR DAQUI

void calculateIntegraImage(Mat In){
	
	for(int y = 0; y < In.rows; y++)
		for(int x = 0; x < In.cols; x++)
			integralImg[y].push_back((int)In.at<uchar>(y, x));		
	
	for(int x = 1; x < In.cols; x++)
		integralImg[0][x] += integralImg[0][x-1];
		
	for(int y = 1; y < In.rows; y++)
		integralImg[y][0] += integralImg[y-1][0];
	
	for(int y = 1; y < In.rows; y++)
		for(int x = 1; x < In.cols; x++)
			integralImg[y][x] = integralImg[y - 1][x] + integralImg[y][x - 1] + integralImg[y][x] - integralImg[y - 1][x - 1];
}

//Calculando Dyy onde s é o tamanho da mascara, sY o tamanho da area com valor no eixo Y, e sX o tamanho da area com valor no eixo X
void calcDyy(int i, int j, int s, int sY, int sX){ 
	int xBegin = ((s-sX)/2) - 1, xEnd = xBegin + sX;
	int D1y = -1, D2y = sY-1, D3y = 2*sY-1;
	int B1y = -1, B2y = sY-1, B3y = 2*sY-1;
	int C1y = sY-1, C2y = 2*sY-1, C3y = 3*sY-1;
	int A1y = sY-1, A2y = 2*sY-1, A3y = 3*sY-1;
	
	int mid = s/2;
	for(int i = 1; i < inputGray.rows - s; i++){
		int midY = mid + i;
		for(int j = 1; j < inputGray.cols - s; j++){
			int midX = mid + j;	
			double sum1 = integralImg[A1y+i][xEnd+j] - integralImg[C1y+i][xBegin+j] - integralImg[B1y+i][xEnd+j] + integralImg[D1y+i][xBegin+j];
			double sum2 = integralImg[A2y+i][xEnd+j] - integralImg[C2y+i][xBegin+j] - integralImg[B2y+i][xEnd+j] + integralImg[D2y+i][xBegin+j];
			double sum3 = integralImg[A3y+i][xEnd+j] - integralImg[C3y+i][xBegin+j] - integralImg[B3y+i][xEnd+j] + integralImg[D3y+i][xBegin+j];
			
			double sum = -2 * sum2 + sum1 + sum3;
			
			Dyy.at<double>(midY, midX) = sum;
		}
	}
}
//Calculando Dxx onde s é o tamanho da mascara, sY o tamanho da area com valor no eixo Y, e sX o tamanho da area com valor no eixo X
void calcDxx(int i, int j, int s, int sY, int sX){ 
	int yBegin = ((s-sY)/2) - 1, yEnd = yBegin + sY;
	int D1x = -1, D2x = sX-1, D3x = 2*sX-1;
	int B1x = -1, B2x = sX-1, B3x = 2*sX-1;
	int C1x = sX-1, C2x = 2*sX-1, C3x = 3*sX-1;
	int A1x = sX-1, A2x = 2*sX-1, A3x = 3*sX-1;
	
	int mid = s/2;
	for(int i = 1; i < inputGray.rows - s; i++){
		int midY = mid + i;
		for(int j = 1; j < inputGray.cols - s; j++){
			int midX = mid + j;			
			double sum1 = integralImg[yEnd+i][A1x+j] - integralImg[yBegin+i][C1x+j] - integralImg[yEnd+i][B1x+j] + integralImg[yBegin+i][D1x+j];
			double sum2 = integralImg[yEnd+i][A2x+j] - integralImg[yBegin+i][C2x+j] - integralImg[yEnd+i][B2x+j] + integralImg[yBegin+i][D2x+j];
			double sum3 = integralImg[yEnd+i][A3x+j] - integralImg[yBegin+i][C3x+j] - integralImg[yEnd+i][B3x+j] + integralImg[yBegin+i][D3x+j];
			
			double sum = -2 * sum2 + sum1 + sum3;
			
			Dxx.at<double>(midY, midX) = sum;	
		}
	}
}

//Calculando Dxy onde s é o tamanho da mascara, sYX o tamanho da area quadrada com valor 
void calcDxy(int i, int j, int s, int sYX){ 	
	int mid = s/2;
	
	int A1x = mid-1, A1y = mid-1, B1x = A1x, B1y = A1y-sYX, D1x = A1x-sYX, D1y = B1y, C1x = D1x, C1y = A1y;
	int A2x = mid+sYX, A2y = mid-1, B2x = A2x, B2y = B1y, D2x = mid, D2y = B2y, C2x = mid, C2y = mid-1;
	int A3x = A2x, A3y = mid + sYX, B3x = A3x, B3y = mid, D3x = mid, D3y = mid, C3x = mid, C3y = A3y;
	int A4x = A1x, A4y = A3y, B4x = A4x, B4y = mid, D4x = C1x, D4y = mid, C4x = D4x, C4y = A4y;
	
	for(int i = 1; i < inputGray.rows - s; i++){
		int midY = mid + i;
		for(int j = 1; j < inputGray.cols - s; j++){
			int midX = mid + j;			
			double sum1 = integralImg[A1y+i][A1x+j] - integralImg[C1y+i][C1x+j] - integralImg[B1y+i][B1x+j] + integralImg[D1y+i][D1x+j];
			double sum2 = integralImg[A2y+i][A2x+j] - integralImg[C2y+i][C2x+j] - integralImg[B2y+i][B2x+j] + integralImg[D2y+i][D2x+j];
			double sum3 = integralImg[A3y+i][A3x+j] - integralImg[C3y+i][C3x+j] - integralImg[B3y+i][B3x+j] + integralImg[D3y+i][D3x+j];
			double sum4 = integralImg[A4y+i][A4x+j] - integralImg[C4y+i][C4x+j] - integralImg[B4y+i][B4x+j] + integralImg[D4y+i][D4x+j];
			
			double sum = sum1 - sum2 + sum3 - sum4;
			
			Dxy.at<double>(midY, midX) = sum;	
		}
	}
}

void initOctaves(){
	int maskSize[4][4] = {{9,15,21,27}, {15,27,39,51}, {27,51,75,99}, {51,99,147,195}}; //Melhorar tempo aqui depois tirando redundancia
	
	Dxx = Mat::zeros(cv::Size(inputGray.cols, inputGray.rows), CV_64F);
	Dyy = Mat::zeros(cv::Size(inputGray.cols, inputGray.rows), CV_64F);
	Dxy = Mat::zeros(cv::Size(inputGray.cols, inputGray.rows), CV_64F);
	
	for(int i = 0; i < 1; i++){  //Colocar i de volta pra 4 depois
		for(int j = 0; j < 4; j++){
			//Eliminando Redundancia - melhorar essa parte 			
			
			if(i==1&&j==0){
				responseBlob[i][j] = responseBlob[0][1];
				continue;
			}else if((i==1&&j==1)||(i==2&&j==0)){
				responseBlob[i][j] = responseBlob[0][3];
				continue;
			}else if((i==2&&j==1)||(i==3&&j==0)){
				responseBlob[i][j] = responseBlob[1][3];
				continue;
			}else if(i==3&&j==1){
				responseBlob[i][j] = responseBlob[2][3];
				continue;
			}
			
			int s = maskSize[i][j];
			int s3 = s/3;		
			calcDyy(i, j, s, s3, 2*s3-1);
			calcDxx(i, j, s, 2*s3-1, s3);
			calcDxy(i, j, s, s3);
			
			responseBlob[i][j] = Dxx.mul(Dyy);
			
			for(int y = 1; y < inputGray.rows ; y++){
				for(int x = 1; x < inputGray.cols ; x++){
					responseBlob[i][j].at<double>(y, x) -= (0,81 * (Dxy.at<double>(y, x) * Dxy.at<double>(y, x)));
				}
			}
			
		}
	}
	
	normalize(responseBlob[0][0], responseBlob[0][0], 0, 255, NORM_MINMAX, CV_8U, Mat());
	normalize(responseBlob[0][1], responseBlob[0][1], 0, 255, NORM_MINMAX, CV_8U, Mat());
	normalize(responseBlob[0][2], responseBlob[0][2], 0, 255, NORM_MINMAX, CV_8U, Mat());
	normalize(responseBlob[0][3], responseBlob[0][3], 0, 255, NORM_MINMAX, CV_8U, Mat());
	
	imwrite("response0.png", 255-responseBlob[0][0]);
	imwrite("response1.png", 255-responseBlob[0][1]);
	imwrite("response2.png", 255-responseBlob[0][2]);
	imwrite("response3.png", 255-responseBlob[0][3]);
	
	Dxx.release();
	Dyy.release();
	Dxy.release();
	
}

//Função Principal
// ROI = Region Of Interest
// Ex Chamada: 
int main(int, char** argv ){	
	char saida[255];
	strcpy(saida, argv[1]);
	saida[strlen(saida)-4] = '\0';
	string saida2(saida);
	
	string saida1 = saida2;
	string saida3 = saida2;
	string saida4 = saida3;
	
	saida1 += ".surf.distribution.txt";
	saida2 += ".surf1.txt";
	saida3 += ".surf2.txt";
	saida4 += ".surf3.txt";
	
	out0 = fopen(saida1.c_str(), "w+");
	out1 = fopen(saida2.c_str(), "w+");
	out2 = fopen(saida3.c_str(), "w+");
	out3 = fopen(saida4.c_str(), "w+");
		
	//Lendo imagem de entrada
	read(argv[1], argv[2]);
	
	//Calculando e salvando imagem integral na matriz integralImg
	calculateIntegraImage(inputGray);
	
	//Calculando Fast Hessian com imagem integral e salvando
	initOctaves();
	
	quantKeyPoints = (int)keyPoint.size();
	printf("quantidade de KeyPoints depois threshold: %d\n", quantKeyPoints);
	
	//Fazendo NonMaximaSupression nos keypoints encontrados
	//nonMaximaSupression();
	
	printf("quantidade final KeyPoints: %d\n", quantKeyPoints);
	
	//Salvando quantidade de Keypoints e para cada KP as coordenadas (x, y) e o response
	//saveKeypoints(); 
	
	//Salvando pontos vermelhos na imagem 
	//showKeyPoints();
		
	//Salvando imagens com Keypoints
	int len = strlen(saida);
	saida[len] = 'R';saida[len+1] = '.';saida[len+2] = 'j';saida[len+3] = 'p';saida[len+4] = 'g';saida[len+5] = '\0';
	
	imwrite(saida, input);
	
	//Mostrando imagem com Keypoints
	//imshow("Result", input);
	//waitKey(0);
	
	return 0;
}
