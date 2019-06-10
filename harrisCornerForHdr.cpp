//g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename harrisCornerForHdr.cpp .cpp` harrisCornerForHdr.cpp `pkg-config --libs opencv`

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

//Incluindo todas as bibliotecas std do c/c++
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

const int INF = (int) 1e9;
const float k = 0.04;//Constante calculo response 

//Criando imagens do tipo Mat
FILE *in, *out0, *out1, *out2, *out3;

Mat input, inputGray, Ix, Iy, Ix2, Iy2, Ixy, response;
Mat roi[4];

bool isHDR = false;

int quantKeyPoints = 0;
int gaussianSize = 9;

float thresholdValue = 0, sumVal;

vector<pair<int, int> > keyPoint;	

//função para aplicar a tranformação logaritmica na image HDR
//Parametros: c constante de multiplicacao da formula
void logTranform(int c){
	for(int y = 0; y < input.rows; y++){
		for(int x = 0; x < input.cols; x++){
			float r = inputGray.at<float>(y, x);
			float val = c * log10(r + 1);
			inputGray.at<float>(y, x) = val;
		}
	}
}

void logTranformUchar(int c){
	for(int y = 0; y < input.rows; y++){
		for(int x = 0; x < input.cols; x++){
			uchar r = inputGray.at<uchar>(y, x);
			uchar val = c * log10(r + 1);
			inputGray.at<uchar>(y, x) = val;
		}
	}
}
	
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

//Pegando maior valor numa imagem float cinza
float getMaxValue(Mat src1, int xBeg, int xEnd,int yBeg, int yEnd){
  float maior = -1e22;
  for(int y = yBeg; y < yEnd; y++){
	for(int x = xBeg; x < xEnd; x++){
		maior = max(maior, src1.at<float>(y, x));
	}
  }
  return maior;
}

float getMinValue(Mat src1){
  float menor = 1e22;
  for(int y = 0; y < src1.rows; y++){
	for(int x = 0; x < src1.cols; x++){
		menor = min(menor, src1.at<float>(y, x));
	}
  }
  return menor;
}

//Calcula o Response map para obter os Keypoints retornando o maior valor de resposta encontrado
void responseCalc(){
	response = Mat::zeros(cv::Size(input.cols, input.rows), CV_32F);
	for(int row = 0; row < input.rows; row++){
		for(int col = 0; col < input.cols; col++){
			float fx2 = Ix2.at<float>(row, col);
			float fy2 = Iy2.at<float>(row, col);
			float fxy = Ixy.at<float>(row, col);
			float det = (fx2 * fy2) - (fxy * fxy);
			float trace = (fx2 + fy2);
			response.at<float>(row, col) = det - k*(trace*trace);
			if(response.at<float>(row, col) < 0) response.at<float>(row, col) = 0; 
			if (row >= roi[0].rows || col >= roi[0].cols) response.at<float>(row, col) = 0; 
			else if (roi[0].at<uchar>(row, col) == 0) response.at<float>(row, col) = 0;
		}
	}
}

//Passando o Limiar na imagem resultante response
void thresholdR(){
	//Valor dentro da area externa
	int begX = 0, begY = 0; 
	int endX = response.cols, endY = response.rows;
		
	float maior = getMaxValue(response, begX, endX, begY, endY);
	//threshold "nulo" para o RR
	thresholdValue = maior * 0.000000005; 
	
	
	for(int row = begY; row < endY; row++){
		for(int col = begX; col < endX; col++){
			float val = response.at<float>(row, col);
			if(val >= thresholdValue){
				keyPoint.push_back(make_pair(row, col));
			}else response.at<float>(row, col) = 0;
		}
	}
	quantKeyPoints = (int)keyPoint.size();
}

//Conferindo se o valor a ser acessado está dentro dos limites da ROI
bool outOfBounds(int i, int j){
	return (i < 0 || j < 0 || i >= input.rows || j >= input.cols);
}

//Selecionando os Keypoinst com a Non Maxima supression
void nonMaximaSupression(){
	//Selecionando os vizinhos
	int maskSize = 21;
	//Criando vector auxiliar
	vector<pair<int, int> > aux;
	
	Mat responseCopy = Mat::zeros(cv::Size(input.cols, input.rows), CV_32F);
	
	for(int i = 0; i < (int)keyPoint.size(); i++){
		bool isMax = true;
		int y = keyPoint[i].first;
		int x = keyPoint[i].second;
		float mid = response.at<float>(y, x);
		
		int mSize2 = maskSize/2;
		for(int k = y - mSize2; k <= y + mSize2; k++){
			for(int j = x - mSize2; j <= x + mSize2; j++){
				if(!outOfBounds(k, j)){
					if(mid < response.at<float>(k, j)){ //Se ele não for o maior dentre os vizinhos
						isMax = false;
						break;
					}
				}
			}
		}		
		if(isMax){
			aux.push_back(make_pair(y, x));
			responseCopy.at<float>(y, x) = response.at<float>(y, x);
		}else{
			responseCopy.at<float>(y, x) = 0;
		}
	}
	//Passando o vector atualizado de volta 
	response = responseCopy;
	keyPoint.clear();
	keyPoint = aux;
	quantKeyPoints = keyPoint.size();
	
}//Fim função

//Coeficiente de Variacao
Mat coefficienceOfVariationMask(){

	if(inputGray.depth() != CV_32F)
		inputGray.convertTo(inputGray, CV_32F);
	
	response = inputGray;
	
	Mat auxResponse = Mat::zeros(cv::Size(response.cols, response.rows), CV_32F);
	
	int n = 5;//maskSize impar
	int N = n*n, cont = 0;//quantidade de pixels visitados
	
	float mediaGeral = 0; 
	
	Mat response2 = Mat::zeros(cv::Size(response.cols, response.rows), CV_64F);
	//response * response 
	for(int y = 0; y < response.rows; y++)
		for(int x = 0; x < response.cols; x++)
			response2.at<float>(y, x) = (response.at<float>(y, x) * response.at<float>(y, x));
	
	//Testando a gaussiana 5x5 na média
	
	
   
	
	float gerador[] = {0.06136,	0.24477,	0.38774,	0.24477,	0.06136};
	Mat gaussianBox = Mat::zeros(cv::Size(5, 5), CV_32F);
	Mat gen1 = cv::Mat(1, 5, CV_32F, gerador);
	Mat gen2 = cv::Mat(5, 1, CV_32F, gerador);
	gaussianBox = gen2*gen1;
	
	float SUM = 0; // A soma de todos os valores da mascara gaussiana 5x5
	
	for(int R = 0; R < 5; R++)
		for(int C = 0; C < 5; C++)
			SUM += gaussianBox.at<float>(R, C);
		
	cout<<"S:"<< SUM <<endl;
	cout<<gaussianBox<<endl;
	
	//"Convolution"
	for(int i = (n/2)+1; i < response.rows - (n/2); i++){
		int yBeg = i-(n/2), yEnd = i+(n/2);
		for(int j = (n/2); j < response.cols - (n/2); j++){
			//passando mascara 
			float sumVal = 0, sumVal2 = 0, maior = 0;
			int xBeg = j-(n/2), xEnd = j+(n/2);
			
			for(int y = yBeg, I = 0; y <= yEnd; y++, I++){
				for(int x = xBeg, J = 0; x <= xEnd; x++, J++){
					sumVal += (response.at<float>(y, x) * gaussianBox.at<float>(I, J));
					sumVal2 += (response2.at<float>(y, x));
				}
			}
			
			float media = sumVal/SUM;
			
			float variancia = (sumVal2/N) - (media*media);
			float S = sqrt(variancia); // desvio padrao
			float CV = media == 0? 0 : S/media; // Coef de Variacao
			auxResponse.at<float>(i, j) = CV;
		}
	}
	//Response recebe o valor de coef salvo em aux
	response = auxResponse;
	
	Mat aux2;
	normalize(response, aux2, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	
	return aux2;
}

void showKeyPoints(){
	
	for(int i = 0; i < (int)keyPoint.size(); i++){
		int x = keyPoint[i].first;
		int y = keyPoint[i].second;
		//printf("%d %d\n", x, y);
		circle(input, Point (y, x), 6, Scalar(0, 0, 255), 1, 8, 0);
		circle(input, Point (y, x), 5, Scalar(0, 0, 255), 1, 8, 0);
		circle(input, Point (y, x), 4, Scalar(0, 0, 255), 1, 8, 0);
		circle(input, Point (y, x), 3, Scalar(0, 255, 0), 1, 8, 0);
		circle(input, Point (y, x), 2, Scalar(0, 255, 0), 1, 8, 0);
		circle(input, Point (y, x), 1, Scalar(0, 255, 0), 1, 8, 0);
	}
}

//Mostrando na tela o response map em forma de imagens com cores falsas
void showResponse(string name){
	Mat imResponse;
	
	normalize(response, imResponse, 0, 255, NORM_MINMAX, CV_8UC1, Mat());

	applyColorMap(imResponse, imResponse, COLORMAP_JET);
	
	for(int row = 0; row < imResponse.rows; row++)
		for(int col = 0; col < imResponse.cols; col++)
			if(response.at<float>(row, col) == 0 && name != "Antes Th")
				imResponse.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
	
	//imshow(name, imResponse); waitKey(0);
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
    
    int quantMaxKP = 500;
    
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
	
	//Salvando Distribution Rate
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

void saveKeypoints2ROIs(){
	printf("Salvando keypoints 2 ROIs no arquivo...\n");
	
	vector<pair<float, pair<int, int> > > aux1, aux2, aux3;
	vector<pair<float, pair<int, int> > > aux;
	
    for(int i = 0; i < (int)keyPoint.size(); i++){
		int y = keyPoint[i].first, x = keyPoint[i].second;
		aux.push_back({-response.at<float>(y, x), {y, x}});
	}
    sort(aux.begin(), aux.end());
    
    int quantMaxKP = 500;
    
    for(int i = 0; i < quantMaxKP && i < aux.size(); i++){
	 	int y = aux[i].second.first, x = aux[i].second.second;
	 	if(roi[1].at<uchar>(y, x) != 0) aux1.push_back({-response.at<float>(y, x), {y, x}});
	 	
	 	else if(roi[3].at<uchar>(y, x) != 0) aux3.push_back({-response.at<float>(y, x), {y, x}});
    }
    
    double T = aux1.size() + aux3.size();
	
	double minFp = min(aux1.size()/T, aux3.size()/T);
	double maxFp = max(aux1.size()/T, aux3.size()/T);
	double D = 1 - (maxFp - minFp);
	
	//Salvando Distribution Rate
	fprintf(out0, "%.4f\n", D);
    
	keyPoint.clear();
	for(int i = 0; i < aux1.size(); i++)
		keyPoint.push_back({aux1[i].second.first, aux1[i].second.second});
	
	for(int i = 0; i < aux3.size(); i++)
		keyPoint.push_back({aux3[i].second.first, aux3[i].second.second});
	
	//Salvando pontos ROI 1
	fprintf(out1, "%d\n", (int)aux1.size());
	for(int i = 0; i < (int)aux1.size(); i++)
		fprintf(out1, "%d %d %.4f\n", aux1[i].second.first, aux1[i].second.second, -aux1[i].first);
	fclose(out1);
	
	//Salvando pontos ROI 3
	fprintf(out3, "%d\n", (int)aux3.size());
	for(int i = 0; i < (int)aux3.size(); i++)
		fprintf(out3, "%d %d %.4f\n", aux3[i].second.first, aux3[i].second.second, -aux3[i].first);
	fclose(out3);
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
	
	saida1 += ".harrisForHDR.distribution.txt";
	saida2 += ".harrisForHDR1.txt";
	saida3 += ".harrisForHDR2.txt";
	saida4 += ".harrisForHDR3.txt";
	
	out0 = fopen(saida1.c_str(), "w+");
	out1 = fopen(saida2.c_str(), "w+");
	out2 = fopen(saida3.c_str(), "w+");
	out3 = fopen(saida4.c_str(), "w+");
		
	
	read(argv[1], argv[2]);
	
	//Inicalizando com a gaussiana
	//GaussianBlur(inputGray, inputGray, Size(gaussianSize,gaussianSize), 0.4, 0.4, BORDER_DEFAULT);
	
	inputGray = coefficienceOfVariationMask();

	inputGray = inputGray.mul(30);
	imwrite("in1.png", inputGray);
	
	//Inicalizando com a gaussiana
	GaussianBlur(inputGray, inputGray, Size(gaussianSize,gaussianSize), 2.4, 2.4, BORDER_DEFAULT);
	
	//Computando sobel operator (derivada da gaussiana) no eixo x e y
	Sobel(inputGray, Ix, CV_32F, 1, 0, 7, 1, 0, BORDER_DEFAULT);
	Sobel(inputGray, Iy, CV_32F, 0, 1, 7, 1, 0, BORDER_DEFAULT);
	
	Ix2 = Ix.mul(Ix); // Ix^2
	Iy2 = Iy.mul(Iy);// Iy^2
	Ixy = Ix.mul(Iy);// Ix * Iy
	
	//Aplicamos a Gaussiana em cada duma das imagens anteriores
	GaussianBlur(Ix2, Ix2, Size(gaussianSize,gaussianSize), 0, 0, BORDER_DEFAULT);
	GaussianBlur(Iy2, Iy2, Size(gaussianSize,gaussianSize), 0, 0, BORDER_DEFAULT);
	GaussianBlur(Ixy, Ixy, Size(gaussianSize,gaussianSize), 0, 0, BORDER_DEFAULT);
	
	//Calculando resposta da derivada 
	responseCalc();
	
	Mat aux1;
	normalize(response, aux1, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	aux1*=5;
	imwrite("response_2.png", aux1);
	
	//Reescalando o response para o range [0.0-1000.0]
	normalize(response, response, 0.0, 1000.0, NORM_MINMAX, CV_32FC1, Mat());

	//Threshold
	thresholdR();
	
	quantKeyPoints = (int)keyPoint.size();
	printf("quantidade de KeyPoints depois threshold: %d\n", quantKeyPoints);
	
	//Fazendo NonMaximaSupression nos keypoints encontrados
	nonMaximaSupression();
	
	//showResponse("Depois nonMax Th");
	printf("quantidade final KeyPoints: %d\n", quantKeyPoints);
	
	//Salvando quantidade de Keypoints e para cada KP as coordenadas (x, y) e o response
	//saveKeypoints(); 
	saveKeypoints2ROIs();
	
	/*
	//Armengue para ver imagem hdr	---------------------------------------------------------
	for(int y = 0; y < input.rows; y++){
		for(int x = 0; x < input.cols; x++){
			input.at<Vec3f>(y, x)[0] *= 300;
			input.at<Vec3f>(y, x)[1] *= 300;
			input.at<Vec3f>(y, x)[2] *= 300;	
		}
	}
	*/
	
	showKeyPoints();
		
	//Salvando imagens com Keypoints
	int len = strlen(saida);
	saida[len] = 'R';saida[len+1] = '.';saida[len+2] = 'j';saida[len+3] = 'p';saida[len+4] = 'g';saida[len+5] = '\0';
	
	imwrite(saida, input);
	
	//Mostrando imagem com Keypoints
	//imshow("Result", input);
	//waitKey(0);
	
	return 0;
}

