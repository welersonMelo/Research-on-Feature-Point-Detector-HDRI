//g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename surfForHdr.cpp .cpp` surfForHdr.cpp `pkg-config --libs opencv`

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

struct KeyPoints{
	int x, y, scale;//posicao (x, y) e o octave ou escala da imagem
	double response;
};

vector<KeyPoints> keyPoint;
	
//Abrindo imagem no argumento da linha de comando
void read(char *name, char *argv2){
	input = imread(name, IMREAD_UNCHANGED);
	
	//Gerando imagem grayscale
	cvtColor(input, inputGray, COLOR_BGR2GRAY);
	
	//Conferindo se é HDR
	if(input.depth() == CV_32F) {
		isHDR = true;
		normalize(inputGray, inputGray, 0.0, 256.0, NORM_MINMAX, CV_32FC1, Mat());
		//normalize(inputGray, inputGray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
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
		
		for(int x = 0; x < input.cols; x++){
			for(int y = 0; y < input.rows; y++){
				roi[0].at<uchar>(y, x) = 1;
				roi[1].at<uchar>(y, x) = 1;
				roi[2].at<uchar>(y, x) = 1;
				roi[3].at<uchar>(y, x) = 1;
			}
		}
	}
}

//Conferindo se o valor a ser acessado está dentro dos limites da ROI
bool outOfBounds(int i, int j, Mat aux){
	return (i < 0 || j < 0 || i >= aux.rows || j >= aux.cols);
}

void showKeyPoints(){
	for(int i = 0; i < (int)keyPoint.size(); i++){
		int x = keyPoint[i].y;
		int y = keyPoint[i].x;
		
		circle(input, Point (x, y), 2, Scalar(0, 255, 0), 1, 8, 0);
		circle(input, Point (x, y), 3, Scalar(0, 255, 0), 1, 8, 0);
		circle(input, Point (x, y), 4, Scalar(0, 0, 255), 1, 8, 0);
		circle(input, Point (x, y), 5, Scalar(0, 0, 255), 1, 8, 0);
		circle(input, Point (x, y), 6, Scalar(0, 0, 255), 1, 8, 0);
	}
}

//Salvando keypoints no arquivo
void saveKeypoints(){
	printf("Salvando keypoints ROIs no arquivo...\n");
	
	vector<pair<double, pair<int, int> > > aux1, aux2, aux3;
	vector<pair<double, pair<int, int> > > aux;
	
    for(int i = 0; i < (int)keyPoint.size(); i++){
		int y = keyPoint[i].y, x = keyPoint[i].x;
		aux.push_back({-keyPoint[i].response, {y, x}});
	}
	
    sort(aux.begin(), aux.end());//Ordenando de forma decrescente
    
    int quantMaxKP = 500;
    
    for(int i = 0, k = 0; k < quantMaxKP && i < aux.size(); i++){
	 	int y = aux[i].second.first, x = aux[i].second.second;
	 	
	 	if(roi[1].at<uchar>(y, x) != 0){ aux1.push_back({aux[i].first, {y, x}}); k++;}
	 	else if(roi[2].at<uchar>(y, x) != 0){ aux2.push_back({aux[i].first, {y, x}}); k++;}
	 	else if(roi[3].at<uchar>(y, x) != 0){ aux3.push_back({aux[i].first, {y, x}}); k++;}
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

//Fazendo threshold
void threshold(){
	int cont = 0;
	int begX = 0, begY = 0; 
	int endX = inputGray.cols, endY = inputGray.rows;
	
	double T = 10;
	
	for(int c = 0; c < 2; c++){ // Escalas
		for(int z = 1; z < 3; z++){ //Layes
			
			for(int y = begY; y < endY-10; y++){
				for(int x = begX; x < endX-10; x++){
					double val = fabs(responseBlob[c][z].at<double>(y, x));
					if(val > T){						
						keyPoint.push_back({x, y, c, val});
						cont++;
					}else{
						 responseBlob[c][z].at<double>(y, x) = 0;
					}
				}
			}
			cout<<"Cont KP : "<<cont<<endl;
			cont = 0;
		}
	}
}

// 21 x 21 x 21 non Maxima Supression

void nonMaximaSupression(){
	int maskSize = 21; // Mascara de 21 x 21 baseado no artigo do prybil
	int cont = 0;
	
	for(int c = 0; c < 2; c++){ // Octave - Escalas - Voltar para 4 depois
		for(int z = 1; z < 3; z++){ // Layers
			Mat matAux = Mat::zeros(Size(inputGray.cols, inputGray.rows), CV_64F);
			
			for(int x = 22; x < responseBlob[c][z].cols-22; x++){
				for(int y = 22; y < responseBlob[c][z].rows-22; y++){
					double mid = responseBlob[c][z].at<double>(y, x);
					
					bool cond = true, diferente = false;
					int mSize2 = maskSize/2;
					
					//Maior entre vizinhos
					for(int i = y - mSize2; i <= y + mSize2; i++){
						for(int j = x - mSize2; j <= x + mSize2; j++){
							if(i == y && j == x) continue;
							
							if(mid != responseBlob[c][z].at<double>(i, j) || mid != responseBlob[c][z-1].at<double>(i, j) || mid != responseBlob[c][z+1].at<double>(i, j))
								diferente = true;
							
							if(mid <= responseBlob[c][z-1].at<double>(i, j) || mid <= responseBlob[c][z].at<double>(i, j) || mid <= responseBlob[c][z+1].at<double>(i, j)){
								cond = false;
								break;
							}
						}
						if(!cond) break;
					}
					
					if(cond && diferente){
						y += mSize2-1;
						matAux.at<double>(y, x) = responseBlob[c][z].at<double>(y, x);
						cont++;
					}else matAux.at<double>(y, x) = 0;
				}
			}
			responseBlob[c][z] = matAux;
			cout<<cont<<", ";//parcial
		}
		cout<<cont<<endl;//Mostrando quantidade de keypoints encontrado total
		cont = 0;
	}
	
}//Fim função

//Coeficiente de Variacao
Mat coefficienceOfVariationMask(Mat aux){
	
	if(aux.depth() != CV_32F)
		aux.convertTo(aux, CV_32F);
	
	Mat response = aux;
	
	Mat auxResponse = Mat::zeros(cv::Size(response.cols, response.rows), CV_32F);
	
	int n = 3;//maskSize impar
	int N = n*n, cont = 0;//quantidade de pixels visitados
	
	Mat response2 = Mat::zeros(cv::Size(response.cols, response.rows), CV_64F);
	//response * response 
	for(int y = 0; y < response.rows; y++)
		for(int x = 0; x < response.cols; x++)
			response2.at<float>(y, x) = (response.at<float>(y, x) * response.at<float>(y, x));
	
	float sum1 = 0, sum2 = 0;
	
	for(int y = 1; y < n; y++){
		for(int x = 0; x <= n; x++){
			sum1 += response.at<float>(y, x);
			sum2 += response2.at<float>(y, x);
		}
	}
	
	//"Convolution"
	for(int i = (n/2)+1; i < response.rows - (n/2); i++){
		int yBeg = i-(n/2), yEnd = i+(n/2);
		for(int j = (n/2); j < response.cols - (n/2); j++){
			//passando mascara 
			float sumVal = 0, sumVal2 = 0, maior = 0;
			int xBeg = j-(n/2), xEnd = j+(n/2);
			
			for(int y = yBeg; y <= yEnd; y++){
				for(int x = xBeg; x <= xEnd; x++){
					sumVal += response.at<float>(y, x);
					sumVal2 += response2.at<float>(y, x);
					maior = max(maior, response.at<float>(y, x));
				}
			}
						
			float media = sumVal/N;
			
			float variancia = (sumVal2/N) - (media*media);

			float S = sqrt(variancia); // desvio padrao
			float CV = media == 0? 0 : S/media; // Coef de Variacao
			auxResponse.at<float>(i, j) = CV * 100;
			
			//printf("%d %d %f\n", i, j, CV);
			//printf("%.8f %.8f %.8f %.8f %.8f %.8f\n", sumVal, sumVal2, media, variancia, S, CV);
		}
	}
	
	//Response recebe o valor de coef salvo em aux
	response = auxResponse;
	
	Mat aux2;
	normalize(response, aux2, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	
	return aux2;
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
	
	for(int i = 0; i < 2; i++){ //Octaves ---  //Colocar i de volta pra 4 depois 
		for(int j = 0; j < 4; j++){ // Layes ----
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
			int L = (int)pow(2, (i+1))*(j+1)+1;
			int s = maskSize[i][j];
			int s3 = s/3;
			double w = 0.912;
			calcDyy(i, j, s, s3, 2*s3-1);
			calcDxx(i, j, s, 2*s3-1, s3);
			calcDxy(i, j, s, s3);
			
			responseBlob[i][j] = Dxx.mul(Dyy);
			
			for(int y = 1; y < inputGray.rows ; y++){
				for(int x = 1; x < inputGray.cols ; x++){
					responseBlob[i][j].at<double>(y, x) = (1.0/(L*L*L*L))*(responseBlob[i][j].at<double>(y, x) - ((w * Dxy.at<double>(y, x)) * (w * Dxy.at<double>(y, x))));
					//responseBlob[i][j].at<double>(y, x) = (responseBlob[i][j].at<double>(y, x) - ((w * Dxy.at<double>(y, x)) * (w * Dxy.at<double>(y, x))));
				}
			}
		}
	}
		
	//Salvando responseBlob para visualização
	Mat im1, im2, im3;	
	normalize(responseBlob[0][0], im1, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(responseBlob[0][1], im2, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(responseBlob[0][2], im3, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	
	//imwrite("response0.png", im1);
	//imwrite("response1.png", im2);
	//imwrite("response2.png", im3);
	
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
	
	saida1 += ".surfForHdr.distribution.txt";
	saida2 += ".surfForHdr1.txt";
	saida3 += ".surfForHdr2.txt";
	saida4 += ".surfForHdr3.txt";
	
	out0 = fopen(saida1.c_str(), "w+");
	out1 = fopen(saida2.c_str(), "w+");
	out2 = fopen(saida3.c_str(), "w+");
	out3 = fopen(saida4.c_str(), "w+");
		
	//Lendo imagem de entrada
	read(argv[1], argv[2]);
	
	inputGray = coefficienceOfVariationMask(inputGray);
	
	//imwrite("CV image.jpg", inputGray);
	
	//Calculando e salvando imagem integral na matriz integralImg
	calculateIntegraImage(inputGray);
	
	//Calculando Fast Hessian com imagem integral e salvando
	initOctaves();
	
	//Fazendo NonMaximaSupression nos keypoints encontrados
	nonMaximaSupression();
	
	//Thresould
	threshold();
	
	//Salvando quantidade de Keypoints e para cada KP as coordenadas (x, y) e o response
	saveKeypoints(); 
	
	//Salvando pontos vermelhos na imagem 
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
