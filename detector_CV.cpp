//g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename detector_CV.cpp .cpp` detector_CV.cpp `pkg-config --libs opencv`

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

Mat input, inputGray;
Mat roi[4];
bool nullRoi = false;

bool isHDR = false;

int quantKeyPoints = 0;
int gaussianSize = 9;

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
	for(int y = 0; y < inputGray.rows; y++){
		for(int x = 0; x < inputGray.cols; x++){
			uchar r = inputGray.at<uchar>(y, x);
			uchar val = (uchar)(int)fmin(c * log10(r + 1), 255);
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
	
	//Lendo arquivo ROI
	string path(argv2);
	if(path != "null") {
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
		nullRoi = true;
		roi[0] = Mat::zeros(cv::Size(input.cols, input.rows), CV_8U);
		roi[1] = Mat::zeros(cv::Size(input.cols, input.rows), CV_8U);
		roi[2] = Mat::zeros(cv::Size(input.cols, input.rows), CV_8U);
		roi[3] = Mat::zeros(cv::Size(input.cols, input.rows), CV_8U);
		
		for(int x = 0; x < input.cols; x++)
			for(int y = 0; y < input.rows; y++)
				roi[0].at<uchar>(y, x) = 1;
	}
}

//Passando o Limiar na imagem resultante response
void thresholdR(Mat response){
	//Valor dentro da area externa
	int begX = 0, begY = 0; 
	int endX = response.cols, endY = response.rows;

	uchar thresholdValue = 50;
	
	for(int row = begY; row < endY-100; row++){
		for(int col = begX; col < endX-100; col++){
			uchar val = response.at<uchar>(row, col);
			
			if(val > thresholdValue){
				keyPoint.push_back(make_pair(row, col));
			}else response.at<uchar>(row, col) = 0;
		}
	}
	quantKeyPoints = (int)keyPoint.size();
}

//Conferindo se o valor a ser acessado está dentro dos limites da ROI
bool outOfBounds(int i, int j){
	return (i < 0 || j < 0 || i >= input.rows || j >= input.cols);
}

//Selecionando os Keypoinst com a Non Maxima supression
Mat nonMaximaSupression(Mat response){
	//Selecionando os vizinhos
	int maskSize = 21;
	//Criando vector auxiliar
	vector<pair<int, int> > aux;
	
	Mat responseCopy = Mat::zeros(cv::Size(input.cols, input.rows), CV_8U);
	
	for(int i = 0; i < (int)keyPoint.size(); i++){
		bool isMax = true;
		int y = keyPoint[i].first;
		int x = keyPoint[i].second;
		uchar mid = response.at<uchar>(y, x);
		
		int mSize2 = maskSize/2;
		for(int k = y - mSize2; k <= y + mSize2; k++){
			for(int j = x - mSize2; j <= x + mSize2; j++){
				if(!outOfBounds(k, j) && !(k == y && j == x)) {
					if(response.at<uchar>(k, j) >= mid){
						isMax = false;
						break;
					}
				}
			}
		}
		if(isMax){
			aux.push_back(make_pair(y, x));
			responseCopy.at<uchar>(y, x) = response.at<uchar>(y, x);
		}else{
			responseCopy.at<uchar>(y, x) = 0;
		}
	}
	//Passando o vector atualizado de volta 
	response = responseCopy;
	keyPoint.clear();
	keyPoint = aux;
	quantKeyPoints = keyPoint.size();
	
	return response;	
	
}//Fim função

// mascara coefficiente de variacao
Mat coefficienceOfVariationMaskGaussian(Mat aux, int n, string gaussianOp){
	if(aux.depth() != CV_32F)
		aux.convertTo(aux, CV_32F);
	
	Mat response = aux;
	
	Mat auxResponse = Mat::zeros(cv::Size(response.cols, response.rows), CV_32F);
	
	int N = n*n, cont = 0;//quantidade de pixels visitados
	
	float mediaGeral = 0; 
	
	Mat response2 = Mat::zeros(cv::Size(response.cols, response.rows), CV_64F);
	//response * response 
	for(int y = 0; y < response.rows; y++)
		for(int x = 0; x < response.cols; x++)
			response2.at<float>(y, x) = (response.at<float>(y, x) * response.at<float>(y, x));

	float gerador[] = {0.0, 0.0, 0.0, 0.0, 0.0};
	//Testando a gaussiana 5x5 na média
	if(gaussianOp == "1"){
		gerador[0] = 0.06136; gerador[1] = 0.24477; gerador[2] = 0.38774; gerador[3] = 0.24477; gerador[4] = 0.06136;//  - SD(sigma) = 1.0
	}
	else if(gaussianOp == "2"){
		gerador[0] = 0.122581; gerador[1] = 0.233062; gerador[2] = 0.288713; gerador[3] = gerador[1]; gerador[4] = gerador[0];// - SD(sigma) = 1.5 
	}
	else if(gaussianOp == "3"){
		gerador[0] = 0.153388; gerador[1] = 0.221461; gerador[2] = 0.250301; gerador[3] = gerador[1]; gerador[4] = gerador[0];// - SD(sigma) = 2.0
	}
	else if(gaussianOp == "4"){
		gerador[0] = 0.169327; gerador[1] = 0.214574; gerador[2] = 0.232198; gerador[3] = gerador[1]; gerador[4] = gerador[0];// SD(sigma); sigma = 2.5
	}
	
	Mat gaussianBox = Mat::zeros(cv::Size(n, n), CV_32F);
	Mat gen1 = cv::Mat(1, n, CV_32F, gerador);
	Mat gen2 = cv::Mat(n, 1, CV_32F, gerador);
	gaussianBox = gen2*gen1;
	
	int contVar = 0;
	
	float SUM = 0; // A soma de todos os valores da mascara gaussiana 5x5
	
	for(int R = 0; R < n; R++)
		for(int C = 0; C < n; C++)
			SUM += gaussianBox.at<float>(R, C);
		
	cout<<"S:"<< SUM <<endl;
	cout<<gaussianBox<<endl;
	
	//"Convolution"
	for(int i = (n/2)+1; i < response.rows - (n/2); i++){
		int yBeg = i-(n/2), yEnd = i+(n/2);
		for(int j = (n/2); j < response.cols - (n/2); j++){
			//passando mascara 
			float sumVal = 0, sumVal2 = 0, maior = 0, sumValG = 0;
			
			int xBeg = j-(n/2), xEnd = j+(n/2);
			
			for(int y = yBeg, I = 0; y <= yEnd; y++, I++){
				for(int x = xBeg, J = 0; x <= xEnd; x++, J++){
					sumValG += (response.at<float>(y, x) * gaussianBox.at<float>(I, J));
					sumVal += (response.at<float>(y, x));
					sumVal2 += (response2.at<float>(y, x) * gaussianBox.at<float>(I, J));
					//sumVal2 += (response2.at<float>(y, x));
				}
			}
			
			float media = sumVal/N;
			
			float variancia = (sumVal2 - (2*media*sumValG) + (sumValG*sumValG));
			//float variancia = (sumVal2/N) - (media*media);
			
			float S = sqrt(variancia); // desvio padrao
			
			//if (isnan(S)) contVar++;
			
			float CV = media == 0? 0 : S/media; // Coef de Variacao
			auxResponse.at<float>(i, j) = CV;
		}
	}
	
	//cout << contVar << " valPerc: " << (contVar*1.0)/(response.rows*response.cols) * 100 << endl;
	
	//Response recebe o valor de coef salvo em aux
	response = auxResponse;
	
	Mat aux2;
	normalize(response, aux2, 0.0, 256.0, NORM_MINMAX, CV_32FC1, Mat());
	
	return aux2;
}

Mat coefficienceOfVariationMask(Mat aux, int n){
	
	if(aux.depth() != CV_32F)
		aux.convertTo(aux, CV_32F);
	
	Mat response = aux;
	
	Mat auxResponse = Mat::zeros(cv::Size(response.cols, response.rows), CV_32F);
	
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
	normalize(response, aux2, 0.0, 256.0, NORM_MINMAX, CV_32FC1, Mat());
	
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

void saveKeypoints(Mat response){	
	printf("Salvando keypoints ROIs no arquivo...\n");
		
	vector<pair<int, pair<int, int> > > aux1, aux2, aux3;
	vector<pair<int, pair<int, int> > > aux;
	
    for(int i = 0; i < (int)keyPoint.size(); i++) {
		int y = keyPoint[i].first, x = keyPoint[i].second;
		aux.push_back({(int)response.at<uchar>(y, x), {y, x}});
	}	
    sort(aux.begin(), aux.end());
    reverse(aux.begin(), aux.end());
    
    int quantMaxKP = 500;
    
    //cout << (int)roi[0].at<uchar>(1012, 1012) << endl << (int)roi[0].at<uchar>(23, 10) << endl;
    
    for(int i = 0, k = 0; k < quantMaxKP && i < aux.size(); i++){
	 	int y = aux[i].second.first, x = aux[i].second.second;
	 	
	 	if((int)roi[1].at<uchar>(y, x) != 0 || nullRoi == true) {
			aux1.push_back({(int)response.at<uchar>(y, x), {y, x}}) ;
			k++;
		}
	 	else if((int)roi[2].at<uchar>(y, x) != 0) {
			aux2.push_back({(int)response.at<uchar>(y, x), {y, x}});
			k++;
		}
	 	else if((int)roi[3].at<uchar>(y, x) != 0) {
			aux3.push_back({(int)response.at<uchar>(y, x), {y, x}});
			k++;
		}	 	
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
		fprintf(out1, "%d %d %d\n", aux1[i].second.first, aux1[i].second.second, aux1[i].first);
	fclose(out1);
	//Salvando pontos ROI 2
	fprintf(out2, "%d\n", (int)aux2.size());
	for(int i = 0; i < (int)aux2.size(); i++)
		fprintf(out2, "%d %d %d\n", aux2[i].second.first, aux2[i].second.second, aux2[i].first);
	fclose(out2);
	//Salvando pontos ROI 3
	fprintf(out3, "%d\n", (int)aux3.size());
	for(int i = 0; i < (int)aux3.size(); i++)
		fprintf(out3, "%d %d %d\n", aux3[i].second.first, aux3[i].second.second, aux3[i].first);
	fclose(out3);
}

void saveKeypoints2ROIs(Mat response){
	// Quando tem somento duas ROIs, uma muito escura e outra muita clara, ou seja sem a região intermediária de tons médios ROI[2]
	printf("Salvando keypoints ROIs no arquivo...\n");
		
	vector<pair<int, pair<int, int> > > aux1, aux3;
	vector<pair<int, pair<int, int> > > aux;
	
    for(int i = 0; i < (int)keyPoint.size(); i++) {
		int y = keyPoint[i].first, x = keyPoint[i].second;
		aux.push_back({(int)response.at<uchar>(y, x), {y, x}});
	}	
    sort(aux.begin(), aux.end());
    reverse(aux.begin(), aux.end());
    
    int quantMaxKP = 500;
    
    //cout << (int)roi[0].at<uchar>(1012, 1012) << endl << (int)roi[0].at<uchar>(23, 10) << endl;
    
    for(int i = 0, k = 0; k < quantMaxKP && i < aux.size(); i++){
	 	int y = aux[i].second.first, x = aux[i].second.second;
	 	
	 	if((int)roi[0].at<uchar>(y, x) == 0)
			continue;
	 	
	 	if((int)roi[1].at<uchar>(y, x) != 0 || nullRoi == true) {
			aux1.push_back({(int)response.at<uchar>(y, x), {y, x}}) ;
			k++;
		}
	 	else if((int)roi[3].at<uchar>(y, x) != 0) {
			aux3.push_back({(int)response.at<uchar>(y, x), {y, x}});
			k++;
		}	 	
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
		fprintf(out1, "%d %d %d\n", aux1[i].second.first, aux1[i].second.second, aux1[i].first);
	fclose(out1);
	
	//Salvando pontos ROI 3
	fprintf(out3, "%d\n", (int)aux3.size());
	for(int i = 0; i < (int)aux3.size(); i++)
		fprintf(out3, "%d %d %d\n", aux3[i].second.first, aux3[i].second.second, aux3[i].first);
	fclose(out3);
}

//edgeThreshold
void edgeThreshold(Mat responseImg){
	Mat Ix, Iy, Ix2, Iy2, Ixy;
	
	//Computando sobel operator (derivada da gaussiana) no eixo x e y
	Sobel(responseImg, Ix, CV_32F, 1, 0, 7, 1, 0, BORDER_DEFAULT);
	Sobel(responseImg, Iy, CV_32F, 0, 1, 7, 1, 0, BORDER_DEFAULT);
	
	Ix2 = Ix.mul(Ix); // Ix^2
	Iy2 = Iy.mul(Iy);// Iy^2
	Ixy = Ix.mul(Iy);// Ix * Iy
	
	int rows = responseImg.rows, cols = responseImg.cols;
	
	Mat response = Mat::zeros(cv::Size(cols, rows), CV_32F);
	for(int row = 0; row < rows; row++){
		for(int col = 0; col < cols; col++){
			float fx2 = Ix2.at<float>(row, col);
			float fy2 = Iy2.at<float>(row, col);
			float fxy = Ixy.at<float>(row, col);
			float det = (fx2 * fy2) - (fxy * fxy);
			float trace = (fx2 + fy2);
			response.at<float>(row, col) = det - 0.04*(trace*trace);
			//if(response.at<float>(row, col) < 0) response.at<float>(row, col) = 0;
			//printf("%f", response.at<float>(row, col));
		}
	}
	
	for(int row = 0; row < rows; row++){
		for(int col = 0; col < cols; col++){
			float val = response.at<float>(row, col);
			if(val < -1e4){
				response.at<float>(row, col) = 0;
			}
		}	
	}
	
	vector<pair<int, int> > auxKp;
	
	for(int i = 0; i < (int)keyPoint.size(); i++){
		int y = keyPoint[i].first, x = keyPoint[i].second;
		if(response.at<float>(y, x) != 0)
			auxKp.push_back(keyPoint[i]);
	}
	keyPoint = auxKp;
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
	
	saida1 += ".detectorCV.distribution.txt";
	saida2 += ".detectorCV1.txt";
	saida3 += ".detectorCV2.txt";
	saida4 += ".detectorCV3.txt";
	
	out0 = fopen(saida1.c_str(), "w+");
	out1 = fopen(saida2.c_str(), "w+");
	out2 = fopen(saida3.c_str(), "w+");
	out3 = fopen(saida4.c_str(), "w+");
	
	read(argv[1], argv[2]);
	
	//inputGray = coefficienceOfVariationMask(inputGray, 5);
	
	string gaussianOp(argv[3]);
	
	inputGray = coefficienceOfVariationMaskGaussian(inputGray, 5, gaussianOp);
	
	normalize(inputGray, inputGray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	
	inputGray = inputGray.mul(25);
	//equalizeHist(inputGray, inputGray);
	//logTranformUchar(150);
	
	//imwrite("response.png", inputGray);
	GaussianBlur(inputGray, inputGray, Size(9, 9), 0, 0, BORDER_DEFAULT);
	//Mat ans; bilateralFilter(inputGray, ans, 10, 175, 175, BORDER_DEFAULT); inputGray = ans;
	
	//imwrite("response.png", inputGray);
	
	//Threshoulding image 
	thresholdR(inputGray);
	
	//imwrite("responseThres.png", inputGray);
	
	//Fazendo NonMaximaSupression nos keypoints encontrados
	Mat responseImg = nonMaximaSupression(inputGray);
	
	//edgeThreshold(responseImg);
	
	printf("quantidade KeyPoints: %d\n", quantKeyPoints);
	//Salvando quantidade de Keypoints e para cada KP as coordenadas (x, y) e o response
	
	//saveKeypoints2ROIs(responseImg);
	saveKeypoints(responseImg);
	
	//Salvando keypoints na imagem de entrada 
	showKeyPoints();
	
	/*
	//Aumentando brilho da imagem
	for(int y = 0; y < input.rows; y++){
		for(int x = 0; x < input.cols; x++){
			input.at<Vec3f>(y, x)[0] *= 100;
			input.at<Vec3f>(y, x)[1] *= 100;
			input.at<Vec3f>(y, x)[2] *= 100;	
		}
	}
	*/
	
	//Salvando imagens com Keypoints
	int len = strlen(saida);
	saida[len] = 'R';saida[len+1] = '.';saida[len+2] = 'j';saida[len+3] = 'p';saida[len+4] = 'g';saida[len+5] = '\0';
	
	imwrite(saida, input);
	
    return 0;
}
