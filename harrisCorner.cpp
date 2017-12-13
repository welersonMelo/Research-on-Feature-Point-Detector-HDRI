//g++ -ggdb `pkg-config --cflags opencv` -o `basename harrisCorner.cpp .cpp` harrisCorner.cpp `pkg-config --libs opencv`

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
FILE *in, *out;

Mat input, inputGray, Ix, Iy, Ix2, Iy2, Ixy, response;

bool isHDR = false;

int quantKeyPoints = 0;
int gaussianSize = 9;

float thresholdValue = 0;

vector<pair<int, int> > keyPoint;
vector<pair<int, int> > ROI;
vector<pair<int, int> > quadROIo;
vector<pair<int, int> > quadROIi;

//Transformaçao Log para a LOG.HDR
void logTranform(int c){
	for(int y = 0; y < input.rows; y++){
		for(int x = 0; x < input.cols; x++){
			float r = inputGray.at<float>(y, x);
			float val = c * log10(r + 1);
			inputGray.at<float>(y, x) = val;
		}
	}
}

//Abrindo imagem no argumento da linha de comando
void read(char *name, char *argv2){
	input = imread(name, IMREAD_UNCHANGED);
	//Gerando imagem grayscale
	
	if(input.channels() != 1)
		cvtColor(input, inputGray, COLOR_BGR2GRAY);
	else{
		inputGray = input;
		cvtColor(input, input, COLOR_GRAY2BGR);
	}
	
	//Conferindo se é HDR
	if(input.depth() == CV_32F) {
		isHDR = true;
		normalize(inputGray, inputGray, 0.0, 256.0, NORM_MINMAX, CV_32FC1, Mat());
		printf("Imagem HDR\n");
	}else isHDR = false;
	
	//Lendo arquivo com os pontos (x, y) da ROI
	if(argv2 != NULL){
		printf("Lendo ROI.txt\n");
		// Abrindo arquivos
		ifstream in(argv2);
		streambuf *cinbuf = std::cin.rdbuf();
		cin.rdbuf(in.rdbuf());
		
		float x, y;
		while(cin>>x>>y){
			ROI.push_back(make_pair(x, y));
		}
		
		in.close();
		
		ROI.push_back(ROI[0]);
		
		//Quadrado externo
		quadROIo.push_back(ROI[0]);
		quadROIo.push_back(ROI[1]);
		quadROIo.push_back(make_pair(ROI[5].first, ROI[1].second));
		quadROIo.push_back(ROI[5]);
		//Quadrado Interno
		quadROIi.push_back(ROI[3]);
		quadROIi.push_back(ROI[2]);
		quadROIi.push_back(make_pair(ROI[5].first, ROI[1].second));
		quadROIi.push_back(ROI[4]);
	}else{
		//Imagem completa (normal)
		ROI.push_back(make_pair(0, 0));
		ROI.push_back(make_pair(input.cols, 0));
		ROI.push_back(make_pair(input.cols, input.rows));
		ROI.push_back(make_pair(0, input.rows));
		ROI.push_back(make_pair(0, 0));
		
		//Quadrado externo
		quadROIo.push_back(make_pair(0, 0));
		quadROIo.push_back(make_pair(input.cols, 0));
		quadROIo.push_back(make_pair(input.cols, input.rows));
		quadROIo.push_back(make_pair(0, input.rows));
		//Quadrado interno
		quadROIi.push_back(make_pair(0, 0));
		quadROIi.push_back(make_pair(1, 0));
		quadROIi.push_back(make_pair(1, 1));
		quadROIi.push_back(make_pair(0, 1));
	}
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
		}
	}
}

//Passando o Limiar na imagem resultante response
void thresholdR(){
	//Atualizando threshold
	thresholdValue = (1e15); // Threshold fixo para teste do pribyl
	
	//Valor dentro da area externa
	int begX = quadROIo[0].first, begY = quadROIo[0].second; 
	int endX = quadROIo[2].first, endY = quadROIo[2].second;
	int error = 0.014981273*(endY - begY);
	begY = begY + error;//Erro no limite superior do ROI
	
	for(int row = begY; row < endY; row++){
		for(int col = begX; col < endX; col++){
			if(row > quadROIi[0].second && row < quadROIi[2].second && col > quadROIi[0].first && col < quadROIi[2].first) //verificando se esta dentro do quadrado menor
				continue;
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

void showKeyPoints(){
	for(int i = 0; i < (int)keyPoint.size(); i++){
		int x = keyPoint[i].first;
		int y = keyPoint[i].second;
		//printf("%d %d\n", x, y);
		circle(input, Point (y, x), 3, Scalar(0, 0, 255), 1, 8, 0);	
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
	
	imshow(name, imResponse);
	waitKey(0);
}

//Função para marcar na imagem final a ROI
void showROI(){
	float x, y, x1, y1;
	x = ROI[0].first;
	y = ROI[0].second;
	for(int i = 1; i < (int)ROI.size(); i++){
		x1 = ROI[i].first;
		y1 = ROI[i].second;
		line(input, Point (x, y), Point (x1, y1), Scalar(0, 0, 255), 5, 8, 0);
		x = x1;
		y = y1;
	}
}

//Função Principal
// ROI = Region Of Interest
//Chamada: ./harrisCorner Imagem ArquivoROIPoints
int main(int, char** argv ){
	char saida[255];
	strcpy(saida, argv[1]);
	saida[strlen(saida)-4] = '\0';
	string saida2(saida);
	
	saida2 += ".harris.txt";
		
	out = fopen(saida2.c_str(), "w+");
		
	read(argv[1], argv[2]);
	
	//Inicalizando com a gaussiana
	GaussianBlur(inputGray, inputGray, Size(gaussianSize,gaussianSize), 0, 0, BORDER_DEFAULT);
	
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
	//showResponse("Antes Th");	
	
	//Limiar na imagem de Response
	thresholdR();
	
	normalize(response, response, 0.0, 1000.0, NORM_MINMAX, CV_32FC1, Mat());
	
	//showResponse("Depois Th");
	
	printf("quantidade de KeyPoints depois threshold: %d\n", quantKeyPoints);
	
	//Fazendo NonMaximaSupression nos keypoints encontrados
	nonMaximaSupression();
	
	//showResponse("Depois nonMax Th");
	printf("quantidade final KeyPoints: %d\n", quantKeyPoints);
	
	//Salvando quantidade de Keypoints e para cada KP as coordenadas (x, y) e o response
	printf("Salvando keypoints no arquivo...\n");
	fprintf(out, "%d\n", quantKeyPoints);
	for(int i = 0; i < (int)keyPoint.size(); i++)
		fprintf(out, "%d %d %.2f\n", keyPoint[i].first, keyPoint[i].second, response.at<float>(keyPoint[i].first, keyPoint[i].second));
	
	fclose(out);
	
	showKeyPoints();
	showROI();
	
	//Salvando imagens com Keypoints
	int len = strlen(saida);
	saida[len] = 'R';saida[len+1] = '.';saida[len+2] = 'j';saida[len+3] = 'p';saida[len+4] = 'g';saida[len+5] = '\0';
	imwrite(saida, input);
	
	//Mostrando imagem com Keypoints
	//imshow("Result", input);
	//waitKey(0);
	
	return 0;
}
