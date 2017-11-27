//g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename sift.cpp .cpp` sift.cpp `pkg-config --libs opencv`

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//Incluindo todas as bibliotecas std do c/c++
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

const int INF = (int) 1e9;

//Criando imagens do tipo Mat
FILE *in, *out;

Mat input, inputGray, inputIm[5], octave[4][5], dogI[4][4], response;

bool isHDR = false;

int gaussianSize = 11;

struct KeyPoints{
	int x, y, scale, z;//posicao (x, y) e o octave ou escalada da imagem
	float response;
};

vector<KeyPoints> keyPoint;
vector<pair<int, int> > ROI;
vector<pair<int, int> > quadROIo;
vector<pair<int, int> > quadROIi;


//Pegando maior valor numa imagem cinza
float getMaxValue1(Mat src1){
  float maior = -INF;
  for(int row = 0; row < src1.rows; row++){
	for(int col = 0; col < src1.cols; col++){
		maior = max(maior, src1.at<float>(row, col));
	}
  }
  return maior;
}

uchar getMaxValue2(Mat src1){
  uchar maior = 0;
  for(int row = 0; row < src1.rows; row++){
	for(int col = 0; col < src1.cols; col++){
		maior = max(maior, src1.at<uchar>(row, col));
	}
  }
  return maior;
}
	
//Abrindo imagem no argumento da linha de comando
void read(char *name){
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
	if(in != NULL){
		float x, y;
		while(fscanf(in, "%f%f", &x, &y) != EOF){
			ROI.push_back(make_pair(x, y));
		}
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

//Função que inicializa os octaves com a mudança de escala e blur para 4 octaves 5 imagens cada (Valor padrão)
void initOctaves(){	
	Mat auxImg;
	inputGray.convertTo(auxImg, CV_32FC1);
	
	float k[] = {0.707107, 1.414214, 2.828428, 5.656856};
	
	for(int c = 0; c < 4; c++){ // escala
		float ko = k[c];
		for(int j = 0; j < 5; j++){
			GaussianBlur(auxImg, octave[c][j], Size(gaussianSize,gaussianSize), ko, ko, BORDER_DEFAULT);
			ko = ko * 1.414213562; // = ko*sqrt(2);
		}
		//resampling inputImage
		resize(auxImg, auxImg, Size(auxImg.cols/2, auxImg.rows/2));
	}
}

//Função que calula a diferença entre as gaussianas DoG e salva em uma nova imagem
void calcDoG(){
	for(int i = 0; i < 1; i++){
		for(int j = 0; j < 4; j++){
			dogI[i][j] = Mat::zeros(Size(octave[i][j].cols, octave[i][j].rows), CV_32F);
			for(int col = 0; col < octave[i][j].cols; col++){
				for(int row = 0; row < octave[i][j].rows; row++){
					float a = octave[i][j].at<float>(row, col) - octave[i][j+1].at<float>(row, col);
					dogI[i][j].at<float>(row, col) = a;
				}
			}
		}
	}
}

//Conferindo se o valor a ser acessado está dentro dos limites da imagem
bool outOfBounds(int i, int j, Mat input){
	return (i < 0 || j < 0 || i >= input.rows || j >= input.cols);
}

//Selecionando os Keypoinst com a Non Maxima supression 
void nonMaximaSupression(){
	//Selecionando os vizinhos
	int maskSize = 21; // Mascara de 21 x 21 baseado no artigo do prybil
	int cont = 0;
	
	for(int c = 0; c < 1; c++){
		for(int z = 1; z < 3; z++){
			Mat matAux = Mat::zeros(Size(dogI[c][z].cols, dogI[c][z].rows), CV_32F);
			for(int x = 0; x < dogI[c][z].cols; x++){
				for(int y = 0; y < dogI[c][z].rows; y++){
					float mid = dogI[c][z].at<float>(y, x);	
					bool cond1 = true, cond2 = true;
					int mSize2 = maskSize/2;
					
					for(int i = y - mSize2; i <= y + mSize2; i++){
						for(int j = x - mSize2; j <= x + mSize2; j++){
							if(!outOfBounds(i, j, dogI[c][z])){
								if(!((mid < dogI[c][z].at<float>(i, j) || (y == i && x == j)) && 
								   mid < dogI[c][z-1].at<float>(i, j) && 
								   mid < dogI[c][z+1].at<float>(i, j))
								  )
								 {
									 cond1 = false;
									 break;
								 }
							}
						}
						if(!cond1)break;
					}
					for(int i = y - mSize2; i <= y + mSize2; i++){
						for(int j = x - mSize2; j <= x + mSize2; j++){
							if(!outOfBounds(i, j, dogI[c][z])){
								if(!((mid > dogI[c][z].at<float>(i, j) || (y == i && x == j)) && 
								   mid > dogI[c][z-1].at<float>(i, j) && 
								   mid > dogI[c][z+1].at<float>(i, j))
								  )
								 {
									 cond2 = false;
									 break;
								 }
							}
						}
						if(!cond2)break;
					}
					if(cond1 || cond2){
						matAux.at<float>(y, x) = dogI[c][z].at<float>(y, x);
						cont++;
					}else matAux.at<float>(y, x) = 0;
				}
			}
			dogI[c][z] = matAux;
		}
		cout<<cont<<endl;//Mostrando quantidade de keypoints encontrado na imagem i j
		cont = 0;
	}
}//Fim função

//Passando o Limiar na imagem resultante response
void threshold(){
	//Atualizando threshold
	float thresholdValue = 8; // Threshold fixo para teste do pribyl
	int cont = 0;
	int begX = quadROIo[0].first, begY = quadROIo[0].second; 
	int endX = quadROIo[2].first, endY = quadROIo[2].second;
	int quadIYb = quadROIi[0].second, quadIYe = quadROIi[2].second;
	int quadIXb = quadROIi[0].first, quadIXe = quadROIi[2].first;
	int error = 0.014981273*(endY - begY);
	
	begY = begY + error;//Erro no limite superior do ROI
	
	for(int c = 0; c < 1; c++){ //scale/octave
		for(int z = 1; z < 3; z++){
			for(int y = begY; y < endY-10; y++){
				for(int x = begX; x < endX-10; x++){
					if(y > quadIYb && y < quadIYe && x > quadIXb && x < quadIXe) //verificando se esta dentro do quadrado menor
						continue;
						
					float val = fabs(dogI[c][z].at<float>(y, x));
					if(val > thresholdValue){
						keyPoint.push_back({x, y, c, z,val});
						cont++;
					}else{
						 dogI[c][z].at<float>(y, x) = 0;	
						 if(val != 0)
							circle(input, Point (x, y), 5, Scalar(255, 0, 0), 1, 8, 0);
					}
				}
			}
			//string s = to_string(c) + to_string(z);
			//imshow(s, dogI[c][z]); waitKey(0);
		}
		cout<<"t:"<<cont<<endl;
		cont = 0;
			
		begX /= 2; endX /= 2;
		begY /= 2; endY /=2;
		error /= 2;
		quadIXb /= 2; quadIXe /= 2;
		quadIYb /= 2; quadIYe /= 2;
	}
}

void showKeyPoints(){
	for(int i = 0; i < (int)keyPoint.size(); i++){
		int x = keyPoint[i].x;
		int y = keyPoint[i].y;

		circle(input, Point (x, y), 4, Scalar(0, 0, 255), 1, 8, 0);
	}
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
//Chamada: ./sift Imagem ArquivoROIPoints
int main(int, char** argv ){
	char saida[255];
	strcpy(saida, argv[1]);
	saida[strlen(saida)-4] = '\0';
	
	in = fopen(argv[2], "r");
	out = fopen(saida, "w+");
	
	read(argv[1]);
	
	//Inicializando octaves
	initOctaves();
	
	//Calculando DoG
	calcDoG();

	//Fazendo NonMaximaSupression na imagem DoG
	nonMaximaSupression();
	
	//Limiar na imagem de Response
	threshold();
	
	//Limiar p/ edges response
	//edgeThreshold(); 
	
	//Salvando quantidade de Keypoints e para cada KP as coordenadas (x, y) e o response
	printf("Salvando keypoints no arquivo...\n");
	fprintf(out, "%d\n", (int)keyPoint.size());
	for(int i = 0; i < (int)keyPoint.size(); i++){
		fprintf(out, "%d %d %.2f\n", keyPoint[i].x , keyPoint[i].y, keyPoint[i].response);
	}	
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
