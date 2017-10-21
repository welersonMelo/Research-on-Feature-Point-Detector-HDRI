//g++ -ggdb `pkg-config --cflags opencv` -o `basename repeatability.cpp .cpp` repeatability.cpp `pkg-config --libs opencv`

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//Incluindo todas as bibliotecas std do c/c++
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

const int INF = (int) 1e9;
FILE *inMat, *inKP;
//Matriz de Homografia, vetor base, vetor resultado
Mat homografia, pointBase, resultPoint;

pair<int, int> point;
vector<pair<float, pair<int, int> > > keyPointB;
vector<pair<float, pair<int, int> > > keyPointS;

int quantPositiveKP; //quantidade de KeyPoints que é encontrado em ambas imagens 

void readMatriz(){
	float value;
	homografia = Mat::zeros(cv::Size(3, 3), CV_32F);
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			fscanf(inMat, "%f", &value);
			homografia.at<float>(i, j) = value;
		}
	}
}

//Lendo keyPoints do arquivo de texto 
//Se val = 1 salva os pontos base, se for igual a 2 salva os pontos de saida
void readKeyPoints(int val){
	int x, y, n;
	float response; //Por enquanto n serve pra nada
	fscanf(inKP, "%d", &n);
	
	for(int i = 0; i < n; i++){
		fscanf(inKP, "%d %d %f", &y, &x, &response);
		if(val == 1)
			keyPointB.push_back(make_pair(-response, make_pair(x, y)));
		else if(val == 2)
			keyPointS.push_back(make_pair(-response, make_pair(x, y)));
	}
}

//Função que faz a transformação dos keypoints da imagem base
void transformacao(){
	int siz = (int)keyPointB.size();
	for(int i = 0; i < siz; i++){
		pointBase.at<float>(0, 0) = keyPointB[i].second.first;
		pointBase.at<float>(1, 0) = keyPointB[i].second.second;
		pointBase.at<float>(2, 0) = 1;
		
		//Calculando multiplicação entre matrizes
		resultPoint = homografia * pointBase;
		
		//Pegando as novas coordenadas do pontoBase
		float aux1 = resultPoint.at<float>(0, 0) / resultPoint.at<float>(2, 0);
		float aux2 = resultPoint.at<float>(1, 0) / resultPoint.at<float>(2, 0);
		point.first = round(aux1); //x
		point.second = round(aux2); //y
		
		keyPointB[i].second = point;
		
		//cout<<point.first<<" "<<point.second<<"\n";
	}
}

//Função para calcular o Repeatability Rate retornando a quantidade de keypoins que deu Positivo (encontrado na outra imagem)
void calculandoRR(int quantK){
	//Ordenando pelo maior response para selecionar os 300 maiores para comparação
	sort(keyPointB.begin(), keyPointB.end());
	sort(keyPointS.begin(), keyPointS.end());
	
	for(int i = 0; i < quantK; i++){
		int x = keyPointB[i].second.first;
		int y = keyPointB[i].second.second;
		for(int j = 0; j < quantK; j++){
			int x1 = keyPointS[j].second.first;
			int y1 = keyPointS[j].second.second;
			// Calculando a distância em pixel entre os dois pontos
			int dist = max(abs(x - x1) , abs(y - y1)); 
			
			//Se estiver dentro do raio de 9 pixels
			if(dist < 10){
				quantPositiveKP++;
				break;
			}
		}
	}	
}

//Função Principal
//Chamada: ./repeatability MatrizDaTransformacao(Homografia) keyPointsBase keyPointsSaida
int main(int, char** argv ){
	inMat = fopen(argv[1], "r");
	readMatriz();
	
	inKP = fopen(argv[2], "r");
	readKeyPoints(1);
	
	inKP = fopen(argv[3], "r");
	readKeyPoints(2);
	
	//Inicializando Matriz coordenada
	pointBase = Mat::zeros(cv::Size(1, 3), CV_32F);
	resultPoint = Mat::zeros(cv::Size(1, 3), CV_32F);
	
	//Fazendo a transformação para todos os pontos da base
	transformacao();
	
	//Setando a quantidade max de keypoints para ser avaliado
	int quantK = min(keyPointB.size(), keyPointS.size()) < 300 ? min(keyPointB.size(), keyPointS.size()) : 300;
	calculandoRR(quantK);	
	
	printf("Encontrados %d kp de %d. RR = %.8f\n", quantPositiveKP, quantK, (double)quantPositiveKP/(double)quantK);
	
	return 0;
}
