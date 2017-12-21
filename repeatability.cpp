//g++ -ggdb `pkg-config --cflags opencv` -o `basename repeatability.cpp .cpp` repeatability.cpp `pkg-config --libs opencv`

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//Incluindo todas as bibliotecas std do c/c++
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

const int INF = (int) 1e9;

struct Line{
	pair<int, int> p1, p2;
};

vector<struct Line> linha;

FILE *inMat, *inKP;
//Matriz de Homografia, vetor base, vetor resultado
Mat homografia, pointBase, resultPoint;

vector<pair<float, pair<int, int> > > keyPointB, keyPointB2;
vector<pair<float, pair<int, int> > > keyPointS;

float sumRR = 0.0;
int quantMaxKP = 300;
int quantPositiveKP; //quantidade de KeyPoints que é encontrado em ambas imagens 

void readMatriz(string argv){
	// Abrindo arquivos
	ifstream in(argv);
    streambuf *cinbuf = std::cin.rdbuf();
    cin.rdbuf(in.rdbuf());
    
	float value;
	homografia = Mat::zeros(cv::Size(3, 3), CV_32F);
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			cin>>value;
			homografia.at<float>(i, j) = value;
		}
	}
	in.close();
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
	keyPointB2 = keyPointB;
	
	for(int i = 0; i < siz && i < quantMaxKP; i++){
		pointBase.at<float>(0, 0) = keyPointB[i].second.first;
		pointBase.at<float>(1, 0) = keyPointB[i].second.second;
		pointBase.at<float>(2, 0) = 1;
		
		//Calculando multiplicação entre matrizes
		resultPoint = homografia * pointBase;
		
		//Pegando as novas coordenadas do pontoBase
		float aux1 = resultPoint.at<float>(0, 0) / resultPoint.at<float>(2, 0);
		float aux2 = resultPoint.at<float>(1, 0) / resultPoint.at<float>(2, 0);
		
		pair<int, int> point;
		
		point.first = round(aux1); //x
		point.second = round(aux2); //y
		
		keyPointB[i].second = point;
	}
}

//Função para calcular o Repeatability Rate retornando a quantidade de keypoins que deu Positivo (encontrado na outra imagem)
void calculandoRR(int quantK){
	
	for(int i = 0; i < quantK; i++){
		int x = keyPointB[i].second.first;
		int y = keyPointB[i].second.second;
		for(int j = 0; j < quantK; j++){
			int x1 = keyPointS[j].second.first;
			int y1 = keyPointS[j].second.second;
			// Calculando a distância em pixel entre os dois pontos
			int dist = max(abs(x - x1) , abs(y - y1)); 
			
			//Se estiver dentro do raio de X pixels
			if(dist < 11){ 
				struct Line linhaAux;
				linhaAux.p1 = make_pair(keyPointB2[i].second.first, keyPointB2[i].second.second);
				linhaAux.p2 = make_pair(x1, y1);
				
				linha.push_back(linhaAux);
				
				quantPositiveKP++;
				break;
			}
		}
	}	
}

void showPointsCorrelation(char *img1, char *img2){
	Mat im1, im2; //imagens com keypoints correlatos
	string aux1 = (string)img1;
	string aux2 = (string)img2;
	aux1 = "../dataset/2D/distance/100/100.gLarson97R.jpg";
	aux2 = "../dataset/2D/distance/103/103.gLarson97R.jpg";
	
	im1 = imread(aux1, IMREAD_UNCHANGED);
	im2 = imread(aux2, IMREAD_UNCHANGED);
	
	Size sz1 = im1.size();
    Size sz2 = im2.size();
    
    //im3 = Junção das duas imagens lado a lado
    Mat im3(sz1.height, sz1.width+sz2.width, CV_8UC3);
    Mat left(im3, Rect(0, 0, sz1.width, sz1.height));    
    im1.copyTo(left);
    Mat right(im3, Rect(sz1.width, 0, sz2.width, sz2.height));
    im2.copyTo(right);
    
    //Desenhando linhas entre keypoints
    for(int i = 0; i < (int)linha.size(); i++){
		int x = linha[i].p1.first;
		int y = linha[i].p1.second;
		int x1 = linha[i].p2.first + im1.cols;
		int y1 = linha[i].p2.second;
		
		line(im3, Point (x, y), Point (x1 , y1), Scalar(0, 0, 255), 1, 8, 0);
	}
	
	//resize(im3, im3, Size(), 0.2, 0.2, CV_INTER_LINEAR);
    
    imwrite("im.jpg", im3);
}

//salvando rr no arquivo
void saveFile(string txt1, string txt2){
	
}

int lookFor(string x, string str){
	//cout<<str<<"|"<<x<<endl;
	for(int i = 0; i < (int) str.size(); i++){
		if(str[i] == x[0]){
			if(str[i+1] == x[1])
				return i;
		}
	}
	return -1;
}

void execute(char** argv, string base, string saida){
	
	//cout<<"Executando h "<<base<<saida<<endl;
	
	string txt0(argv[1]), txt1(argv[2]), txt2(argv[3]);
	
	//../dataset/2D/distance/H.BX.SX.txt ../dataset/2D/distance/BX/BX.gLarson97.harris1.txt ../dataset/2D/distance/SX/SX.gLarson97.harris1.txt
	
	//processando string base e saida
	int posi;
	//para o arquivo H
	posi = lookFor("BX", txt0);
	txt0.replace(posi, 2, base);
	posi = lookFor("SX", txt0);
	txt0.replace(posi, 2, saida);
	
	//para os arquivos com os KP
	posi = lookFor("BX", txt1);
	txt1.replace(posi, 2, base);
	posi = lookFor("BX", txt1);
	txt1.replace(posi, 2, base);
	
	posi = lookFor("SX", txt2);
	txt2.replace(posi, 2, saida);
	posi = lookFor("SX", txt2);
	txt2.replace(posi, 2, saida);

	//cout<<txt0<<" "<<txt1<<" "<<txt2<<endl;
	
	readMatriz(txt0);
	
	inKP = fopen(txt1.c_str(), "r");
	readKeyPoints(1);
	
	inKP = fopen(txt2.c_str(), "r");
	readKeyPoints(2);
	
	//Inicializando Matriz coordenada
	pointBase = Mat::zeros(cv::Size(1, 3), CV_32F);
	resultPoint = Mat::zeros(cv::Size(1, 3), CV_32F);	
	
	//Ordenando pelo maior response para selecionar os x maiores para comparação
	sort(keyPointB.begin(), keyPointB.end());
	sort(keyPointB2.begin(), keyPointB2.end());
	sort(keyPointS.begin(), keyPointS.end());
	
	//Fazendo a transformação para todos os pontos da base
	transformacao();
	
	//Setando a quantidade max de keypoints para ser avaliado
	int quantK = min(keyPointB.size(), keyPointS.size()) < quantMaxKP ? min(keyPointB.size(), keyPointS.size()) : quantMaxKP;
	
	if(quantK != 0)
		calculandoRR(quantK);	
	else quantK = 1;
	
	float RR = (double)quantPositiveKP/(double)quantK;
	
	//salvando resultado no arquivo
	saveFile(txt1, txt2);
	
	sumRR += RR;
	printf("%.8f\n", RR);
	
	//showPointsCorrelation(argv[2], argv[3]);
}

//Função Principal
//Chamada: ./repeatability MatrizDaTransformacao(Homografia) keyPointsBase keyPointsSaida
int main(int, char** argv ){
	
	//Executando rr para distancia
	string distance[] = {"100", "103", "109", "122", "147", "197", "297"};
	int cont = 0;
	for(int i = 0; i < 7; i++){
		for(int j = 0; j < 7; j++){
			if(i != j){
				cont++;
				quantPositiveKP = 0;
				keyPointB.clear();
				keyPointB2.clear();
				keyPointS.clear();
				
				execute(argv, distance[i], distance[j]);
			}
		}
	}
	
	printf("Media: %.8f\n", sumRR/cont);
	
	//executanto rr para ...
	
	return 0;
}
