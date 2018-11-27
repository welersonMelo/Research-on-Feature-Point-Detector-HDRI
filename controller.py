import time
import os

num = "1234"

finalResultHDR = []
finalResultlogHDR = []

for gaussianOp in num:
	files = ["executar.detector_CV.distance1", "executar.detector_CV.distance2", "executar.detector_CV.lighting1", "executar.detector_CV.lighting2", "executar.detector_CV.viewpoint1", "executar.detector_CV.viewpoint2", "executar.detector_CV.viewpoint3", "executar.detector_CV.viewpoint4"]
	x = []
	x.append(open(files[0]+'.copy').read())
	x.append(open(files[1]+'.copy').read())
	x.append(open(files[2]+'.copy').read())
	x.append(open(files[3]+'.copy').read())
	x.append(open(files[4]+'.copy').read())
	x.append(open(files[5]+'.copy').read())
	x.append(open(files[6]+'.copy').read())
	x.append(open(files[7]+'.copy').read())

	for i in range(len(x)):
		s = x[i].replace('\n', ' '+gaussianOp+' \n')
		f = open(files[i], "w")
		f.write(s)
		f.close()

	print('iniciando execucoes')

	print (os.system("bash executar.ALL"))

	print('entrando em espera')
	time.sleep(60*14)

	print('fim execucao\nRodar repetibilidade')

	print(os.system('bash executar.detector_CV.distribution_and_RR '))

	print ('entrando na pasta dataset/2D')

	print ('Pegando dados gerados e fazendo media')

	volta = '../dataset/2D/'

	df1 = open(volta+'distance/Dist/distribution.HDR.detectorCV.distribution.txt')
	df2 = open(volta+'lighting/Dist/distribution.HDR.detectorCV.distribution.txt')
	df3 = open(volta+'viewpoint/Dist/distribution.HDR.detectorCV.distribution.txt')

	dv1 = float(df1.read())
	dv2 = float(df2.read())
	dv3 = float(df3.read())

	DmediaHDR = (dv1+dv2+dv3) / 3.0

	rf1 = open(volta+'distance/RR/RR.HDR.detectorCV1.txt')
	rf2 = open(volta+'lighting/RR/RR.HDR.detectorCV1.txt')
	rf3 = open(volta+'viewpoint/RR/RR.HDR.detectorCV1.txt')

	rv1 = float(rf1.read()[2:])
	rv2 = float(rf2.read()[2:])
	rv3 = float(rf3.read()[2:])

	RmediaHDR = (rv1+rv2+rv3) / 3.0

	
	finalResultHDR.append('(' + str (DmediaHDR) + ' ; '+ str(RmediaHDR) +')')

	df1 = open(volta+'distance/Dist/distribution.logHDR.detectorCV.distribution.txt')
	df2 = open(volta+'lighting/Dist/distribution.logHDR.detectorCV.distribution.txt')
	df3 = open(volta+'viewpoint/Dist/distribution.logHDR.detectorCV.distribution.txt')

	dv1 = float(df1.read())
	dv2 = float(df2.read())
	dv3 = float(df3.read())

	DmediaHDR = (dv1+dv2+dv3) / 3.0

	rf1 = open(volta+'distance/RR/RR.logHDR.detectorCV1.txt')
	rf2 = open(volta+'lighting/RR/RR.logHDR.detectorCV1.txt')
	rf3 = open(volta+'viewpoint/RR/RR.logHDR.detectorCV1.txt')

	rv1 = float(rf1.read()[2:])
	rv2 = float(rf2.read()[2:])
	rv3 = float(rf3.read()[2:])

	RmediaHDR = (rv1+rv2+rv3) / 3.0

	finalResultlogHDR.append('(' + str (DmediaHDR) + ' ; ' + str (RmediaHDR) + ')')


resultsF = open('finalResults', 'w')
resultsF.write(str(finalResultHDR) + '\n' + str(finalResultlogHDR))
resultsF.close()

print ('resultados HDR')
for k in finalResultHDR:
	print (k)

print ('resultados logHDR')
for k in finalResultlogHDR:
	print (k)
