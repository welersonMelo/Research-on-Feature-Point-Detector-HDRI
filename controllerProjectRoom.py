import time
import os

num = "1234"

finalResultHDR = []
finalResultlogHDR = []

for gaussianOp in num:
	files = ["executar.detector_CV.3Dlighting1", "executar.detector_CV.3Dlighting2"]
		x = []
	
	for i in range(len(files)):
		x.append(open(files[i]+'.copy').read())

	for i in range(len(x)):
		s = x[i].replace('\n', ' '+gaussianOp+' \n')
		f = open(files[i], "w")
		f.write(s)
		f.close()

	print('iniciando execucoes')

	print (os.system("bash executar.ALL3D"))

	print('entrando em espera')
	minutes = 4 # tempode de espera em minutos
	time.sleep(60*minutes) 

	print('fim execucao\nRodar repetibilidade')

	os.system('bash executar.detector_CV.distribution_and_RR3D')

	print ('entrando na pasta dataset/projectRoom')

	print ('Pegando dados gerados e fazendo media')

	volta = '../dataset/projectRoom/'

	#df1 = open(volta+'distance/Dist/distribution.HDR.detectorCV.distribution.txt')
	df2 = open(volta+'lighting/Dist/distribution.HDR.detectorCV.distribution.txt')
	#df3 = open(volta+'viewpoint/Dist/distribution.HDR.detectorCV.distribution.txt')

	#dv1 = float(df1.read())
	dv2 = float(df2.read())
	#dv3 = float(df3.read())

	DmediaHDR = dv2

	#rf1 = open(volta+'distance/RR/RR.HDR.detectorCV1.txt')
	rf2 = open(volta+'lighting/RR/RR.HDR.detectorCV1.txt')
	#rf3 = open(volta+'viewpoint/RR/RR.HDR.detectorCV1.txt')

	#rv1 = float(rf1.read()[2:])
	rv2 = float(rf2.read()[2:])
	#rv3 = float(rf3.read()[2:])

	RmediaHDR = rv2

	
	finalResultHDR.append('(' + str (DmediaHDR) + ' ; '+ str(RmediaHDR) +')')

	#df1 = open(volta+'distance/Dist/distribution.logHDR.detectorCV.distribution.txt')
	df2 = open(volta+'lighting/Dist/distribution.logHDR.detectorCV.distribution.txt')
	#df3 = open(volta+'viewpoint/Dist/distribution.logHDR.detectorCV.distribution.txt')

	#dv1 = float(df1.read())
	dv2 = float(df2.read())
	#dv3 = float(df3.read())

	DmediaHDR = dv2

	#rf1 = open(volta+'distance/RR/RR.logHDR.detectorCV1.txt')
	rf2 = open(volta+'lighting/RR/RR.logHDR.detectorCV1.txt')
	#rf3 = open(volta+'viewpoint/RR/RR.logHDR.detectorCV1.txt')

	#rv1 = float(rf1.read()[2:])
	rv2 = float(rf2.read()[2:])
	#rv3 = float(rf3.read()[2:])

	RmediaHDR = rv2

	finalResultlogHDR.append('(' + str (DmediaHDR) + ' ; ' + str (RmediaHDR) + ')')


print ('resultados HDR')
for k in finalResultHDR:
	print (k)

print ('resultados logHDR')
for k in finalResultlogHDR:
	print (k)
