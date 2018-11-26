import time
import os

print('iniciando execucoes')

#print (os.system("bash executar.ALL"))

print('entrando em espera')

#time.sleep(60*17)

print('fim execucao\nRodar repetibilidade')

os.system('bash executar.detector_CV.distribution_and_RR')

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

print ('................................RESULTADO FINAL......................................................')
print ('hdr: (', DmediaHDR, ';', RmediaHDR, ')')

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

print ('log: (', DmediaHDR, ';', RmediaHDR, ')')






