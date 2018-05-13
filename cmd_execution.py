
import sys
import scipy.fftpack as pack
import numpy as np
import math
from PIL import Image, ImageOps
from timeit import default_timer as timer
from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn import preprocessing
from time import time as tick
import scipy.misc
import scipy.ndimage
import sys

start = timer()
def getPontosLinha(pt0,pt1, linepts):
    #retorna os pontos da linha pt0 - pt1
    if pt0[1] < pt1[1]:
        x0 = int(pt0[1])
        y0 = int(pt0[0])
        x1 = int(pt1[1])
        y1 = int(pt1[0])
    else:
        x0 = int(pt1[1])
        y0 = int(pt1[0])
        x1 = int(pt0[1]) 
        y1 = int(pt0[0])

    dx = x1 - x0;   dy = y1 - y0;
    ind = 0;
    #print(dx)
    #print(dy)
    #linepts = zeros((sizeJanela,2)) #vrf
    #print(dx+dy)
    step = 1;
    if dx == 0:
        x = x0
        if dy < 0:
            step = -1
        for y in range(y0,y1+1,step):
            linepts[ind,:] = [y,x]
            ind = ind + 1;
    else:
        if abs(dy) > abs(dx):
            v=1
            if dy < 0:
                step = -1
                v=-1
            for y in range (y0,y1+v,step):
                x = round((dx/dy)*(y - y0) + x0)
                linepts[ind,:] = [y,x]
                ind = ind + 1
        else:
            for x in range(x0,x1+1):
                y = round((dy/dx)*(x - x0) + y0);
                linepts[ind,:] = [y,x]
                #print(ind)
                ind = ind + 1;
    #for i in range (ind, linepts.shape[0]):
    #    linepts[i,:] = [-1,-1]
    return linepts
    
def setLinha(theta, sizeJanela, linepts):
    halfsize = ceil((sizeJanela-1)/2)
    #print(halfsize)
    if theta == 0:
        #mask[int(halfsize),:] = 255
        ind=0
        for x in range(0,sizeJanela):
            linepts[ind,:] = [halfsize,x]
            ind+=1
    else:
        if theta == 90:
            #mask[:,int(halfsize)] = 255
            ind=0
            for y in range(0,sizeJanela):
                linepts[ind,:] = [y,halfsize]
                ind+=1
        else:
            x0 = -halfsize
            y0 = round(x0*(math.sin(math.radians(theta))/math.cos(math.radians(theta))))
            
            if y0 < -halfsize:
                y0 = -halfsize
                x0 = round(y0*(math.cos(math.radians(theta))/math.sin(math.radians(theta))))
            
            x1 = halfsize
            y1 = round(x1*(math.sin(math.radians(theta))/math.cos(math.radians(theta))))

            if y1 > halfsize:
                y1 = halfsize
                x1 = round(y1*(math.cos(math.radians(theta))/math.sin(math.radians(theta))))
            
            
            #print(sizeJanela)
            pt0y = halfsize-y0
            pt0x = halfsize+x0
            pt1y = halfsize-y1
            pt1x = halfsize+x1
            #quando o halfsize não pode ser no meio (sizeJanela é Par)
            if pt0y== sizeJanela:
                pt0y = pt0y - 1
            if pt0x == sizeJanela:
                pt0x = pt0x -1
            if pt1y == sizeJanela:
                pt1y = pt1y - 1
            if pt1x == sizeJanela:
                pt1x = pt1x -1
                
            #pt0 = [halfsize-y0, halfsize+x0]
            #pt1 = [halfsize-y1, halfsize+x1]
            
            pt0 = [pt0y, pt0x]
            pt1 = [pt1y, pt1x]
            
            #print(pt0)
            #print(pt1)
            getPontosLinha(pt0, pt1, linepts)
        return linepts
            #desenharLinha(pt0,pt1,mask)
    
    
def getReta(angulo,sizeJanela):
    linepts = np.zeros((sizeJanela,2))
    if angulo > 90:
        setLinha(180 - angulo,sizeJanela, linepts) #180-angulo porque a função foi desenhada para angulos 0 a 90 
        for i in range(0,sizeJanela): #rodar porque o angulo foi alterado 180o
            linepts[i,1] = sizeJanela-1-int(linepts[i,1]  )
    else:
        setLinha(angulo,sizeJanela, linepts)
    return linepts

    
def getRetas(retas,retasOrtog,sizeJanela, sizeOrtogonais):    
    for angulo in range (0,180,15):
        retas[:,:,int(angulo/15)] = getReta(angulo,sizeJanela)
        if angulo > 90:
            OrtoAngulo = angulo - 90 
        else:
            OrtoAngulo = angulo + 90 
        retasOrtog[:,:,int(angulo/15)] = getReta(OrtoAngulo,sizeOrtogonais)
        for i in range(0, sizeOrtogonais): #por pontos na janela 15 por 15
            retasOrtog[i,0,int(angulo/15)] += 6
            retasOrtog[i,1,int(angulo/15)] += 6
    
def processPixel(imgJanela,mask, retas, retasOrtog):
    
    sizeJanela = imgJanela.shape[0]
    halfSize = int ((sizeJanela-1)/2) 

                #Calcular media dos valores dentro da mascara --> N
    soma=0
    count=0
    for i in range (0, sizeJanela): 
        for j in range (0, sizeJanela):
            if (mask[i,j] > 0):
                soma += imgJanela[i,j]
                count += 1
    N=soma/count 
    
                #os valores fora da mask ficam iguais à media dos outros pixeis
    for i in range (0, sizeJanela): 
        for j in range (0, sizeJanela):
            if (mask[i,j] == 0):
                imgJanela[i,j] = N   
                
    L=0
    LOrto=0
    for reta in range (0, retas.shape[2]):  
        #Calcular media dos valores de cada Linha --> L
        sumLine=0
        count=0
        for i in range (0, sizeJanela): 
            sumLine += imgJanela[int(retas[i,0,reta]),int(retas[i,1,reta])]
            count += 1
        meanLine = sumLine/count
        if meanLine > L:
            L = meanLine
            bestReta=reta
    S=L-N 
    
    #Calcular media dos valores de cada Linha Ortogonal--> L para S0
    sumLine=0
    for i in range (0,3): 
        sumLine += imgJanela[int(retasOrtog[i,0,bestReta]),int(retasOrtog[i,1,bestReta])]
    LOrto = sumLine/3
    
    S0=LOrto-N     
       
    
    vector=np.zeros((3))
    vector[0]=S 
    vector[1]=S0
    vector[2]=imgJanela[halfSize,halfSize] #I - valor do proprio pixel
    return vector
	
	
	
	
	
	
	
	
	
	
if sys.argv[2]=='training':	
		#LER IMAGENS (TRAINING)
	print('a ler imagens de treino e teste...')
	borda=15
	DriveImgTreino=20
	driveTrainingImg  = np.zeros((584+borda*2, 565+borda*2,DriveImgTreino))
	driveTrainingMask = np.zeros((584+borda*2, 565+borda*2,DriveImgTreino))
	outputTrain       = np.zeros((584, 565,DriveImgTreino)) 

	for i in range(21,41):
		tempImg               = imread(sys.argv[1]+'DRIVE/training/images/' + str(i) + '_training.tif')[:, :, 1] #canal verde
		tempMask              = imread(sys.argv[1]+'DRIVE/training/mask/' + str(i) + '_training_mask.gif')
		outputTrain[:,:,i-21] = imread(sys.argv[1]+'DRIVE/training/1st_manual/' + str(i) + '_manual1.gif')
		#imgi.setflags(write=1) # to read-only error
		
		for y in range(0,outputTrain.shape[0]): #passar 255 para 1 porque é uma imagem
			for x in range(0,outputTrain.shape[1]):
				if (outputTrain[y,x,i-21] == 255):
					outputTrain[y,x,i-21] = 1
				
		#criar bordas para nunca se passar limites - os pixeis da borda não serão processados
		for y in range(borda,driveTrainingImg.shape[0]-borda):             
			for x in range(borda,driveTrainingImg.shape[1]-borda):        
				driveTrainingImg[y,x,i-21]  = 255-tempImg[y-borda,x-borda] #também inverte
				driveTrainingMask[y,x,i-21] = tempMask[y-borda,x-borda]
		#imsave('testleitura/img'+str(i)+'.tif',driveTrainingImg[:,:,i-21])
		#imsave('testleitura/mask'+str(i)+'.tif',driveTrainingMask[:,:,i-21])

		

	# LER IMAGENS (TEST)
	borda=15
	DriveImgTest=20
	driveTestImg =  np.zeros((584+borda*2, 565+borda*2,DriveImgTest))
	driveTestMask = np.zeros((584+borda*2, 565+borda*2,DriveImgTest))
	outputTest=             np.zeros((584, 565,DriveImgTest)) 

	for i in range(1,21):
		if i<10:   #acrescentar 0
			tempImg               = imread(sys.argv[1]+'DRIVE/test/images/0' + str(i) + '_test.tif')[:, :, 1] #canal verde
			tempMask              = imread(sys.argv[1]+'DRIVE/test/mask/0' + str(i) + '_test_mask.gif')
			outputTest[:,:,i-21]  = imread(sys.argv[1]+'DRIVE/test/1st_manual/0' + str(i) + '_manual1.gif')
		else:
			tempImg               = imread(sys.argv[1]+'DRIVE/test/images/' + str(i) + '_test.tif')[:, :, 1] #canal verde
			tempMask              = imread(sys.argv[1]+'DRIVE/test/mask/' + str(i) + '_test_mask.gif')
			outputTest[:,:,i-21]  = imread(sys.argv[1]+'DRIVE/test/1st_manual/' + str(i) + '_manual1.gif')
			#imgi.setflags(write=1) # to read-only error

		for y in range(0,outputTrain.shape[0]): #passar 255 para 1 porque é uma imagem
			for x in range(0,outputTrain.shape[1]):
				if (outputTest[y,x,i-21] == 255):
					outputTest[y,x,i-21] = 1    
			
		#criar bordas para nunca se passar limites - os pixeis da borda não serão processados
		for y in range(borda,driveTrainingImg.shape[0]-borda):             
			for x in range(borda,driveTrainingImg.shape[1]-borda):        
				driveTestImg[y,x,i-21] = 255-tempImg[y-borda,x-borda] #também inverte
				driveTestMask[y,x,i-21] = tempMask[y-borda,x-borda]
		#imsave('testleitura/img'+str(i)+'.tif',driveTrainingImg[:,:,i-21])
		#imsave('testleitura/mask'+str(i)+'.tif',driveTrainingMask[:,:,i-21])
	print('Leitura Completa')












	#Calcular pixeis das retas - funções acima
	sizeJanela=15
	sizeOrtogonais=3
	retas      = np.zeros((sizeJanela, 2, 12))  #pontos das 12 retas com angulos 0 15 30 45 60 75 90 105 120 135 150 165
	retasOrtog = np.zeros((sizeOrtogonais, 2, 12)) #pontos das 12 retas normais com angulos 0 15 30 45 60 75 90 105 120 135 150 165
	getRetas(retas, retasOrtog, sizeJanela, sizeOrtogonais)
	print('Retas Calculadas')

	#para verificar se ver se as linhas estão corretas <----
	#desenharRetas      = zeros((15, 15, 12)) #12 retas em 12 imagens 15x15 para visualização
	#for r in range(0,12): 
	#    for i in range(0,3):
	#        desenharRetas[int(retasOrtog[i,0,r]), int(retasOrtog[i,1,r]),r]=255
	#    for i in range(0,15):
	#        desenharRetas[int(retas[i,0,r]), int(retas[i,1,r]),r]=255
	#    imsave('Output/Retas/angulo'+str(r*15)+'.tif', desenharRetas[:,:,r])












	print('A começar processamento de imagens de treino...')
	#cada imagem demora 74 segundos em media, este tempo pode ser reduzido utilizando paralelismo, uma vez que o calculo
	#de um pixel é independente do calculo dos outros pixeis
	imgi = np.zeros((driveTrainingImg.shape[0], driveTrainingImg.shape[1]))
	mask = np.zeros((driveTrainingImg.shape[0], driveTrainingImg.shape[1]))
	vectToTrain = np.zeros((driveTrainingImg.shape[0]-borda*2, driveTrainingImg.shape[1]-borda*2, DriveImgTreino, 3))
	for i in range(0,20): 
		imgi=driveTrainingImg[:,:,i]
		mask=driveTrainingImg[:,:,i]
		for y in range(borda,imgi.shape[0]-borda):   
			for x in range(borda, imgi.shape[1]-borda):   
				if mask[y,x] > 0:
					vectToTrain[y-borda,x-borda,i,:] = processPixel(imgi[ y-int((sizeJanela-1)/2) : y+int((sizeJanela-1)/2)+1 , x-int((sizeJanela-1)/2) : x+int((sizeJanela-1)/2)+1], mask[ y-int((sizeJanela-1)/2) : y+int((sizeJanela-1)/2)+1 , x-int((sizeJanela-1)/2) : x+int((sizeJanela-1)/2)+1], retas , retasOrtog)
		imsave('trainImagesS/S'+str(i)+'.tif', vectToTrain[:,:,i,0])    
		print('imagem '+str(i+1)+' completa')








	print('A começar processamento de imagens de teste...')
	#cada imagem demora 74 segundos em media, este tempo pode ser reduzido utilizando paralelismo, uma vez que o calculo
	#de um pixel é independente do calculo dos outros pixeis
	imgi = np.zeros((driveTestImg.shape[0], driveTestImg.shape[1]))
	mask = np.zeros((driveTestImg.shape[0], driveTestImg.shape[1]))
	vectToTest = np.zeros((driveTestImg.shape[0]-borda*2, driveTestImg.shape[1]-borda*2, DriveImgTest, 3))
	for i in range(0,20): 
		imgi=driveTestImg[:,:,i]
		mask=driveTestImg[:,:,i]
		for y in range(borda,imgi.shape[0]-borda):   
			for x in range(borda, imgi.shape[1]-borda):   
				if mask[y,x] > 0:
					vectToTest[y-borda,x-borda,i,:] = processPixel(imgi[ y-int((sizeJanela-1)/2) : y+int((sizeJanela-1)/2)+1 , x-int((sizeJanela-1)/2) : x+int((sizeJanela-1)/2)+1], mask[ y-int((sizeJanela-1)/2) : y+int((sizeJanela-1)/2)+1 , x-int((sizeJanela-1)/2) : x+int((sizeJanela-1)/2)+1], retas , retasOrtog)
		imsave('testImagesS/S'+str(i)+'.tif', vectToTest[:,:,i,0])    
		print('imagem '+str(i+1)+' completa')






	#aleatoriedade controlada - introduzir variabilidade
	#vamos dividir cada imagem em 1000 subvetores (960 subvetores de 330 pixeis + 40 subvetores de 329 pixeis)
	##em cada um destes subvetores vamos escolher um pixel, desta forma:
	#vamos dividir cada um dos subvetores de 330/329 pixeis em 20subvetores (uma vez que temos 20 imagens) 
	#SE subvetor = 330 
		#ficam 10 janelas de 17 pixeis + 10 janelas de 16 pixeis
	#SE subvetor = 329
		#ficam 9 janelas de 17 pixeis + 11 janelas de 16 pixeis
	#na imagem 1 escolhemos 1 pixel random da 1a janela de cada subvetor
	#na imagem 2 escolhemos 1 pixel random da 2a janela de cada subvetor (etc)
	#na imagem 20 escolhemos 1 pixel random da 20a janela de cada subvetor 

	#dados de treino
	#.ravel passa para uma dimensão - essencial para o treino com svm
	X_temp = array([vectToTrain[:,:,:,0].ravel(), vectToTrain[:,:,:,1].ravel(), vectToTrain[:,:,:,2].ravel()])
	X_temp = transpose(X_temp)
	y_temp = outputTrain.ravel()

	#print(X_train.shape[0]/20) #pixeis por imagem = 329960

	amostraX = np.zeros((1000 * DriveImgTreino, 3))
	amostraY = np.zeros((1000 * DriveImgTreino))

	nmrPixel=0
	indexPixel=0
	for numImagem in range(0,20):
		nmrPixel=numImagem*329960 #pixel inicial #(nmrPixeis numa imagem = 329960)   
		indexEscolhido=0
		for subvetor in range(0,1000):
			if subvetor < 960:
				tamanhoVetor=330
				if numImagem < 10: #10 janelas de 16
					tamanhoJanela=16
					localJanela=16*numImagem
				else: #10 janelas de 17
					tamanhoJanela=17
					localJanela=16*10+17*(numImagem-10)
			else: #40 subvetores de 329 pixeis
				tamanhoVetor=329
				if numImagem < 11: #11 janelas de 16
					tamanhoJanela=16
					localJanela=16*numImagem
				else: #9 janelas de 17
					tamanhoJanela=17
					localJanela=16*11+17*(numImagem-11)
			indexEscolhido = nmrPixel + localJanela + randint(0,tamanhoJanela)
			amostraX[indexPixel,:] = X_temp[indexEscolhido,:]
			amostraY[indexPixel] = y_temp[indexEscolhido]
			nmrPixel+= tamanhoVetor 
			indexPixel+=1
	print('Amostras selecionadas adequeadamente')
	np.save('vectToTrain', vectToTrain)
	np.save('vectToTest', vectToTest)
	np.save('X_train', X_train)
	np.save('X_test', X_test)
	np.save('y_test', y_test)
	np.save('y_train', y_train)
	np.save('clf', clf)

	
	
else:
	#cada imagem demora 74 segundos em media, este tempo pode ser reduzido utilizando paralelismo, uma vez que o calculo
	#de um pixel é independente do calculo dos outros pixeis
	imgi = np.zeros((driveTrainingImg.shape[0], driveTrainingImg.shape[1]))
	mask = np.zeros((driveTrainingImg.shape[0], driveTrainingImg.shape[1]))
	vectToTrain = np.zeros((driveTrainingImg.shape[0]-borda*2, driveTrainingImg.shape[1]-borda*2, DriveImgTreino, 3))
	for i in range(0,20): 
		imgi=driveTrainingImg[:,:,i]
		mask=driveTrainingImg[:,:,i]
		for y in range(borda,imgi.shape[0]-borda):   
			for x in range(borda, imgi.shape[1]-borda):   
				if mask[y,x] > 0:
					vectToTrain[y-borda,x-borda,i,:] = processPixel(imgi[ y-int((sizeJanela-1)/2) : y+int((sizeJanela-1)/2)+1 , x-int((sizeJanela-1)/2) : x+int((sizeJanela-1)/2)+1], mask[ y-int((sizeJanela-1)/2) : y+int((sizeJanela-1)/2)+1 , x-int((sizeJanela-1)/2) : x+int((sizeJanela-1)/2)+1], retas , retasOrtog)
		imsave('trainImagesS/S'+str(i)+'.tif', vectToTrain[:,:,i,0])    
		print('imagem '+str(i+1)+' completa')




	#cada imagem demora 74 segundos em media, este tempo pode ser reduzido utilizando paralelismo, uma vez que o calculo
	#de um pixel é independente do calculo dos outros pixeis
	imgi = np.zeros((driveTestImg.shape[0], driveTestImg.shape[1]))
	mask = np.zeros((driveTestImg.shape[0], driveTestImg.shape[1]))
	vectToTest = np.zeros((driveTestImg.shape[0]-borda*2, driveTestImg.shape[1]-borda*2, DriveImgTest, 3))
	for i in range(0,20): 
		imgi=driveTestImg[:,:,i]
		mask=driveTestImg[:,:,i]
		for y in range(borda,imgi.shape[0]-borda):   
			for x in range(borda, imgi.shape[1]-borda):   
				if mask[y,x] > 0:
					vectToTest[y-borda,x-borda,i,:] = processPixel(imgi[ y-int((sizeJanela-1)/2) : y+int((sizeJanela-1)/2)+1 , x-int((sizeJanela-1)/2) : x+int((sizeJanela-1)/2)+1], mask[ y-int((sizeJanela-1)/2) : y+int((sizeJanela-1)/2)+1 , x-int((sizeJanela-1)/2) : x+int((sizeJanela-1)/2)+1], retas , retasOrtog)
		imsave('testImagesS/S'+str(i)+'.tif', vectToTest[:,:,i,0])    
		print('imagem '+str(i+1)+' completa')

		
		
		

	vectToTrain = np.load('vectToTrain.npy')
	vectToTest = np.load('vectToTest.npy')
	X_train = np.load('X_train.npy')
	X_test = np.load('X_test.npy' )
	y_test = np.load('y_test.npy' )
	y_train = np.load('y_train.npy' )
	clf = np.load('clf.npy')
	amostraX = np.load('X_train.npy')
	amostraY = np.load('y_train.npy')
	y_train=np.asarray(y_train, dtype=int)
	y_test=np.asarray(y_test, dtype=int)









	X_train = amostraX
	y_train = amostraY
	print('dimensão de dados de entrada (treino) = ' + str(X_train.shape))
	print('dimensão de dados de saida (treino) = ' + str(y_train.shape))
	random_state = np.random.RandomState(0)
	X_train, y_train = shuffle(X_train, y_train, random_state=random_state) #introduzir variabilidade

	#dados de teste
	X_test = array([vectToTest[:,:,:,0].ravel(), vectToTest[:,:,:,1].ravel(), vectToTest[:,:,:,2].ravel()])
	X_test = transpose(X_test)
	y_test = outputTest.ravel()
	print('dimensão de dados de entrada (teste) = ' + str(X_test.shape))
	print('dimensão de dados de saida (teste)= ' + str(y_test.shape))
	random_state = np.random.RandomState(0)

	#X_test, y_test = shuffle(X_test, y_test, random_state=random_state) teste não precisa de shuffle


	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	#treino com svm
	clf = SVC(kernel='linear')
	tin = tick()
	print('Treino com svm...')
	clf = clf.fit(X_train, y_train)
	tout = tick()

	print("Taxa de sucesso (Treino): ",
		  np.mean(clf.predict(X_train) == y_train) * 100)

	data = clf.predict(X_test)
	print("Taxa de sucesso (Teste): ",
		  np.mean(data == y_test) * 100)

	print("Número de vectors de dados (treino/teste): {} / {}".
		  format(X_train.shape[0], X_test.shape[0]))
	print("Número de vectores de suport: ", clf.support_vectors_.shape[0])
	print('Training time: {:.3f} s'.format(tout - tin))


	imnum=1
	for i in range(0,data.shape[0],329960):  #guardar as imagens
		img = data[i:i+329960].reshape((584,565))
		imsave('sys.argv[3]' + str(imnum)+'_testSVM.tif',img)
		imnum+=1
end = timer()
print('segundos: ' + str(end - start)) 
print('Leitura Completa')
##fim do programa




