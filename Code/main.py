from mnist import MNIST
import math
import random
mndata = MNIST('../dataset')
training=mndata.load_training()
testing=mndata.load_testing()
final_testing1=[]
final_testing2=[]
final_training1=[]
final_training2=[]
wji=[] #30*785 = nH*(d+1)
wkj=[] #10*31  = c*(nH+1)

def average(s):
	return sum(s)*1.0/len(s);

error_rates=[]

'''preprocessing'''
for i in range(0,len(testing[1])):
	final_testing1.append(testing[0][i])
	pos=testing[1][i]
	bit_array=[]
	for j in range(0,10):
		if(j!=pos):
			bit_array.append(0)
		else:
			bit_array.append(1)
	final_testing2.append(bit_array)

testing=[final_testing1,final_testing2]


for i in range(0,len(training[1])):
	final_training1.append(training[0][i])
	pos=training[1][i]
	bit_array=[]
	for j in range(0,10):
		if(j!=pos):
			bit_array.append(0)
		else:
			bit_array.append(1)
	final_training2.append(bit_array)

training=[final_training1,final_training2]

c=10
d=784
nH=350
input_layers=784
hidden_layers=350
output_layers=10
eta=0.01
theta=0.001
gamma=10
#sigmoid function
def f(z):
    if(z>100.0):
        return 1.0
    elif(z<-100.0):
	return 0.0
    k=1.0+math.exp(-1*z)
    return 1.0/k


#criterion function
def J(t,z):
	ans=0;
	for i in range(0,c):
		ans+=(t[i]-z[i])*(t[i]-z[i])
	ans =  ans/2.0
	temp=0
	for i in range(0,nH):
		for j in range(1,d+1):
			temp+=wji[i][j]*wji[i][j]
	temp1=0
	for i in range(0,c):
		for j in range(1,nH+1):
			temp1+=wkj[i][j]*wkj[i][j]

	ans = ans + (gamma/(2*len(training[1])))*temp + (gamma/(2*len(training[1])))*temp1
	return ans

#derivative of sigmoid function
def f_dash(z):
	return f(z)*(1-f(z))


#initialize
m=0
for j in range(0,nH):
	temp=[]
	for i in range(0,d+1):
		temp.append(random.uniform(0.1,1.0))
	wji.append(temp)

for k in range(0,c):
	temp=[]
	for j in range(0,nH+1):
		temp.append(random.uniform(0.1,1.0))
	wkj.append(temp)

for i in range(0,c):
	for j in range(0,nH+1):
		wkj[i][j]=(wkj[i][j]*2-1)/(input_layers**0.5)

for j in range(0,nH):
	for i in range(0,d+1):
		wji[j][i]=(wji[j][i]*2-1)/(hidden_layers**0.5)

#print wkj
#print wji
#print len(wji),len(wji[0])	
perm_training=training
perm_testing=testing
for me in range(0,6):
	confusion_matrix=[]
	for i in range(0,10):
		temp=[]
		for j in range(0,10):
			temp.append(0)
		confusion_matrix.append(temp)
	
	if(me==0):
		#1st fold for 5-fold cross validation
		testing=[perm_training[0][0:12000],perm_training[1][0:12000]]
		training=[perm_training[0][12000:60000],perm_training[1][12000:60000]]
	
	if(me==1):
		#2nd fold for 5-fold cross validation
		testing=[perm_training[0][12000:24000],perm_training[1][12000:24000]]
		training=[perm_training[0][0:12000]+perm_training[0][24000:60000],perm_training[1][0:12000]+perm_training[1][24000:60000]]

	if(me==2):
		#3rd fold for 5-fold cross validation
		testing=[perm_training[0][24000:36000],perm_training[1][24000:36000]]
		training=[perm_training[0][0:24000]+perm_training[0][36000:60000],perm_training[1][0:24000]+perm_training[1][36000:60000]]


	if(me==3):
		#4th fold for 5-fold cross validation
		testing=[perm_training[0][36000:48000],perm_training[1][36000:48000]]
		training=[perm_training[0][0:36000]+perm_training[0][48000:60000],perm_training[1][0:36000]+perm_training[1][48000:60000]]

	if(me==4):
		#5th fold for 5-fold cross validation
		testing=[perm_training[0][48000:60000],perm_training[1][48000:60000]]
		training=[perm_training[0][0:48000],perm_training[1][0:48000]]

	if(me==5):
		testing=perm_testing
		training=perm_training

	while(1):
		m=random.randint(0,len(training[0]))
		x=training[0][m]
		t=training[1][m]
		#print x
		#print t
		y=[]
		z=[]
		netj=[]
		netk=[]
		for j in range(0,nH):
			sumi=wji[j][0]
			for i in range(0,d):
				sumi+=((x[i]/255.0)*wji[j][i+1])
			netj.append(sumi)
		for i in range(0,len(netj)):
			y.append(f(netj[i]))
		#print y
		for k in range(0,c):
			sumi=wkj[k][0]
			for j in range(0,nH):
				sumi+=y[j]*wkj[k][j+1]
			netk.append(sumi)
		for i in range(0,len(netk)):
			z.append(f(netk[i]))

		#print y
		#print z
		prev_J = J(t,z)
		#print prev_J
		deltak=[]
		for i in range(0,len(t)):
			deltak.append((t[i]-z[i])*f_dash(netk[i]))

		#print deltak
		deltaj=[]
		temp=[]
		for i in range(1,nH+1):
			sumi=0
			for j in range(0,c):
				sumi+=deltak[j]*wkj[j][i]
			temp.append(sumi)
		#print len(temp)

		for i in range(0,len(netj)):
			deltaj.append(f_dash(netj[i])*temp[i])

		#print deltaj
		#update wji
		new_x=[1.0]
		for i in range(0,len(x)):
			new_x.append(x[i])

		prod=[]
		for i in range(0,nH):
			temp=[]
			for j in range(0,d+1):
				temp.append(deltaj[i]*new_x[j])
			prod.append(temp)

		for i in range(0,nH):
			for j in range(0,d+1):
				wji[i][j]+=eta*prod[i][j]

		for i in range(0,nH):
			for j in range(1,d+1):
				wji[i][j]=wji[i][j] - (eta*gamma/len(training[1]))*wji[i][j]

		#print len(wji),len(wji[0])

		#print wji
		#update wkj

		new_y=[1.0]
		for i in range(0,len(y)):
			new_y.append(y[i])

		prod=[]
		for i in range(0,c):
			temp=[]
			for j in range(0,nH+1):
				temp.append(deltak[i]*new_y[j])
			prod.append(temp)
		
		for i in range(0,c):
			for j in range(0,nH+1):
				wkj[i][j]+=eta*prod[i][j]

		for i in range(0,c):
			for j in range(1,nH+1):
				wkj[i][j]=wkj[i][j] - (eta*gamma/len(training[1]))*wkj[i][j]
		#print len(wkj),len(wkj[0])
		#print wkj
		x1=training[0][m]
		t1=training[1][m]
		y1=[]
		z1=[]
		netj1=[]
		netk1=[]
		for j in range(0,nH):
			sumi=wji[j][0]
			for i in range(0,d):
				sumi+=((x1[i]/255.0)*wji[j][i+1])
			netj1.append(sumi)

		#print netj1

		for i in range(0,len(netj1)):
			y1.append(f(netj1[i]))

		for k in range(0,c):
			sumi=wkj[k][0]
			for j in range(0,nH):
				sumi+=y1[j]*wkj[k][j+1]
			netk1.append(sumi)

		for i in range(0,len(netk1)):
			z1.append(f(netk1[i]))

		new_J=J(t1,z1)
		delta_J = prev_J - new_J
		#print delta_J
		if(abs(delta_J) < theta):
			break
		m=((m+1)%(len(training[1])))
		#print wji
		#print wkj
		
	#print wji
	#	print wkj
	#	print len(wji),len(wji[0])
	#	print len(wkj),len(wkj[0])

	correct=0
	for m in range(0,len(testing[1])):
		x=training[0][m]
		t=training[1][m]
		ans=0
		ans1=0
		for i in range(0,10):
			if(t[i]==1):
				ans=i
		#print x
		#print t
		y=[]
		z=[]
		netj=[]
		netk=[]
		for j in range(0,nH):
			sumi=wji[j][0]
			for i in range(0,d):
				sumi+=((x[i]/255.0)*wji[j][i+1])
			netj.append(sumi)
		for i in range(0,len(netj)):
			y.append(f(netj[i]))
		#print y
		for k in range(0,c):
			sumi=wkj[k][0]
			for j in range(0,nH):
				sumi+=y[j]*wkj[k][j+1]
			netk.append(sumi)
		for i in range(0,len(netk)):
			z.append(f(netk[i]))
		maxi=-1
		for i in range(0,len(z)):
		    if(z[i]>maxi):
			    maxi=z[i]
			    ans1=i
		if(int(ans) == int(ans1) ):
			correct=correct+1

		confusion_matrix[ans][ans1]+=1
		#print ans,ans1

	print "\n"
	print "Accuracy for fold ",me +1 , correct/float(len(testing[1]))
	print "\n"
	temp=float(correct)/len(testing[1])
	print "Error Rate for fold ",me+1, 100-temp*100
	print "\n"
	error_rates.append(100-temp)
	print "\n"
	for i in range(0,10):
		for j in range(0,10):
			print confusion_matrix[i][j],
		print "\n"

	true_positive=[]
	true_negative=[]
	false_positive=[]
	false_negative=[]
	for i in range(0,10):
		true_positive.append(confusion_matrix[i][i])

	sumi=0
	sumj=0
	for i in range(0,10):
		if(i!=0):
			sumi+=confusion_matrix[i][0]
			sumj+=confusion_matrix[0][i]
	false_positive.append(sumi)
	false_negative.append(sumj)
	true_negative.append(12000-sumi-sumj-confusion_matrix[0][0])
	
	sumi=0
	sumj=0
	for i in range(0,10):
		if(i!=1):
			sumi+=confusion_matrix[i][1]
			sumj+=confusion_matrix[1][i]
	false_positive.append(sumi)
	false_negative.append(sumj)
	true_negative.append(12000-sumi-sumj-confusion_matrix[1][1])
	
	sumi=0
	sumj=0
	for i in range(0,10):
		if(i!=2):
			sumi+=confusion_matrix[i][2]
			sumj+=confusion_matrix[2][i]
	false_positive.append(sumi)
	false_negative.append(sumj)
	true_negative.append(12000-sumi-sumj-confusion_matrix[2][2])

	sumi=0
	sumj=0
	for i in range(0,10):
		if(i!=3):
			sumi+=confusion_matrix[i][3]
			sumj+=confusion_matrix[3][i]
	false_positive.append(sumi)
	false_negative.append(sumj)
	true_negative.append(12000-sumi-sumj-confusion_matrix[3][3])

	sumi=0
	sumj=0
	for i in range(0,10):
		if(i!=4):
			sumi+=confusion_matrix[i][4]
			sumj+=confusion_matrix[4][i]
	false_positive.append(sumi)
	false_negative.append(sumj)
	true_negative.append(12000-sumi-sumj-confusion_matrix[4][4])
	sumi=0
	sumj=0
	for i in range(0,10):
		if(i!=5):
			sumi+=confusion_matrix[i][5]
			sumj+=confusion_matrix[5][i]
	false_positive.append(sumi)
	false_negative.append(sumj)
	true_negative.append(12000-sumi-sumj-confusion_matrix[5][5])
	sumi=0
	sumj=0
	for i in range(0,10):
		if(i!=6):
			sumi+=confusion_matrix[i][6]
			sumj+=confusion_matrix[6][i]
	false_positive.append(sumi)
	false_negative.append(sumj)
	true_negative.append(12000-sumi-sumj-confusion_matrix[6][6])
	sumi=0
	sumj=0
	for i in range(0,10):
		if(i!=7):
			sumi+=confusion_matrix[i][7]
			sumj+=confusion_matrix[7][i]
	false_positive.append(sumi)
	false_negative.append(sumj)
	true_negative.append(12000-sumi-sumj-confusion_matrix[7][7])
	sumi=0
	sumj=0
	for i in range(0,10):
		if(i!=8):
			sumi+=confusion_matrix[i][8]
			sumj+=confusion_matrix[8][i]
	false_positive.append(sumi)
	false_negative.append(sumj)
	true_negative.append(12000-sumi-sumj-confusion_matrix[8][8])
	sumi=0
	sumj=0
	for i in range(0,10):
		if(i!=9):
			sumi+=confusion_matrix[i][9]
			sumj+=confusion_matrix[9][i]
	false_positive.append(sumi)
	false_negative.append(sumj)
	true_negative.append(12000-sumi-sumj-confusion_matrix[9][9])

	tp=average(true_positive)
	fp=average(false_positive)
	tn=average(true_negative)
	fn=average(false_negative)
	precision = float(tp)/(tp+fp)
	print "\n"
	print "Precision for fold " , me+1 , precision
	print "\n"
	senstivity = float(tp)/(tp+fn)
	print "Senstivity for fold ", me+1, senstivity
	print "\n"
	specificity = float(tn)/(fp+tn)
	print "Speceficity for fold ", me+1 , specificity
	print "\n"

avg = average(error_rates);
variance = map(lambda x:(x - avg)**2, error_rates);
sd = math.sqrt(average(variance));
print "\n"
print "Average error rate" , avg
print "\n"
print "Standard Deviation of error rates", sd
print "\n"
