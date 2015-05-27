import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import exp
from math import sqrt
import matplotlib.pyplot as plt
import pickle
from numpy.linalg import inv
from numpy.linalg import det
from math import log

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    X_and_y=np.concatenate((X,y),axis=1)
    X_and_y_sort=np.sort(X_and_y.view('i8,i8,i8'), order=['f2'], axis=0).view(np.float)
    X_sorted=X_and_y_sort[:, :2]
    y_array = np.squeeze(np.asarray(y))  # Convert N x 1 matrix to N x 1 array 
    unique_elements_count=np.bincount((y_array.astype(int))) # get the number of unique elements present in the array
    slice_index = 0
    means=np.zeros((1,2))
    covmat=np.zeros((2,2))
    X_sorted_mean=X_sorted.mean(0)
    for index in unique_elements_count :
        if(index!=0) :
           old_slice_index=slice_index
           slice_index=slice_index+index
           X_temp = X_sorted[old_slice_index:slice_index, :]
           means_temp = X_temp.mean(0)
           means=np.vstack((means,means_temp))
           X_temp=X_temp-X_sorted_mean
           no_of_rows=X_temp.shape[0] 
           covmats_temp=np.dot(X_temp.transpose(),X_temp)/no_of_rows
           covmats_temp=covmats_temp*X_temp.shape[0]/X.shape[0]
           covmat=covmat+covmats_temp           
    means=means[1:,:]
    means=means.transpose()
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    X_and_y=np.concatenate((X,y),axis=1)
    X_and_y_sort=np.sort(X_and_y.view('i8,i8,i8'), order=['f2'], axis=0).view(np.float)
    X_sorted=X_and_y_sort[:, :2]
    y_array = np.squeeze(np.asarray(y))  # Convert N x 1 matrix to N x 1 array 
    unique_elements_count=np.bincount((y_array.astype(int))) # get the number of unique elements present in the array
    slice_index = 0
    means=np.zeros((1,2))
    covmats=[]
    X_sorted_mean=X_sorted.mean(0)
    for index in unique_elements_count :
        if(index!=0) :
           old_slice_index=slice_index
           slice_index=slice_index+index
           X_temp = X_sorted[old_slice_index:slice_index, :]
           means_temp = X_temp.mean(0)
           means=np.vstack((means,means_temp))
           X_temp=X_temp-X_sorted_mean
           covmats_temp=np.cov(X_temp,rowvar=0)
           covmats.append(covmats_temp)       
    means=means[1:,:]
    means=means.transpose()                                   
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    
    covmat_inv=inv(covmat)
    means=means.transpose()
    discriminant_max=[]
    for Xtest_row in Xtest :
        discriminant=[]
        for means_row in means : 
           const1=np.dot(means_row,covmat_inv)
           const1=np.dot(const1,Xtest_row)
           const2=np.dot(means_row,covmat_inv)
           const2=0.5*np.dot(const2,means_row.transpose())
           const3=log(1.0/means.shape[0])
           diff=const1-const2+const3
           discriminant.append(diff)    
        discriminant_max.append(np.argmax(discriminant)+1)           
    discriminant_max=np.array(discriminant_max)
    discriminant_max=discriminant_max.reshape(discriminant_max.shape[0],1)
    acc=str(100*np.mean((discriminant_max == ytest).astype(float)))
    return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    
    means=means.transpose()
    discriminant_max=[]
    for Xtest_row in Xtest :
        discriminant=[]
        comvat_index=0
        for means_row in means :
           covmat=covmats[comvat_index]
           covmat_inv=inv(covmats[comvat_index])
           const1=-0.5*log(det(covmat))
           const2_temp=Xtest_row-means_row
           const2=np.dot(covmat_inv,const2_temp)
           const2=np.dot(const2_temp.transpose(),const2)
           const2=-0.5*const2
           const3=log(1.0/means.shape[0])
           diff=const1+const2+const3
           discriminant.append(diff)
           comvat_index=comvat_index+1   
        discriminant_max.append(np.argmax(discriminant)+1)           
    discriminant_max=np.array(discriminant_max).astype(float)
    discriminant_max=discriminant_max.reshape(discriminant_max.shape[0],1)  
    acc=str(100*np.mean((discriminant_max == ytest).astype(float))) 
    return acc

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                              
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD                                                   
    
    xtx=np.dot(X.transpose(),X)
    xtx_inv=inv(xtx)
    temp=np.dot(xtx_inv,X.transpose())
    w=np.dot(temp,y)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD

    xtx = np.dot(X.transpose(),X)
    no_of_columns = X.shape[1]
    no_of_rows = X.shape[0]
    lambda_identity_matrix = lambd*np.eye(no_of_columns)
    lambda_identity_matrix = no_of_rows*lambda_identity_matrix
    w = xtx + lambda_identity_matrix
    w = inv(w)
    w = np.dot(w,X.transpose())
    w = np.dot(w,y)
                                                                                                 
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    
    diff=(ytest - (np.dot(Xtest,w)))
    diff=pow(diff[:],2)
    diff_sum=np.sum(diff[:]) 
    number_of_elements=diff.shape[0]*diff.shape[1]
    rmse=sqrt(diff_sum)/number_of_elements
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD

    N_rows_X=X.shape[0]
    w=w.reshape(X.shape[1],1)  
    error=(0.5/N_rows_X)*np.dot((y-np.dot(X,w)).transpose(),(y-np.dot(X,w)))+0.5*lambd*np.dot(w.transpose(),w)
    error_grad=(1.0/N_rows_X)*(np.dot(w.transpose(),np.dot(X.transpose(),X))-np.dot(y.transpose(),X))+lambd*w.transpose() 
    error_grad=error_grad.transpose()
    error_grad=error_grad.flatten()
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    
    Xd = np.zeros((x.shape[0],p+1))
    for power in range(0,p+1) :
        Xd[:,power] = pow(x,power)    
    return Xd
    
def plot_lda(means,covmat,xx,yy,ldaacc) :
    covmat_inv=inv(covmat)
    means=means.transpose()    
    for X in xx :
        for Y in yy :
            Xtest_row=np.array([X,Y])
            discriminant=[]
            for means_row in means : 
                const1=np.dot(means_row,covmat_inv)
                const1=np.dot(const1,Xtest_row)
                const2=np.dot(means_row,covmat_inv)
                const2=0.5*np.dot(const2,means_row.transpose())
                const3=log(1.0/means.shape[0])
                diff=const1-const2+const3
                discriminant.append(diff)
            Xtest_row=np.array([])        
            maxval=np.argmax(discriminant)+1
            if maxval==1.0 :
                plt.plot(X,Y,marker='.' , color='b')
            elif maxval==2.0 :    
                plt.plot(X,Y,marker='.' , color='c')
            elif maxval==3.0:    
                plt.plot(X,Y,marker='.' , color='g')
            elif maxval==4.0:   
                plt.plot(X,Y,marker='.' , color='r')
            elif maxval==5.0:    
                plt.plot(X,Y,marker='.' , color='m')
    plt.axis('off')
    plt.title("Classification Plot:LDA accuracy : "+str(ldaacc)+"%")                                   
    plt.savefig("problem1_lda.jpg")
    plt.clf() 
    return

def plot_qda(means,covmat,xx,yy,qdaacc) :
    means=means.transpose()
    maxval_c=0
    
    for X in xx :
        for Y in yy :
            Xtest_row=np.array([X,Y])
            discriminant=[]
            comvat_index=0
            maxval=0
            for means_row in means :
                covmat=covmats[comvat_index]
                covmat_inv=inv(covmats[comvat_index])
                const1=-0.5*log(det(covmat))
                const2_temp=Xtest_row.transpose()-means_row
                const2=np.dot(covmat_inv,const2_temp.transpose()) 
                const2=np.dot(const2_temp,const2)
                const2=-0.5*const2
                const3=log(1.0/means.shape[0])
                diff=const1+const2+const3
                discriminant.append(diff)
                comvat_index=comvat_index+1  
            maxval=np.argmax(discriminant)+1
            maxval_c=maxval_c+1
            if maxval==1.0 :
                plt.plot(X,Y,marker='.' , color='b')
            elif maxval==2.0 :    
                plt.plot(X,Y,marker='.' , color='c')
            elif maxval==3.0:    
                plt.plot(X,Y,marker='.' , color='g')
            elif maxval==4.0:    
                plt.plot(X,Y,marker='.' , color='r')
            elif maxval==5.0:    
                plt.plot(X,Y,marker='.' , color='m')
    plt.axis('off')
    plt.title("Classification Plot:QDA accuracy : "+str(qdaacc)+"%")                                   
    plt.savefig("problem1_qda.jpg")
    plt.clf()            
    return 
    
          
# Main script
######################################### problem 1 ############################################
# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
# LDA
means,covmat = ldaLearn(X,y)
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
min_range=max(x_min,y_min)
max_range=min(x_max,y_max)
xx, yy = np.meshgrid(np.arange(min_range,max_range, 0.2),np.arange(min_range,max_range, 0.2),sparse=True)
xx=xx.transpose()  
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
plot_lda(means,covmat,xx,yy,ldaacc)


# QDA
means,covmats = qdaLearn(X,y)
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
min_range=max(x_min,y_min)
max_range=min(x_max,y_max)
xx, yy = np.meshgrid(np.arange(min_range,max_range, 0.2),np.arange(min_range,max_range, 0.2),sparse=True)
xx=xx.transpose()
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))
plot_qda(means,covmats,xx,yy,qdaacc)
######################################### problem 1 ############################################

######################################### problem 2 ############################################
#Problem 2
X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
mle_train = testOLERegression(w,X,y)
w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
mle_i_train = testOLERegression(w_i,X_i,y)
#print w_i
print('RMSE without intercept training data : '+str(mle_train))
print('RMSE with intercept training data : '+str(mle_i_train))
print('RMSE without intercept test data: '+str(mle))
print('RMSE with intercept test data: '+str(mle_i))
######################################### problem 2 ############################################


######################################### problem 3 ############################################
# Problem 3
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses3_train = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses3_train[i] = testOLERegression(w_l,X_i,y) 
    #if str(lambd) == str(0.00024):
        #print "optimum lambda",lambd 
        #print w_l
    i = i + 1
#print rmses3
#print rmses3_train        
test=plt.plot(lambdas,rmses3,label='Test Data')
train=plt.plot(lambdas,rmses3_train,label='Training Data')
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title("RidgeRegression - RMSE vs Lambda")
plt.legend(loc='upper right',frameon=False)
plt.savefig("Problem3.jpg")
plt.clf()
lambda_opt_prob_3 = lambdas[np.argmin(rmses3)]
#print "lambda optimal"
#print lambda_opt_prob_3 

######################################### problem 3 ############################################


######################################### problem 4 ############################################ 
lambdas = np.linspace(0, 0.004, num=k)
k = 101
i = 0
rmses4 = np.zeros((k,1))
rmses4_train = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))

for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    rmses4_train[i] = testOLERegression(w_l_1,X_i,y)
    i = i + 1
#print rmses4
#print rmses4_train      
plt.plot(lambdas,rmses4,label='Test Data')
plt.plot(lambdas,rmses4_train,label='Train Data')
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title("Gradient De-scent Ridge Regression-RMSE vs Lambda")
plt.legend(loc='lower right',frameon=False)
plt.savefig("Problem4.jpg")
plt.clf()
######################################### problem 4 ############################################


################################## problem 5 ################################################
# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
lambda_opt_train=lambdas[np.argmin(rmses4_train)]
#print lambda_opt_train
rmses5 = np.zeros((pmax,2))
rmses5_train = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    rmses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    rmses5_train[p,1] = testOLERegression(w_d2,Xd,y)
plt.title('Non-linear Regression-RMSE vs P - Test Data')    
plt.xlabel('P')
plt.ylabel('RMSE')
plt.plot(range(pmax),rmses5[:,0],label='No Regularization(lambda=0)')
plt.plot(range(pmax),rmses5[:,1],label='Regularization(lambda='+str(lambda_opt)+')')
plt.legend(loc='upper right',frameon=False)
plt.savefig("Problem5.jpg")
plt.clf()
#print rmses5 
plt.title('Non-linear Regression-RMSE vs P - Train Data')    
plt.xlabel('P')
plt.ylabel('RMSE')
plt.plot(range(pmax),rmses5_train[:,0],label='No Regularization(lambda=0)')
plt.plot(range(pmax),rmses5_train[:,1],label='Regularization(lambda='+str(lambda_opt)+')')
plt.legend(loc='upper right',frameon=False)
plt.savefig("Problem5_1.jpg")
#print rmses5_train
plt.clf()

################################## problem 5 ################################################
