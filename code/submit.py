import numpy as np
import sklearn
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hinge_loss


################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################
    
    df_new = {}
    
    for i in range(0, Z_train.shape[0]):
        
        x = int(Z_train[i][-6]+2*Z_train[i][-7]+4*Z_train[i][-8]+8*Z_train[i][-9])
        y = int(Z_train[i][-2]+2*Z_train[i][-3]+4*Z_train[i][-4]+8*Z_train[i][-5])

        t = 16*x+y

        flag=False
        if(x>y):
            flag=True
            t=16*y+x
            
        if t not in df_new:
            df_new[t] = np.empty([1, Z_train.shape[1]])

        Z_train_copyrow = Z_train[i].copy()
       
        if (flag):
            Z_train_copyrow[-1]=1-Z_train_copyrow[-1]

             
        df_new[t] = np.vstack((df_new[t], Z_train_copyrow))
      

    for key in df_new:
        df_new[key]=np.delete(df_new[key],0,axis=0)

    R = 64

    models = {}

    for i in range(0,15):
        for j in range(i+1,16):
            
            # clf = LinearSVC(penalty='l2',loss='squared_hinge',random_state=0, tol=1e-4, max_iter=10000, C=11)
            clf=LogisticRegression( penalty='l2', solver='newton-cg', tol=1e-2, max_iter=10000, C=26)
            
            clf.fit( df_new[16*i+j][ :, :R ], df_new[16*i+j][ :, -1 ] )
            models[16*i+j]=clf
                
    return models




################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model):
################################
#  Non Editable Region Ending  #
################################

    X_pred=np.zeros((X_tst.shape[0]))
    R=64


    for i in range(0, X_tst.shape[0]):
        
        x = int(X_tst[i][-5]+2*X_tst[i][-6]+4*X_tst[i][-7]+8*X_tst[i][-8])
        y = int(X_tst[i][-1]+2*X_tst[i][-2]+4*X_tst[i][-3]+8*X_tst[i][-4])

        
        t = 16*x+y
       
        flag=False
        if(x>y):
            flag=True
            t=16*y+x

        X_array_pred=X_tst[ i, :R ].reshape(1,R) 
    
        clf=model[t].predict(X_array_pred)
          
        if (flag):
            clf=1-clf
            
        X_pred[i]=clf
    return X_pred

   
            