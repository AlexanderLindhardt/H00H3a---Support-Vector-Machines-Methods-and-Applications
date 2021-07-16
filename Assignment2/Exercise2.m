clear,clc,clf
rng('default')
X = ( -3:0.01:3)';
Y = sinc ( X ) + 0.1.* randn ( length ( X ) , 1) ;

Xtrain = X (1:2: end ) ;
Ytrain = Y (1:2: end ) ;
Xtest = X (2:2: end ) ;
Ytest = Y (2:2: end ) ;
gam = 10^6; 
sig2 = 100;
type = 'function estimation';
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
Yt = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
hold on;  
%plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
hold on
err = immse(Yt,Ytest)
plot(Xtest, Ytest,'b.')
plot(Xtest,Yt,'r-.'); 
xlabel('$x$','Interpreter','latex')
ylabel('$sinc(x)$','Interpreter','latex')
title(['\gamma =', num2str(gam),',  ' ,'\sigma^2 =', num2str(sig2), ',  ', 'MSE=',num2str(err)])


%% 1.2.1 parameter tuning
clc
c = [];
tic
for i=1:50
    [ gam , sig2 , cost ] = tunelssvm({ Xtrain , Ytrain , 'f', [] , [] ,'RBF_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'mae'}) ;
    c = [c; cost];
end
toc
sum(c)/50
%% Bayesian framework
clc
sig2 = 0.4;
gam = 10;
%%
crit_L1 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 } , 1) ;
crit_L2 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 } , 2) ;
crit_L3 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 } , 3) ;
%% 
[~ , alpha , b ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 } , 1) ;
[~ , gam ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 } , 2) ;
[~ , sig2 ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 } , 3) ;
%%
 sig2e = bay_errorbar ({ Xtrain , Ytrain , 'f', gam , sig2 } , 'figure') ;

Yt = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
immse(Yt,Ytest)

%%  Automatic Relevance Determination
clc
X = 6.* rand (100 , 3) - 3;
Y = sinc ( X (: ,1) ) + 0.1.* randn (100 ,1) ;
[ gam , sig2 , cost ] = tunelssvm({ X , Y , 'f', [] , [] ,'RBF_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'mae'}) ;
[ selected , ranking ] = bay_lssvmARD ({ X , Y , 'f', gam , sig2 }) ;
%%
clf,clc
hold on;
X1 = sinc ( X (: ,1) ) + 0.1.* randn (100 ,1) ;
X2 = sinc ( X (: ,2) ) + 0.1.* randn (100 ,1) ;
X3 = sinc ( X (: ,3) ) + 0.1.* randn (100 ,1) ;
subplot(3,1,1) 
plot([X1 Y])
legend('Y(X1)','Y')
%plot(Y)
subplot(3,1,2) 
plot([X2 Y])
legend('Y(X2)','Y')
subplot(3,1,3) 
plot([X3 Y])
legend('Y(X3)','Y')

%% Robust regression
clc, clf, clear
X = ( -6:0.2:6)';
Y = sinc ( X ) + 0.1.* rand ( size ( X ) ) ;
out = [15 17 19];
Y ( out ) = 0.7+0.3* rand ( size ( out ) ) ;
out = [41 44 46];
Y ( out ) = 1.5+0.2* rand ( size ( out ) ) ;
%% No robustness
model = initlssvm (X , Y , 'f', [] , [] , 'RBF_kernel') ;
costFun = 'crossvalidatelssvm';
model = tunelssvm ( model , 'simplex', costFun , {10 , 'mse';}) ;
plotlssvm ( model ) ;
%% Robustness
model = initlssvm (X , Y , 'f', [] , [] , 'RBF_kernel') ;
costFun = 'rcrossvalidatelssvm';
wFun = 'wmyriad';
model = tunelssvm ( model , 'simplex', costFun , {10 , 'mae';} , wFun ) ;
model = robustlssvm ( model ) ;
plotlssvm ( model ) 

%% Logmap dataset
clc, clear, clf
load logmap.mat
%% 
order = 10;
X = windowize (Z , 1:( order + 1) ) ;
Y = X (: , end ) ;
X = X (: , 1: order ) ;
%%
clf
gam = 10;
sig2 = 10;
[ alpha , b ] = trainlssvm ({ X , Y , 'f', gam , sig2 }) ;

clf
Xs = Z ( end - order +1: end , 1) ;
nb = 50;
prediction = predict ({ X , Y , 'f', gam , sig2 } , Xs , nb ) ;

hold on;
plot ( Ztest , 'b') ;
plot ( prediction , 'r') ;
xlabel('time step')
legend('True', 'Predicted')

hold off;
%% Tune parameters
clc, clear, clf
load logmap.mat
error_list = [];
for order=20:25
    order
    error = 0;
    for i=1:10
        X = windowize (Z , 1:( order + 1) ) ;
        Y = X (: , end ) ;
        X = X (: , 1: order ) ;
        [ gam , sig2 , cost ] = tunelssvm({ X , Y , 'f', [] , [] ,'RBF_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'mae'}) ;
        [ alpha , b ] = trainlssvm ({ X , Y , 'f', gam , sig2 }) ;
        Xs = Z ( end - order +1: end , 1) ;
        nb = 50;
        prediction = predict ({ X , Y , 'f', gam , sig2 } , Xs , nb ) ;
        error = error + immse(Ztest, prediction);
    end
    error_list = [error_list; error/5];
end
        
%%
clf
order = 23;
X = windowize (Z , 1:( order + 1) ) ;
Y = X (: , end ) ;
X = X (: , 1: order ) ;
[ gam , sig2 , cost ] = tunelssvm({ X , Y , 'f', [] , [] ,'RBF_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'mae'}) ;
[ alpha , b ] = trainlssvm ({ X , Y , 'f', gam , sig2 }) ;
Xs = Z ( end - order +1: end , 1) ;
nb = 50;
prediction = predict ({ X , Y , 'f', gam , sig2 } , Xs , nb ) ;
immse(Ztest, prediction)
hold on;
plot ( Ztest , 'b') ;
plot ( prediction , 'r') ;
xlabel('time step')
legend('True', 'Predicted')


%% Santa fe Dataset
clear, clc, clf
load santafe.mat
hold on;
order =50;
X = windowize (Z , 1:( order + 1) ) ;
Y = X (: , end ) ;
X = X (: , 1: order ) ;
plot(Z,'b')
plot(1001:1200, Ztest,'r')
xlabel('time step')
legend('Training', 'Test')

%%
clf
gam = 10;
sig2 = 10;
[ gam , sig2 , cost ] = tunelssvm({ X , Y , 'f', [] , [] ,'RBF_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'mae'}) ;
[ alpha , b ] = trainlssvm ({ X , Y , 'f', gam , sig2 }) ;

clf
Xs = Z ( end - order +1: end , 1) ;
nb = 200;
prediction = predict ({ X , Y , 'f', gam , sig2 } , Xs , nb ) ;

hold on;
plot ( Ztest , 'b') ;
plot ( prediction , 'r') ;
xlabel('time step')
legend('True', 'Predicted')
%%
clc, clear, clf
load santafe.mat
error_list = [];
for order=20:60
    order
    error = 0;
    for i=1:1
        X = windowize (Z , 1:( order + 1) ) ;
        Y = X (: , end ) ;
        X = X (: , 1: order ) ;
        [ gam , sig2 , cost ] = tunelssvm({ X , Y , 'f', [] , [] ,'RBF_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'mae'}) ;
        [ alpha , b ] = trainlssvm ({ X , Y , 'f', gam , sig2 }) ;
        Xs = Z ( end - order +1: end , 1) ;
        nb = 200;
        prediction = predict ({ X , Y , 'f', gam , sig2 } , Xs , nb ) ;
        error = error + immse(Ztest, prediction);
    end
    error_list = [error_list; error/5];
end
        