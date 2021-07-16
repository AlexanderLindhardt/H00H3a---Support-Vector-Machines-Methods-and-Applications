%% 1.1 Two Gaussians
clc, clf, clear
rng('default')
X1 = randn (50 ,2) + 1;
X2 = randn (51 ,2) - 1;

Y1 = ones (50 ,1) ;
Y2 = - ones (51 ,1) ;
x = linspace(-4, 4,100);
y = linspace(4, -4, 100);

hold on;
plot ( X1 (: ,1) , X1 (: ,2) , 'ro') ;
plot ( X2 (: ,1) , X2 (: ,2) , 'bo') ;
plot(x, y, 'k')
hold off;
legend('X1', 'X2', 'line classifier', 'location', 'northwest')

%% 1.3.1.1
clc,clf, clear
load iris.mat

type='c';
gam = 1;
t = 1;
err = zeros(10,1);
for degree = 7:7
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'});

    plotlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});

    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xtest);
    err(degree) = sum(Yht~=Ytest);
end
%plot(err)
%% 1.3.1.2
clc,clf, clear
load iris.mat

type='c';
gam = 1;
t = 1;
err = [];
for sig2 = [0.1, 0.5, 1,2,3,4, 5,6,7,8,9 10]
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = [err; sum(Yht~=Ytest)];
end
plot([0.1, 0.5, 1,2,3,4, 5,6,7,8,9 10], err, 'x-')
%% 1.3.1.2
clc,clf, clear
load iris.mat

type='c';
sig2 = 1;
t = 1;
err = [];
for gam = 0.1
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = [err; sum(Yht~=Ytest)];
end
%plot([0.01, 0.1, 0.5, 1,2,3,4, 5,6,7,8,9 10,25,50], err, 'x-')

%% 1.3.2 Radnom split
clc, clf, clear
load iris.mat
perf = [];
gam = 1000;
sig2 = [0.001,0.01,0.1,1,10,100, 1000];

for sigma = sig2
    perf = [perf; 1-rsplitvalidate({ Xtrain , Ytrain , 'c', gam , sigma ,'RBF_kernel'} , 0.80 , 'misclass')] ;
end


clf,clc
plot(log10(sig2), perf)
xlabel('$log{\sigma^2}$','Interpreter','latex')
ylabel('Validation accuracy')
title(['\gamma =', num2str(gam)])
axis([-3 3 0 1])
%% 1.3.2 10fold-CV
clc, clf, clear
load iris.mat
perf = [];
gam = 0.01; 
sig2 = [0.001,0.01,0.1,1,10,100, 1000];

for sigma = sig2
    perf = [perf; 1-crossvalidate({ Xtrain , Ytrain , 'c', gam , sigma ,'RBF_kernel'} , 10 , 'misclass')] ;
end


clf,clc
plot(log10(sig2), perf)
xlabel('$log{\sigma^2}$','Interpreter','latex')
ylabel('Validation accuracy')
title(['\gamma =', num2str(gam)])
axis([-3 3 0 1])

%% 1.3.2 leave-one-out
clc, clf, clear
load iris.mat
perf = [];
gam = 0.01;
sig2 = [0.001,0.01,0.1,1,10,100, 1000];

for sigma = sig2
    perf = [perf; 1-leaveoneout({ Xtrain , Ytrain , 'c', gam , sigma ,'RBF_kernel'} , 'misclass') ] ;
end


clf,clc
plot(log10(sig2), perf)
xlabel('$log{\sigma^2}$','Interpreter','latex')
ylabel('Validation accuracy')
title(['\gamma =', num2str(gam)])
axis([-3 3 0 1])

%% 1.3.3 gridsearch
c = [];
tic
for i=1:100
    [ gam , sig2 , cost ] = tunelssvm({ Xtrain , Ytrain , 'c', [] , [] ,'RBF_kernel'} , 'gridsearch', 'crossvalidatelssvm',{10 , 'misclass'}) ;
    c = [c; cost];
end
toc
sum(c)/100

%% 1.3.3 simplex
c = [];
tic
for i=1:100
    [ gam , sig2 , cost ] = tunelssvm({ Xtrain , Ytrain , 'f', [] , [] ,'RBF_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'mae'}) ;
    c = [c; cost];
end
toc
sum(c)/100

%% 1.3.4 ROC curve
% Train the classification model.
[ gam , sig2 , cost ] = tunelssvm({ Xtrain , Ytrain , 'c', [] , [] ,'RBF_kernel'} , 'gridsearch', 'crossvalidatelssvm',{10 , 'misclass'}) ;
[ alpha , b ] = trainlssvm({ Xtrain , Ytrain ,'c', gam , sig2 ,'RBF_kernel'}) ;
% Classification of the test data.
[ Yest , Ylatent ] = simlssvm({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'} , { alpha , b } , Xtest ) ;
% Generating the ROC curve.
roc( Ylatent , Ytest) ;

%% 1.3.5 Bayesian framework with tuned parameters

[ gam , sig2 , cost ] = tunelssvm({ Xtrain , Ytrain , 'c', [] , [] ,'RBF_kernel'} , 'gridsearch', 'crossvalidatelssvm',{10 , 'misclass'}) ;
bay_modoutClass({ Xtrain , Ytrain , 'c', gam , sig2 } , 'figure') ;
colorbar
%% 1.3.5 Bayesian framework with other parameters

gam = 0.05
sig2 = 2
bay_modoutClass({ Xtrain , Ytrain , 'c', gam , sig2 } , 'figure') ;
colorbar
%% Ripley data set %%
clear, clc, clf
load ripley.mat

%% test set
hold on;
plot ( Xtest (1:500 ,1) , Xtest (1:500 ,2) , 'r.','MarkerSize',12) ;
plot ( Xtest (501:end ,1) , Xtest(501:end ,2) , 'b.','MarkerSize',12) ;
title('Ripley test set')
hold off;
legend('Class 1', 'Class 2', 'location', 'northwest')
%% train set
clc, clf
hold on;
plot ( Xtrain (1:125 ,1) , Xtrain (1:125 ,2) , 'r.','MarkerSize',12) ;
plot ( Xtrain (126:end ,1) , Xtrain(126:end ,2) , 'b.','MarkerSize',12) ;
title('Ripley train set')
hold off;
legend('Class 1', 'Class 2', 'location', 'northwest')
%% Linear SVM
gam = tunelssvm({ Xtrain , Ytrain , 'c', [] , [] ,'lin_kernel'} , 'gridsearch', 'crossvalidatelssvm',{10 , 'misclass'}) ;
%%
clc, clf
type = 'c';
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'});

plotlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'}, {alpha,b}, Xtest);
roc( Zt , Ytest) ;
err = sum(Yht~=Ytest)/length(Ytest)

%% poly SVM
[gam, t_degree] = tunelssvm({ Xtrain , Ytrain , 'c', [] , [] ,'poly_kernel'} , 'gridsearch', 'crossvalidatelssvm',{10 , 'misclass'}) ;

clc, clf
type = 'c';
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,t_degree,'poly_kernel'});

plotlssvm({Xtrain,Ytrain,type,gam,t_degree,'poly_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,t_degree,'poly_kernel'}, {alpha,b}, Xtest);
roc( Zt , Ytest) ;
err = sum(Yht~=Ytest)/length(Ytest)

%% RBF SVM
[gam, sig2] = tunelssvm({ Xtrain , Ytrain , 'c', [] , [] ,'RBF_kernel'} , 'gridsearch', 'crossvalidatelssvm',{10 , 'misclass'}) ;

clc, clf
type = 'c';
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
roc( Zt , Ytest) ;
err = sum(Yht~=Ytest)/length(Ytest)
%% Breast cancer data set %%
clear, clc, clf
load breast.mat
Xtrain = trainset;
Ytrain = labels_train;

Xtest = testset;
Ytest = labels_test;
%% test set
hold on;
plot ( Xtest (1:500 ,1) , Xtest (1:500 ,2) , 'r.','MarkerSize',12) ;
plot ( Xtest (501:end ,1) , Xtest(501:end ,2) , 'b.','MarkerSize',12) ;
title('Ripley test set')
hold off;
legend('Class 1', 'Class 2', 'location', 'northwest')
%% train set
clc, clf
hold on;
plot ( Xtrain (1:125 ,1) , Xtrain (1:125 ,2) , 'r.','MarkerSize',12) ;
plot ( Xtrain (126:end ,1) , Xtrain(126:end ,2) , 'b.','MarkerSize',12) ;
title('Ripley train set')
hold off;
legend('Class 1', 'Class 2', 'location', 'northwest')
%% Linear SVM
gam = tunelssvm({ Xtrain , Ytrain , 'c', [] , [] ,'lin_kernel'} , 'gridsearch', 'crossvalidatelssvm',{10 , 'misclass'}) ;
%%
clc, clf
type = 'c';
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'});

plotlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'}, {alpha,b}, Xtest);
roc( Zt , Ytest) ;
err = sum(Yht~=Ytest)/length(Ytest)

%% poly SVM
[gam, t_degree] = tunelssvm({ Xtrain , Ytrain , 'c', [] , [] ,'poly_kernel'} , 'gridsearch', 'crossvalidatelssvm',{10 , 'misclass'}) ;

clc, clf
type = 'c';
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,t_degree,'poly_kernel'});

plotlssvm({Xtrain,Ytrain,type,gam,t_degree,'poly_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,t_degree,'poly_kernel'}, {alpha,b}, Xtest);
roc( Zt , Ytest) ;
err = sum(Yht~=Ytest)/length(Ytest)

%% RBF SVM
[gam, sig2] = tunelssvm({ Xtrain , Ytrain , 'c', [] , [] ,'RBF_kernel'} , 'gridsearch', 'crossvalidatelssvm',{10 , 'misclass'}) ;

clc, clf
type = 'c';
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
roc( Zt , Ytest) ;
err = sum(Yht~=Ytest)/length(Ytest)
%% Diabetes data set %%
clear, clc, clf
load diabetes.mat
Xtrain = trainset;
Ytrain = labels_train;

Xtest = testset;
Ytest = labels_test;
%% test set
hold on;
plot ( Xtest (1:500 ,1) , Xtest (1:500 ,2) , 'r.','MarkerSize',12) ;
plot ( Xtest (501:end ,1) , Xtest(501:end ,2) , 'b.','MarkerSize',12) ;
title('Ripley test set')
hold off;
legend('Class 1', 'Class 2', 'location', 'northwest')
%% train set
clc, clf
hold on;
plot ( Xtrain (1:125 ,1) , Xtrain (1:125 ,2) , 'r.','MarkerSize',12) ;
plot ( Xtrain (126:end ,1) , Xtrain(126:end ,2) , 'b.','MarkerSize',12) ;
title('Ripley train set')
hold off;
legend('Class 1', 'Class 2', 'location', 'northwest')
%% Linear SVM
gam = tunelssvm({ Xtrain , Ytrain , 'c', [] , [] ,'lin_kernel'} , 'gridsearch', 'crossvalidatelssvm',{10 , 'misclass'}) ;
%%
clc, clf
type = 'c';
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'});

plotlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'}, {alpha,b}, Xtest);
roc( Zt , Ytest) ;
err = sum(Yht~=Ytest)/length(Ytest)

%% poly SVM
[gam, t_degree] = tunelssvm({ Xtrain , Ytrain , 'c', [] , [] ,'poly_kernel'} , 'gridsearch', 'crossvalidatelssvm',{10 , 'misclass'}) ;

clc, clf
type = 'c';
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,t_degree,'poly_kernel'});

plotlssvm({Xtrain,Ytrain,type,gam,t_degree,'poly_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,t_degree,'poly_kernel'}, {alpha,b}, Xtest);
roc( Zt , Ytest) ;
err = sum(Yht~=Ytest)/length(Ytest)

%% RBF SVM
[gam, sig2] = tunelssvm({ Xtrain , Ytrain , 'c', [] , [] ,'RBF_kernel'} , 'gridsearch', 'crossvalidatelssvm',{10 , 'misclass'}) ;

clc, clf
type = 'c';
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
roc( Zt , Ytest) ;
err = sum(Yht~=Ytest)/length(Ytest)