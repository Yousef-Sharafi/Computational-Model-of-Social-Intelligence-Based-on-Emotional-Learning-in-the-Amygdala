% paper: Computational model of social intelligence based on emotional learning in the Amygdala
% Authors: Alireza Falahiazar, Saeed Setayeshi, Yousef Sharafi
% link paper: http://www.tjmcs.com/includes/files/articles/Vol14_Iss1_77%20-%2086_Computational_Model_of_Social_Intel.pdf
% Journal of mathematics and computer Science 14(2015)77-86.
function main
clc;
close all;
clear all;
global depth;
depth = 2;
train_size = 0.3;
[data_input target] = Read_Data();
data_train = data_input(1:ceil(train_size*size(data_input,1)),:);
target_train = target(1:ceil(train_size*size(data_input,1)));
data_test = data_input(ceil(train_size*size(data_input,1))+1:size(data_input,1),:);
target_test = target(ceil(train_size*size(data_input,1))+1:size(data_input,1));

Pattern_number = numel(target_train);
Dimension = size(data_input,2)
eta = 0.1;
n1 = depth;
n2 = depth;
Population = 1;
v = unifrnd(-1,+1,Population,n1);
w = unifrnd(-1,+1,Population,n2);
w_bel = unifrnd(-1,+1,Population,1);
E = zeros(Population,1);
e = ones(Pattern_number,1);

temp_v = v;
temp_w = w;
temp_w_bel = w_bel;
fail =0;


E_old = mse(SMBL(v,w,w_bel,data_train,target_train,Pattern_number,Population));    

for c=0:1000


    
[v,w,w_bel] = Update(v,w,w_bel,eta,data_train,target_train,Pattern_number,Population);

E_new = mse(SMBL(v,w,w_bel,data_train,target_train,Pattern_number,Population));    
if E_new >= E_old || isnan(E_new)
    eta = eta / 10;
    v = temp_v;
    w = temp_w;
    w_bel = temp_w_bel;

else
    temp_v = v;
    temp_w = w;
    temp_w_bel = w_bel;
    E_old = E_new;
    display(E_old);
end

end

%% Show Result
[error,out] = SMBL(v,w,w_bel,data_train,target_train,Pattern_number,Population);    

figure('name','Social Brain Emotional Learning for estimate Mackey-Glass Time Series');
subplot(1,2,1);
plot(target_train','-b');
hold on;
plot(out','-r');
ylabel('Mackey-Glass Time Series');
title(strcat('MSE-Train=',num2str(mse(error))));
legend('Real Value','estimate Value','Location','NW');

[error,out] = SMBL(v,w,w_bel,data_test,target_test,numel(target_test),Population);    
subplot(1,2,2);
plot(target_test','-b');
hold on;
plot(out','-r');
ylabel('Mackey-Glass Time Series');
title(strcat('MSE-Test=',num2str(mse(error))));
legend('Real Value','estimate Value','Location','NW');




 net = newff(data_train',target_train',[40],{'tansig'},'traingd');
 net.trainParam.epochs = 1000;
 net = train(net,data_train',target_train');
 Y1 = sim(net,data_train');
 Y2 = sim(net,data_test');
figure('name','feed-forward backpropagation network for estimate Mackey-Glass Time Series');
subplot(1,2,1); 
plot(target_train','-b');
hold on;
plot(Y1','-r');
ylabel('Mackey-Glass Time Series');
title(strcat('MSE-Train=',num2str(mse(target_train-Y1'))));
legend('Real Value','estimate Value','Location','NW');

subplot(1,2,2);
plot(target_test','-b');
hold on;
plot(Y2','-r');
ylabel('Mackey-Glass Time Series');
title(strcat('MSE-Test=',num2str(mse(target_test-Y2'))));
legend('Real Value','estimate Value','Location','NW');


 net = newelm(data_train',target_train',[40],{'tansig'},'traingd');
 net.trainParam.epochs = 1000;
 net = train(net,data_train',target_train');
 Y1 = sim(net,data_train');
 Y2 = sim(net,data_test');

figure('name','Elman backpropagation network for estimate Mackey-Glass Time Series');
subplot(1,2,1); 
plot(target_train','-b');
hold on;
plot(Y1','-r');
ylabel('Mackey-Glass Time Series');
title(strcat('MSE-Train=',num2str(mse(target_train-Y1'))));
legend('Real Value','estimate Value','Location','NW');

subplot(1,2,2);
plot(target_test','-b');
hold on;
plot(Y2','-r');
ylabel('Mackey-Glass Time Series');
title(strcat('MSE-Test=',num2str(mse(target_test-Y2'))));
legend('Real Value','estimate Value','Location','NW');

end
%% Train Weights of SMBL(Computational model of social intelligence based on emotional learning in the Amygdala)
function [v,w,w_bel]=Update(v,w,w_bel,eta,data_train,target_train,Pattern_number,Population)

e = ones(Pattern_number,1);
E = zeros(Population,1);
for i=1:Pattern_number
    for j=1:Population
        E(j) = ( ( data_train(i,:) * v(j,:)') + max(data_train(i,:),[],2) ) - (data_train(i,:) * w(j,:)');
    end
    e(i) = target_train(i) - ( E' * w_bel );
    
    for j=1:Population
        for k=1:size(v,2)
            v(j,k) = v(j,k) - eta * (-e(i) * w_bel(j) * data_train(i,k));
            w(j,k) = w(j,k) - eta * (+e(i) * w_bel(j) * data_train(i,k));
        end
        w_bel(j) = w_bel(j) - eta * (-e(i)*E(j));
    end
    
end

end
%% Procedure of SMBL(Computational model of social intelligence based on emotional learning in the Amygdala)
function [error,out]=SMBL(v,w,w_bel,data_train,target_train,Pattern_number,Population)

error = ones(Pattern_number,1);
E = zeros(Population,1);
out = zeros(size(target_train,1),1);
for i=1:Pattern_number    
    for j=1:Population
        E(j) = ( ( data_train(i,:) * v(j,:)') + max(data_train(i,:),[],2) ) - (data_train(i,:) * w(j,:)');
    end
    out(i) = ( E' * w_bel );
    error(i) = target_train(i) - out(i);
end

end

%% Read Mackey-Glass Data and Fill Inputs and outputs(desired) vector for Training
function [data_input output]=Read_Data()
global depth;
data = xlsread('Mackey-Glass.xlsx');
data_input = zeros((numel(data)-depth),depth);
output = zeros((numel(data)-depth),1);

data_input = zeros((numel(data)-depth),depth);
output = zeros((numel(data)-depth),1);

j =1;
for i=1:(numel(data)-depth)
    data_input(i,1:depth) = data(j:j+depth-1);
    j = j+1;
end
j= depth+1;
for i=1:size(data_input,1)
output(i) = data(j);
j = j + 1;
end

end