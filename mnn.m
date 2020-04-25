%% XOR Problem
clear;
 
% Initialize Input and Expected Output Vectors
x1 = [-1 -1 1 1];
x2 = [-1 1 -1 1];
t = [-1 1 1 -1];

% Initialize Threshold and Learning Rate
eta = 0.01;
theta = 0.001;

% Initialize Weight Vectors
w_ij = rand(2, 3);
w_jk = rand(1, 3);

% Augment Input Data
x_aug = [ones(1, 4); x1; x2];

% Initialize Iterations Variable
epochs = 0;

error = zeros(1, 100);

z = zeros(1, 4);
delta_j = zeros(1, 2);

gradJ_jk = zeros(1, 3);
gradJ_ij = zeros(2, 3);

num_samples = length(x_aug(1, :));
num_features = 2;

while (1)
    epochs = epochs + 1;
    
    for i = 1:num_samples
        netj = w_ij*x_aug(:, i);
        y = tanh(netj);
        y_aug = [1; y];
        netk = w_jk * y_aug;
        z(i) = tanh(netk);

        delta_k = (t(i)-z(i))*(1-z(i).^2);
        
        for j = 1:num_features
           delta_j(j) = (1-y(j).^2) * delta_k *w_jk(j+1);
        end
        
        for j = 1:num_features+1
            for k = 1:num_features
                gradJ_ij(k, j) = gradJ_ij(k, j) + delta_j(k) * x_aug(j, i)';
            end
        end
        
        for j = 1:num_features+1
           gradJ_jk(j) = gradJ_jk(j) + delta_k * y_aug(j); 
        end
    end
    
    
    error(epochs) = (1/2)*((t-z)*(t-z)');
    
    if (epochs == 1)
        deltaJ = error(epochs);
    else
        deltaJ = abs(error(epochs) - error(epochs-1));
    end
    
    
    
    w_jk = w_jk + eta*gradJ_jk;
    w_ij = w_ij + eta*gradJ_ij;
    
    if (deltaJ < theta) 
       break; 
    end
end

figure(1)
plot(error)
title({"XOR Data Set Learning Curve (Error)"; sprintf("Convergence reached in %d iterations", epochs)});
xlabel("Number of Iterations");
ylabel("Error");
xlim([1 epochs])
ylim([0 max(error)])

% Begin Computation of Classification Accuracy
z = zeros(1, num_samples);
classification = zeros(1, num_samples);
num_correct = 0;
result = zeros(1, num_samples);

for i = 1:num_samples
    netj = w_ij*x_aug(:, i);
    y = tanh(netj);
    y_aug = [1; y];
    netk = w_jk * y_aug;
    z(i) = tanh(netk);
    
    if (z(i) > 0) 
       classification(i) = 1;
    else
       classification(i) = -1;
    end
    
    result(i) = classification(i) == t(i);
    if (result(i)) 
        num_correct = num_correct + 1;
    end
end
accuracy = (num_correct/num_samples)*100

%% Wine Data Set Classification
clear;
load wine.data
x1 = [wine(wine(:, 1) == 1, 2); wine(wine(:, 1) == 3, 2)];
x2 = [wine(wine(:, 1) == 1, 3); wine(wine(:, 1) == 3, 3)];
labels = [wine(wine(:,1) == 1, 1); wine(wine(:,1) == 3, 1)];

% Normalize and Augment the Input Data
x_normalized = normalize([x1 x2]);
data = [ones(length(x1), 1) x_normalized];
data(labels(:) == 3, 1:3) = data(labels(:) == 3, 1:3)*-1;
data_length = length(data(:,1));

% Setup data for processing
x_aug = data';
num_samples = length(x_aug(1, :));
w_ij = rand(2, 3);
w_jk = rand(1, 3);
t = x_aug(1, :);

z = zeros(1, num_samples);
gradJ_jk = zeros(1, 3);
gradJ_ij = zeros(2, 3);

eta = 0.01;
theta = 0.001;
epochs = 0;
num_features = 2;

while (1)
    epochs = epochs + 1;
    
    for i = 1:num_samples
        netj = w_ij*x_aug(:, i);
        y = tanh(netj);
        y_aug = [1; y];
        netk = w_jk * y_aug;
        z(i) = tanh(netk);

        delta_k = (t(i)-z(i))*(1-z(i).^2);
        
        for j = 1:num_features
           delta_j(j) = (1-y(j).^2) * delta_k *w_jk(j+1);
        end
        
        for j = 1:num_features+1
            for k = 1:num_features
                gradJ_ij(k, j) = gradJ_ij(k, j) + delta_j(k) * x_aug(j, i)';
            end
        end
        
        for j = 1:num_features+1
           gradJ_jk(j) = gradJ_jk(j) + delta_k * y_aug(j); 
        end
    end
    
    
    error(epochs) = (1/2)*((t-z)*(t-z)');
    
    if (epochs == 1)
        deltaJ = error(epochs);
    else
        deltaJ = abs(error(epochs) - error(epochs-1));
    end
    
    if (deltaJ < theta) 
       break; 
    end
    
    w_jk = w_jk + eta*gradJ_jk;
    w_ij = w_ij + eta*gradJ_ij;
end

figure(2)
plot(error)
title({"Wine Data Set Learning Curve (Error)"; sprintf("Convergence reached in %d iterations", epochs)});
xlabel("Number of Iterations");
ylabel("Error");
xlim([1 epochs])
ylim([0 max(error)])

% Begin Computation of Classification Accuracy
z = zeros(1, num_samples);
classification = zeros(1, num_samples);
num_correct = 0;
result = zeros(1, num_samples);

for i = 1:num_samples
    netj = w_ij*x_aug(:, i);
    y = tanh(netj);
    y_aug = [1; y];
    netk = w_jk * y_aug;
    z(i) = tanh(netk);
    
    if (z(i) > 0) 
       classification(i) = 1;
    else
       classification(i) = -1;
    end
    
    result(i) = classification(i) == t(i);
    if (result(i)) 
        num_correct = num_correct + 1;
    end
end

accuracy = (num_correct/num_samples)*100
t = t';
t_labels(t(:, 1) == 1, 1) = "w1";
t_labels(t(:, 1) == -1, 1) = "w2";

classification = classification';
class_labels(classification(:, 1) == 1, 1) = "w1";
class_labels(classification(:, 1) == -1, 1) = "w2";

result = result';
result_labels(result(:, 1) == 1, 1) = "Correct";
result_labels(result(:, 1) == 0, 1) = "Wrong";

t = table(t_labels, class_labels, result_labels);

filename = "wineclassificationtable.xlsx";
writetable(t,filename,'Sheet','Sheet1');