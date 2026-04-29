clc; clear; close all;
total_tic = tic;

tbl = readtable('HCC_survival_raw.csv','VariableNamingRule','preserve');
tbl = standardizeMissing(tbl, {'?', 'NA'});
data = table2array(tbl);

X_all = data(:,1:end-1);
Y_all = data(:,end);

num_features = size(X_all,2);
feature_freq = zeros(1,num_features);

outer_cv = cvpartition(Y_all,'KFold',5,'Stratify',true);

all_acc = zeros(5,1);
all_f1  = zeros(5,1);
all_feat = zeros(5,1);

model_acc_bag = zeros(5,1);
model_acc_boost = zeros(5,1);

selected_features_all = cell(5,1);

for fold = 1:5
    
    fprintf('\n===== OUTER FOLD %d =====\n', fold);
    
    trainIdx = training(outer_cv, fold);
    testIdx  = test(outer_cv, fold);
    
    [X_train, norm_params] = preprocess_train(X_all(trainIdx,:));
    X_test = preprocess_apply(X_all(testIdx,:), norm_params);
    
    Y_train = Y_all(trainIdx);
    Y_test  = Y_all(testIdx);
    
    %% Stage 1: Statistical Filtering
    pre_idx = StatisticalFiltering(X_train, Y_train, 30);
    
    %% Stage 2: HHO
    [best_mask, ~] = HHO_BEST(X_train(:,pre_idx), Y_train, 14, 18);
    
    final_idx = pre_idx(best_mask);
    
    if length(final_idx) < 12
        final_idx = pre_idx(1:18);
    end
    
    % Track feature usage
    feature_freq(final_idx) = feature_freq(final_idx) + 1;
    selected_features_all{fold} = final_idx;
    
    %% Class weights
    classWeights = ones(size(Y_train));
    pos_weight = sum(Y_train==0) / (sum(Y_train==1)+1e-6);
    classWeights(Y_train==1) = pos_weight;
    
    %% Train BOTH models
    t = templateTree('MaxNumSplits',20);
    
    bag = fitcensemble(X_train(:,final_idx), Y_train, ...
        'Method','Bag', ...
        'NumLearningCycles',180, ...
        'Learners',t, ...
        'Weights',classWeights);
    
    boost = fitcensemble(X_train(:,final_idx), Y_train, ...
        'Method','LogitBoost', ...
        'NumLearningCycles',180);
    
    %% Predictions
    pred_bag = predict(bag, X_test(:,final_idx));
    pred_boost = predict(boost, X_test(:,final_idx));
    
    %% Metrics
    stats_bag = compute_metrics(Y_test, pred_bag);
    stats_boost = compute_metrics(Y_test, pred_boost);
    
    model_acc_bag(fold) = stats_bag.Accuracy;
    model_acc_boost(fold) = stats_boost.Accuracy;
    
    %% Select best model per fold
    if stats_bag.F1 >= stats_boost.F1
        pred = pred_bag;
        best_model = bag;
        model_name = 'Bagging';
    else
        pred = pred_boost;
        best_model = boost;
        model_name = 'Boosting';
    end
    
    % fallback safety
    if isscalar(unique(pred))
        pred = mode(Y_train) * ones(size(Y_test));
    end
    
    stats = compute_metrics(Y_test, pred);
    
    all_acc(fold) = stats.Accuracy;
    all_f1(fold)  = stats.F1;
    all_feat(fold)= length(final_idx);
    
    fprintf(['Accuracy: %.2f%% | F1: %.2f%% | Features: %d | Best Model: %s\n'], ...
        stats.Accuracy*100, stats.F1*100, length(final_idx), model_name);
end

%% FINAL RESULTS
fprintf('\nFINAL RESULTS:\n');
fprintf('Accuracy: %.2f ± %.2f %%\n', mean(all_acc)*100, std(all_acc)*100);
fprintf('F1 Score: %.2f ± %.2f %%\n', mean(all_f1)*100, std(all_f1)*100);
fprintf('Avg Features: %.2f\n', mean(all_feat));

%% MODEL COMPARISON
fprintf('\nMODEL PERFORMANCE COMPARISON:\n');
fprintf('Bagging Avg Accuracy: %.2f%%\n', mean(model_acc_bag)*100);
fprintf('Boosting Avg Accuracy: %.2f%%\n', mean(model_acc_boost)*100);

%% WORST FOLD
[worst_acc, worst_fold] = min(all_acc);
fprintf('\nWORST FOLD ANALYSIS:\n');
fprintf('Fold %d had lowest accuracy: %.2f%%\n', worst_fold, worst_acc*100);

%% FEATURE IMPORTANCE
[sorted_freq, idx] = sort(feature_freq,'descend');

fprintf('\nTOP 5 MOST IMPORTANT FEATURES:\n');
disp(idx(1:5));
disp(sorted_freq(1:5));

fprintf('\nLEAST IMPORTANT FEATURES:\n');
disp(idx(end-5:end));

fprintf('\nFEATURE STABILITY (freq >=3):\n');
disp(find(feature_freq >= 3));

%% SAVE MODEL
save('Final_HCC_Model.mat', 'best_model', 'final_idx', 'norm_params');

fprintf('Time: %.2f sec\n', toc(total_tic));

function [best_mask, curve] = HHO_BEST(X, Y, N, T)

dim = size(X,2);

Rabbit = rand(1,dim) > 0.5;
BestFit = inf;

Pop = false(N,dim);
for i = 1:N
    Pop(i,:) = rand(1,dim) > 0.5;
end

curve = zeros(1,T);

for t = 1:T
    
    for i = 1:N
        
        mask = Pop(i,:);
        
        if sum(mask)==0
            mask(randi(dim)) = 1;
        end
        
        cv = cvpartition(Y,'KFold',3,'Stratify',true);
        scores = zeros(3,1);
        
        for k=1:3
            tr = training(cv,k);
            te = test(cv,k);
            
            mdl = fitcensemble(X(tr,mask), Y(tr), ...
                'Method','Bag','NumLearningCycles',100);
            
            pred = predict(mdl, X(te,mask));
            stats = compute_metrics(Y(te), pred);
            
            scores(k) = 0.8*stats.F1 + 0.2*stats.Accuracy;
        end
        
        fitness = 1 - mean(scores);
        
        feat_count = sum(mask);
        
        if feat_count < 12
            fitness = fitness + 0.08;
        elseif feat_count > 25
            fitness = fitness + 0.02;
        end
        
        if fitness < BestFit
            BestFit = fitness;
            Rabbit = mask;
        end
    end
    
    for i=1:N
        if rand < 0.6
            Pop(i,:) = Rabbit;
        else
            Pop(i,:) = rand(1,dim) > 0.5;
        end
    end
    
    curve(t)=BestFit;
end

best_mask = Rabbit;
end

function model = TrainStackingModel_FIXED(X, Y)

K = 5;
cv = cvpartition(Y,'KFold',K);

metaX = zeros(length(Y),3);

for k=1:K
    tr = training(cv,k);
    te = test(cv,k);
    
    rf = fitcensemble(X(tr,:),Y(tr),'Method','Bag','NumLearningCycles',100);
    svm = fitcsvm(X(tr,:),Y(tr),'KernelFunction','rbf','Standardize',true);
    boost = fitcensemble(X(tr,:),Y(tr),'Method','LogitBoost','NumLearningCycles',100);
    
    [~,score] = predict(rf,X(te,:));
    metaX(te,1) = score(:,2);

    [~,score] = predict(svm,X(te,:));
    metaX(te,2) = score(:,2);

    [~,score] = predict(boost,X(te,:));
    metaX(te,3) = score(:,2);
end

model.rf = fitcensemble(X,Y,'Method','Bag','NumLearningCycles',100);
model.svm = fitcsvm(X,Y,'KernelFunction','rbf','Standardize',true);
model.boost = fitcensemble(X,Y,'Method','LogitBoost','NumLearningCycles',100);

model.meta = fitcensemble(metaX,Y,'Method','Bag','NumLearningCycles',80);

end

function [best_mask, curve] = HHO_BEST(X, Y, N, T)

dim = size(X,2);

Rabbit = rand(1,dim) > 0.5;
BestFit = inf;

Pop = false(N,dim);
for i = 1:N
    Pop(i,:) = rand(1,dim) > 0.5;
end

curve = zeros(1,T);

for t = 1:T
    
    for i = 1:N
        
        mask = Pop(i,:);
        
        if sum(mask)==0
            mask(randi(dim)) = 1;
        end
        
        cv = cvpartition(Y,'KFold',3,'Stratify',true);
        scores = zeros(3,1);
        
        for k=1:3
            tr = training(cv,k);
            te = test(cv,k);
            
            mdl = fitcensemble(X(tr,mask), Y(tr), ...
                'Method','Bag','NumLearningCycles',100);
            
            pred = predict(mdl, X(te,mask));
            stats = compute_metrics(Y(te), pred);
            
            scores(k) = 0.8*stats.F1 + 0.2*stats.Accuracy;
        end
        
        fitness = 1 - mean(scores);
        
        feat_count = sum(mask);
        
        if feat_count < 12
            fitness = fitness + 0.08;
        elseif feat_count > 25
            fitness = fitness + 0.02;
        end
        
        if fitness < BestFit
            BestFit = fitness;
            Rabbit = mask;
        end
    end
    
    for i=1:N
        if rand < 0.6
            Pop(i,:) = Rabbit;
        else
            Pop(i,:) = rand(1,dim) > 0.5;
        end
    end
    
    curve(t)=BestFit;
end

best_mask = Rabbit;
end

clc; clear; close all;

%% Load saved model
load('Final_HCC_Model.mat');

%% Load dataset
tbl = readtable('HCC_survival_raw.csv','VariableNamingRule','preserve');
tbl = standardizeMissing(tbl, {'?', 'NA'});
data = table2array(tbl);

X_all = data(:,1:end-1);
Y_all = data(:,end);

%% Apply preprocessing (IMPORTANT)
X_all = preprocess_apply(X_all, norm_params);

%% Use selected features
X_selected = X_all(:, final_idx);

%% Predict using saved model
pred = predict(best_model, X_selected);

%% Compute confusion matrix
cm = confusionmat(Y_all, pred);

TP = cm(2,2);
FP = cm(1,2);
FN = cm(2,1);
TN = cm(1,1);

%% Metrics
accuracy = (TP + TN) / sum(cm(:));
precision = TP / (TP + FP + 1e-6);
recall = TP / (TP + FN + 1e-6);
f1 = 2*(precision*recall)/(precision+recall+1e-6);

%% Print results
fprintf('\nFINAL METRICS (FULL DATA):\n');
fprintf('Accuracy: %.2f%%\n', accuracy*100);
fprintf('Precision: %.2f%%\n', precision*100);
fprintf('Recall: %.2f%%\n', recall*100);
fprintf('F1 Score: %.2f%%\n', f1*100);

%% Confusion Matrix
disp('Confusion Matrix:');
disp(cm);