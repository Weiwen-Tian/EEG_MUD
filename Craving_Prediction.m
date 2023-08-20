function [ypred,rmse, model] = Craving_Prediction(data,ytrue, repnum, cvnum, titlename)
% Perform cross-validation on craving prediction using RVM
%   
y = ytrue';
N = length(ytrue);
            
%% 10x10 cross validation
rand('seed', 10);
model = repmat(struct('w', []), [repnum, cvnum]);
ypred_rep = zeros(repnum, N);
fprintf('Starting %dx%d cross-validation for condition %s \n', ...
repnum, cvnum, titlename);
for rep = 1:repnum
    rng(112054+rep, 'twister');
    randN = randperm(N);
    cvidx = floor(linspace(0, N, cvnum + 1));
    for cv = 1:cvnum
        fprintf('%dx%d run \n', rep, cv);
        testind = randN(cvidx(cv) + 1:cvidx(cv + 1));
        trainind = setdiff(1:N, testind);

        X_train = data(trainind,:);
        X_test = data(testind,:);
        y_train = y(trainind);

        NN = size(X_train, 1);
        MM = size(X_train, 2);
        BASIS = [ones(NN, 1), X_train];
        OPTIONS = SB2_UserOptions('freeBasis', 1);
        [PARAMETER, HYPERPARAMETER, ~] = SparseBayes('Gaussian', BASIS, y_train, OPTIONS);
        w_infer						= zeros(MM + 1, 1);
        w_infer(PARAMETER.Relevant)	= PARAMETER.Value;
        model(rep, cv).w = w_infer;
                    
        %% Test the model
        y_pred=model(rep, cv).w'*[ones(size(X_test, 1), 1) X_test]';

        for index = 1:length(testind)
            ypred_rep(rep,testind(index)) = y_pred(index);
        end
    end
end
ypred = median(ypred_rep, 1);
rmse = sqrt(mean((ypred - ytrue).^2));

end

