clear
nfolds = 5;
MU = 0.005;GAMMA=0.002;

% load adjacency matrix
dataname = 'Pauwels_dataset.mat';
load(dataname);dataname

y = side_effect;
[n,m]=size(y);

lambda_list = [1,4,4,2,2;
                4,8,8,4,0.25;
                8,8,16,4,16;
                1,2,2,1,0.5;
                4,16,8,2,2];
aupr_list = [];
auc_list = [];
nruns=10;
for run=1:nruns
    % split folds
        data_ntk_name = ['pau_ntk_' int2str(run-1) '.mat'];
        load(data_ntk_name)
        
        crossval_idx = cv_index+1;
        crossval_idx = reshape(crossval_idx,[m,n]);
        crossval_idx = crossval_idx';

        results = zeros(n,m);
        fold_aupr = [];
        fold_auc = [];
        for fold=1:nfolds
            train_idx = find(crossval_idx~=fold);
            test_idx  = find(crossval_idx==fold);
            y_train = y;
            y_train(test_idx) = 0;
            
           % kernel
            %GIP
            K1_list(:,:,1) = kernel_gip(y_train,1, 1);
            K2_list(:,:,1) = kernel_gip(y_train,2, 1);
            %Cos
            K1_cos = kernel_cosine(y_train,1,MU,GAMMA);
            K1_list(:,:,2) = Knormalized(K1_cos);
            K2_cos = kernel_cosine(y_train,2,MU,GAMMA);
            K2_list(:,:,2) = Knormalized(K2_cos);
            %Corr
            K1_list(:,:,3) = kernel_corr(y_train,1,MU,GAMMA);
            K2_list(:,:,3) = kernel_corr(y_train,2,MU,GAMMA);
            %MI
            K1_list(:,:,4) = cal_mi_m(y_train,1);
            K2_list(:,:,4) = cal_mi_m(y_train,2);
            %NTK
            K1_ntk = ntk_kernel_list{fold,1};
            K1_list(:,:,5) = Knormalized(K1_ntk);
            K2_ntk = ntk_kernel_list{fold,3};
            K2_list(:,:,5) = Knormalized(K2_ntk);
            
            K1_knn_list(:,:,1) = neighborhood_Com(Knormalized(K1_list(:,:,1)),17);
            K1_knn_list(:,:,2) = neighborhood_Com(Knormalized(K1_list(:,:,2)),17);
            K1_knn_list(:,:,3) = neighborhood_Com(Knormalized(K1_list(:,:,3)),17);
            K1_knn_list(:,:,4) = neighborhood_Com(Knormalized(K1_list(:,:,4)),17);
            K1_knn_list(:,:,5) = neighborhood_Com(Knormalized(K1_list(:,:,5)),17);
            
            K2_knn_list(:,:,1) = neighborhood_Com(Knormalized(K2_list(:,:,5)),17);
            K2_knn_list(:,:,2) = neighborhood_Com(Knormalized(K2_list(:,:,5)),17);
            K2_knn_list(:,:,3) = neighborhood_Com(Knormalized(K2_list(:,:,5)),17);
            K2_knn_list(:,:,4) = neighborhood_Com(Knormalized(K2_list(:,:,5)),17);
            K2_knn_list(:,:,5) = neighborhood_Com(Knormalized(K2_list(:,:,5)),17);
            % train
            iteration = 1;
            e = 2;
            sita = 2.^[-8];
            beta = 2.^[0];
            miu = 2.^[-7];
            [pre_comm] = Mv_weighted_kronrls_lap(y_train,K1_list,K2_list,K1_knn_list,K2_knn_list,lambda_list,miu,beta,sita,e,iteration);
            %evalulate
            results(test_idx) = pre_comm(test_idx);
            [~,~,~,aupr] = perfcurve(y(test_idx),pre_comm(test_idx),1, 'xCrit', 'reca', 'yCrit', 'prec');
            [~,~,~,auc,~,~,~] = perfcurve(y(test_idx),pre_comm(test_idx),1);
            fold_aupr = [fold_aupr,aupr];
            fold_auc = [fold_auc,auc];
            fprintf('%d - FOLD %d - AUPR: %f \n', run, fold, aupr)
            fprintf('%d - FOLD %d - AUC: %f \n', run, fold, auc)
        end
        mean_aupr = mean(fold_aupr);
        mean_auc = mean(fold_auc);
        aupr_list = [aupr_list,mean_aupr];
        auc_list = [auc_list,mean_auc];
end

 