classdef SANS < ALGORITHM
    %------------------------------- Reference --------------------------------
    % H. Gu, H. Wang, Y. Mei, M. Zhang and Y. Jin, "Surrogate-Assisted Neighborhood Search With Only a Few Weight Vectors for Expensive
    % Large-Scale Multiobjective Binary Optimization," in IEEE Transactions on Evolutionary Computation, doi: 10.1109/TEVC.2024.3512795.
    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            warning off
            Population = Problem.Initialization();
            Arc = Population;
            TrainData = [Population.decs,Population.objs];
            Problem.N = 100;
            if Problem.M == 2
                [W1,~] = UniformPoint(3,Problem.M);
            else
                [W1,~] = UniformPoint(6,Problem.M);
                W1 = [W1;[0.58,0.58,0.58]];   % 1/(3)^(1/2)
            end
            W = [];
            
            Q = zeros(2,size(W1,1));
            Z = min(Population.objs,[],1);
            flag4 = 0;
            alp = 0.1;
            gam = 0.95;
            epsilon = 0.1;
            localtraindata = cell(size(W1,1),1);
            for i = 1:size(W1,1)
                giniall(:,i) = max(abs(Population.objs-repmat(Z,Problem.N,1)).*repmat(W1(i,:),Problem.N,1),[],2);
            end
            [~,posini] = min(giniall);
            for i = 1:size(W1,1)
                if isempty(localtraindata{i})
                    aa = 0;
                else
                    aa = length(localtraindata{i});
                end
                localtraindata{i}(aa+1) = Population(posini(i));
            end
            %% Optimization
            while Algorithm.NotTerminated(Arc)
                
                RBFModel = [];
                pair          = pdist2(TrainData, TrainData);
                D_max         = max(max(pair, [], 2));
                spread        = D_max * (Problem.D * Problem.N) ^ (-1 / Problem.D);
                for m = 1:Problem.M
                    RBFModel{m}      = newrbe(transpose( TrainData(:,1:Problem.D)), transpose(TrainData(:,Problem.D+m)), spread);
                end
                if length(Arc)==Problem.N
                    for i = 1:Problem.N
                        b = randperm(size(W1,1));
                        W(i,:) = W1(b(1),:);
                        train_Q2(i) = b(1);
                    end
                end
                for i = 1 : Problem.N
                    if length(Arc)>=200
                        [~,index1] = max(Q,[],2);
                        for j = 1:2
                            if isempty(find(Q(j,:)~=0, 1))
                                col1 = randperm(size(W1,1));
                                index1(j) = col1(1);
                            end
                        end
                        
                        if rand > epsilon
                            if flag4 ==1
                                test_Q2(i) = index1(2);
                            else
                                test_Q2(i) = index1(1);
                            end
                        else
                            c = randperm(size(W1,1));
                            test_Q2(i) = c(1);
                        end
                        W(i,:) = W1(test_Q2(i),:);
                    end
                    for k  = 1:size(W1,1)
                        if all(W(i,:) == W1(k,:))
                            wk = k;
                        end
                    end
                    for m = 1:Problem.M
                        if length(localtraindata{wk})>20
                            if ~isempty(localtraindata{wk})
                                traobj =   localtraindata{wk}.objs;
                                RBFModellocl{m}      = newrbe(transpose( localtraindata{wk}.decs), transpose(traobj(:,m)), spread);
                            end
                        end
                    end
                    Parent = Arc.decs;
                    parobj = Arc.objs;
                    
                    
                    gpar = max(abs(parobj-repmat(Z,length(Arc),1)).*repmat(W(i,:),length(Arc),1),[],2);
                    [~,so1] = sort(gpar);
                    Parentgood = Parent(so1(1:length(Arc)/2,1),:);
                    Parentbad = Parent(so1(Problem.N/2+1:end,1),:);
                    
                    off1 = [];
                    objoff1 = [];
                    off1 = repmat(Parentgood(1,:),1,1);
                    
                    flag = 1;
                    objoff1 = [];
                    
                    %% VNS
                    offtemp = off1;
                    while flag <=10
                        for p = 1:20
                            if flag ==1 && p ==1
                                offtemp1 = repmat(offtemp,100,1);
                                for n = 1:100
                                    b = randperm(Problem.D);
                                    offtemp1(n,b(1:10)) = 1-offtemp1(n,b(1:10));
                                end
                                offtemp1 = OperatorGA(Problem,offtemp1);
                                
                                for m = 1:Problem.M
                                    objtemp1(:,m) = transpose(sim(RBFModel{m}, transpose(offtemp1)));
                                    if length(localtraindata{wk})>20
                                        if ~isempty(localtraindata{wk})
                                            objoff1localRBF(:,m) = transpose(sim(RBFModellocl{m}, transpose(offtemp1)));
                                        end
                                    end
                                end
                                if ~isempty(localtraindata{wk})
                                    if length(localtraindata{wk})>20
                                        objmodellocal = objoff1localRBF;
                                        glocal = max(abs(objmodellocal-repmat(Z,Problem.N,1)).*repmat(W(i,:),Problem.N,1),[],2);
                                        [~,poslocal] = sort(glocal);
                                    end
                                end
                                gtemp = max(abs(objtemp1-repmat(Z,100,1)).*repmat(W(i,:),100,1),[],2);
                                [~,pos] = sort(gtemp);
                                for k = 1:100
                                    rank1(pos(k)) = k;
                                    if length(localtraindata{wk})>20
                                        rank2(poslocal(k)) = k;
                                    else
                                        rank2(k) = 0;
                                    end
                                end
                                rankall = rank1+rank2;
                                [~,posmin] = min(rankall);
                                g = gtemp(posmin);
                                offtemp = offtemp1(posmin,:);
                                gori = g;
                            else
                                a = randperm(Problem.D);
                                offtemp(:,a(1)) = 1 - off1(:,a(1));
                                for m = 1:Problem.M
                                    objoff1(:,m)=transpose(sim(RBFModel{m}, transpose(offtemp)));
                                end
                                g = max(abs(objoff1-repmat(Z,1,1)).*repmat(W(i,:),1,1),[],2);
                            end
                            if g <= gori
                                flag = 1 + flag;
                                gori = g;
                                off1 = offtemp;
                                break
                            end
                        end
                        if p ==20
                            break
                        end
                    end
                    offtemp = off1;
                    flag = 1;
                    objoff1 = [];
                    
                    while flag <=10
                        for p = 1:20
                            a = randperm(Problem.D);
                            offtemp(:,a(1:2)) = 1 - off1(:,a(1:2));
                            for m = 1:Problem.M
                                objoff1(:,m)=transpose(sim(RBFModel{m}, transpose(offtemp)));
                            end
                            g = max(abs(objoff1-repmat(Z,1,1)).*repmat(W(i,:),1,1),[],2);
                            if g <= gori
                                flag = 1 + flag;
                                gori = g;
                                off1 = offtemp;
                                break
                            end
                        end
                        if p ==20
                            break
                        end
                    end
                    
                    offfinal = off1;
                    Offspring = Problem.Evaluation(offfinal);
                    Z1 =  Z;
                    Z = min(Z,Offspring.obj);
                    if length(Arc)<=200
                        % train
                        if ~isempty(find(Z~=Z1, 1))
                            if flag4 ==1
                                Q(2,train_Q2(i)) = (1-alp)*Q(2,train_Q2(i)) + alp*(2+gam*max(Q(2,:)));
                            else
                                Q(1,train_Q2(i)) = (1-alp)*Q(1,train_Q2(i)) + alp*(2+gam*max(Q(2,:)));
                            end
                            flag4 = 1;
                        else
                            if flag4 ==1
                                Q(2,train_Q2(i)) = (1-alp)*Q(2,train_Q2(i)) + alp*(-0.5+gam*max(Q(1,:)));
                            else
                                Q(1,train_Q2(i)) = (1-alp)*Q(1,train_Q2(i)) + alp*(-0.5+gam*max(Q(1,:)));
                            end
                            flag4 = 0;
                        end
                    else
                        
                        if ~isempty(find(Z~=Z1, 1))
                            if flag4 ==1
                                Q(2,test_Q2(i)) = (1-alp)*Q(2,test_Q2(i)) + alp*(2+gam*max(Q(2,:)));
                            else
                                Q(1,test_Q2(i)) = (1-alp)*Q(1,test_Q2(i)) + alp*(2+gam*max(Q(2,:)));
                            end
                            flag4 = 1;
                        else
                            if flag4 ==1
                                Q(2,test_Q2(i)) = (1-alp)*Q(2,test_Q2(i)) + alp*(-0.5+gam*max(Q(1,:)));
                            else
                                Q(1,test_Q2(i)) = (1-alp)*Q(1,test_Q2(i)) + alp*(-0.5+gam*max(Q(1,:)));
                            end
                            flag4 = 0;
                        end
                    end
                    Arc = [Arc, Offspring];
                    Algorithm.NotTerminated(Arc);
                    for k  = 1:size(W1,1)
                        if all(W(i,:) == W1(k,:))
                            if isempty(localtraindata{k})
                                aa = 0;
                            else
                                aa = length(localtraindata{k});
                            end
                            localtraindata{k}(aa+1) = Offspring;
                        end
                    end
                end
                
                TrainData = [Arc.decs,Arc.objs];
            end
        end
    end
end