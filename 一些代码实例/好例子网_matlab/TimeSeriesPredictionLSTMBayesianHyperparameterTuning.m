clc; clear; close all;
%% ---------------------------- init Variabels ----------------------------
opt.Delays = 1:30;
opt.dataPreprocessMode  = 'Data Standardization'; % 'None' 'Data Standardization' 'Data Normalization'
opt.learningMethod      = 'LSTM';
opt.trPercentage        = 0.80;                   %  divide data into Test  and Train dataset

% ---- General Deep Learning Parameters(LSTM and CNN General Parameters)
opt.maxEpochs     = 400;                         % maximum number of training Epoch in deeplearning algorithms.
opt.miniBatchSize = 32;                         % minimum batch size in deeplearning algorithms .
opt.executionEnvironment = 'cpu';                % 'cpu' 'gpu' 'auto'
opt.LR                   = 'adam';               % 'sgdm' 'rmsprop' 'adam'
opt.trainingProgress     = 'none';  % 'training-progress' 'none'

% ------------- BILSTM parameters
opt.isUseBiLSTMLayer  = true;                     % if it is true the layer turn to the Bidirectional-LSTM and if it is false it will turn the units to the simple LSTM
opt.isUseDropoutLayer = true;                    % dropout layer avoid of bieng overfit
opt.DropoutValue      = 0.5;

% ------------ Optimization Parameters
opt.optimVars = [
    optimizableVariable('NumOfLayer',[1 4],'Type','integer')
    optimizableVariable('NumOfUnits',[50 200],'Type','integer')
    optimizableVariable('isUseBiLSTMLayer',[1 2],'Type','integer')
    optimizableVariable('InitialLearnRate',[1e-2 1],'Transform','log')
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];

opt.isUseOptimizer         = true;

opt.MaxOptimizationTime    = 14*60*60;
opt.MaxItrationNumber      = 60;
opt.isDispOptimizationLog  = true;

opt.isSaveOptimizedValue       = false;        %  save all of Optimization output on mat files 
opt.isSaveBestOptimizedValue   = true;         %  save Best Optimization output oد a mat file  


%% --------------- load Data
data = loadData(opt);
if ~data.isDataRead
    return;
end

%% --------------- Prepair Data
[opt,data] = PrepareData(opt,data);

%% --------------- Find Best LSTM Parameters with Bayesian Optimization
[opt,data] = OptimizeLSTM(opt,data);

%% --------------- Evaluate Data
[opt,data] = EvaluationData(opt,data);



%% ---------------------------- Local Functions ---------------------------
function data = loadData(opt)
[chosenfile,chosendirectory] = uigetfile({'*.xlsx';'*.csv'},...
    'Select Excel time series Data sets','data.xlsx');
filePath = [chosendirectory chosenfile];
if filePath ~= 0
    data.DataFileName = chosenfile;
    data.CompleteData = readtable(filePath);
    if size(data.CompleteData,2)>1
        warning('Input data should be an excel file with only one column!');
        disp('Operation Failed... '); pause(.9);
        disp('Reloading data. ');     pause(.9);
        data.x = [];
        data.isDataRead = false;
        return;
    end
    data.seriesdataHeder = data.CompleteData.Properties.VariableNames(1,:);
    data.seriesdata = table2array(data.CompleteData(:,:));
    disp('Input data successfully read.');
    data.isDataRead = true;
    data.seriesdata = PreInput(data.seriesdata);
    
    figure('Name','InputData','NumberTitle','off');
    plot(data.seriesdata); grid minor;
    title({['Mean = ' num2str(mean(data.seriesdata)) ', STD = ' num2str(std(data.seriesdata)) ];});
    if strcmpi(opt.dataPreprocessMode,'None')
        data.x = data.seriesdata;
    elseif strcmpi(opt.dataPreprocessMode,'Data Normalization')
        data.x = DataNormalization(data.seriesdata);
        figure('Name','NormilizedInputData','NumberTitle','off');
        plot(data.x); grid minor;
        title({['Mean = ' num2str(mean(data.x)) ', STD = ' num2str(std(data.x)) ];});
    elseif strcmpi(opt.dataPreprocessMode,'Data Standardization')
        data.x = DataStandardization(data.seriesdata);
        figure('Name','NormilizedInputData','NumberTitle','off');
        plot(data.x); grid minor;
        title({['Mean = ' num2str(mean(data.x)) ', STD = ' num2str(std(data.x)) ];});
    end
    
else
    warning(['In order to train network, please load data.' ...
        'Input data should be an excel file with only one column!']);
    disp('Operation Cancel.');
    data.isDataRead = false;
end
end
function data = PreInput(data)
if iscell(data)
    for i=1:size(data,1)
        for j=1:size(data,2)
            if strcmpi(data{i,j},'#NULL!')
                tempVars(i,j) = NaN; %#ok
            else
                tempVars(i,j) = str2num(data{i,j});   %#ok
            end
        end
    end
    data = tempVars;
end
end
function vars = DataStandardization(data)
for i=1:size(data,2)
    x.mu(1,i)   = mean(data(:,i),'omitnan');
    x.sig(1,i)  = std (data(:,i),'omitnan');
    vars(:,i) = (data(:,i) - x.mu(1,i))./ x.sig(1,i);
end
end
function vars = DataNormalization(data)
for i=1:size(data,2)
    vars(:,i) = (data(:,i) -min(data(:,i)))./ (max(data(:,i))-min(data(:,i)));
end
end
% --------------- data preparation for LSTM ---
function [opt,data] = PrepareData(opt,data)
% prepare delays for time serie network
data = CreateTimeSeriesData(opt,data);

% divide data into test and train data
data = dataPartitioning(opt,data);

% LSTM data form
data = LSTMInput(data);
end

% ----Run Bayesian Optimization Hyperparameters for LSTM Network Parameters
function [opt,data] = OptimizeLSTM(opt,data)
if opt.isDispOptimizationLog
    isLog = 2;
else
    isLog = 0;
end
if opt.isUseOptimizer
    opt.ObjFcn  = ObjFcn(opt,data);
    BayesObject = bayesopt(opt.ObjFcn,opt.optimVars, ...
        'MaxTime',opt.MaxOptimizationTime, ...
        'IsObjectiveDeterministic',false, ...
        'MaxObjectiveEvaluations',opt.MaxItrationNumber,...
        'Verbose',isLog,...
        'UseParallel',false);
end
end

% ---------------- objective function
function ObjFcn = ObjFcn(opt,data)
ObjFcn = @CostFunction;

function [valError,cons,fileName] = CostFunction(optVars)
inputSize    = size(data.X,1);
outputMode   = 'last';
numResponses = 1;
dropoutVal   = .5;

if optVars.isUseBiLSTMLayer == 2
    optVars.isUseBiLSTMLayer = 0;
end

if opt.isUseDropoutLayer % if dropout layer is true
    if optVars.NumOfLayer ==1
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif optVars.NumOfLayer==2
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif optVars.NumOfLayer ==3
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif optVars.NumOfLayer==4
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    end
else % if dropout layer is false
    if optVars.NumOfLayer ==1
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif optVars.NumOfLayer ==2
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif optVars.NumOfLayer ==3
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif optVars.NumOfLayer ==4
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    end
end
miniBatchSize    = opt.miniBatchSize;
maxEpochs        = opt.maxEpochs;
trainingProgress = opt.trainingProgress;
executionEnvironment = opt.executionEnvironment;
validationFrequency  = floor(numel(data.XTr)/miniBatchSize);

opt.opts = trainingOptions(opt.LR, ...
    'MaxEpochs',maxEpochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',optVars.InitialLearnRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'L2Regularization',optVars.L2Regularization, ...
    'Verbose',1, ...
    'MiniBatchSize',miniBatchSize,...
    'ExecutionEnvironment',executionEnvironment,...
    'ValidationData',{data.XVl,data.YVl}, ...
    'ValidationFrequency',validationFrequency,....
    'Plots',trainingProgress);
disp('LSTM architect successfully created.');

% --------------- Train Network
try
    data.BiLSTM.Net = trainNetwork(data.XTr,data.YTr,opt.layers,opt.opts);
    disp('LSTM Netwwork successfully trained.');
    data.IsNetTrainSuccess =true;
catch me
    disp('Error on Training LSTM Network');
    data.IsNetTrainSuccess = false;
    return;
end
close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_UIFIGURE'))


predict(data.BiLSTM.Net,data.XVl,'MiniBatchSize',opt.miniBatchSize);
valError = mse(predict(data.BiLSTM.Net,data.XVl,'MiniBatchSize',opt.miniBatchSize)-data.YVl);

Net  = data.BiLSTM.Net;
Opts = opt.opts;

fieldName = ['ValidationError' strrep(num2str(valError),'.','_')];
if ismember('OptimizedParams',evalin('base','who'))
    OptimizedParams =  evalin('base', 'OptimizedParams');
    OptimizedParams.(fieldName).Net  = Net;
    OptimizedParams.(fieldName).Opts = Opts;
    assignin('base','OptimizedParams',OptimizedParams);
else
    OptimizedParams.(fieldName).Net  = Net;
    OptimizedParams.(fieldName).Opts = Opts;
    assignin('base','OptimizedParams',OptimizedParams);
end

fileName = num2str(valError) + ".mat";
if opt.isSaveOptimizedValue
    save(fileName,'Net','valError','Opts')
end
cons = [];

end

end

% --------------- Train Network ---------------
% ---------------------------------------------
% make some delays on input filed
function data = CreateTimeSeriesData(opt,data)
Delays = opt.Delays;

x = data.x';
T = size(x,2);

MaxDelay = max(Delays);

Range = MaxDelay+1:T;

X= [];
for d = Delays
    X=[X; x(:,Range-d)];
end

Y = x(:,Range);
data.X  = X;
data.Y  = Y;
end

% partitioning input data
function data = dataPartitioning(opt,data)
data.XTr   = [];
data.YTr   = [];
data.XTs   = [];
data.YTs   = [];

numTrSample = round(opt.trPercentage*size(data.X,2));

data.XTr   = data.X(:,1:numTrSample);
data.YTr   = data.Y(:,1:numTrSample);

data.XTs   = data.X(:,numTrSample+1:end);
data.YTs   = data.Y(:,numTrSample+1:end);

disp(['Time Series data divided to ' num2str(opt.trPercentage*100) '% Train data and ' num2str((1-opt.trPercentage)*100) '% Test data']);
end

% Prepare input data for LSTM network.
function data = LSTMInput(data)

for i=1:size(data.XTr,2)
    XTr{i,1} = data.XTr(:,i);
    YTr(i,1) = data.YTr(:,i);
end

for i=1:size(data.XTs,2)
    XTs{i,1} =  data.XTs(:,i);
    YTs(i,1) =  data.YTs(:,i);
end
data.XTr   = XTr;
data.YTr   = YTr;
data.XTs   = XTs;
data.YTs   = YTs;
data.XVl   = XTs;
data.YVl   = YTs;

disp('Time Series data prepared as suitable LSTM Input data.');
end

% --------------- Evaluate Data ---------------
% ---------------------------------------------
function [opt,data] = EvaluationData(opt,data)
if opt.isUseOptimizer
    OptimizedParams =  evalin('base', 'OptimizedParams');
    % find best Net
    [valBest,indxBest] = sort(str2double(extractAfter(strrep(fieldnames(OptimizedParams),'_','.'),'Error')));
    data.BiLSTM.Net = OptimizedParams.(['ValidationError' strrep(num2str(valBest(1)),'.','_')]).Net;
    if opt.isSaveBestOptimizedValue
        fileName = ['BestNet ' num2str(valBest(1)) ' ' char(datetime('now','Format','yyyy.MM.dd HH.mm')) '.mat'];
        Net = data.BiLSTM.Net;
        save(fileName,'Net')
    end
elseif ~opt.isUseOptimizer
    [chosenfile,chosendirectory] = uigetfile({'*.mat'},...
    'Select Net File','BestNet.mat');
    if chosenfile==0
        error('Please Select saved Network File or set isUseOptimizer: true');
    end
    filePath = [chosendirectory chosenfile];
    Net = load(filePath);
    data.BiLSTM.Net = Net.Net;
end
   
data.BiLSTM.TrainOutputs = deNorm(data.seriesdata,predict(data.BiLSTM.Net,data.XTr,'MiniBatchSize',opt.miniBatchSize),opt.dataPreprocessMode);
data.BiLSTM.TrainTargets = deNorm(data.seriesdata,data.YTr,opt.dataPreprocessMode);
data.BiLSTM.TestOutputs  = deNorm(data.seriesdata,predict(data.BiLSTM.Net,data.XTs,'MiniBatchSize',opt.miniBatchSize),opt.dataPreprocessMode);
data.BiLSTM.TestTargets  = deNorm(data.seriesdata,data.YTs,opt.dataPreprocessMode);
data.BiLSTM.AllDataTargets = [data.BiLSTM.TrainTargets data.BiLSTM.TestTargets];
data.BiLSTM.AllDataOutputs = [data.BiLSTM.TrainOutputs data.BiLSTM.TestOutputs];

data = PlotResults(data,'Tr',...
    data.BiLSTM.TrainOutputs, ...
    data.BiLSTM.TrainTargets);
data = plotReg(data,'Tr',data.BiLSTM.TrainTargets,data.BiLSTM.TrainOutputs);

data = PlotResults(data,'Ts',....
    data.BiLSTM.TestOutputs, ...
    data.BiLSTM.TestTargets);
data = plotReg(data,'Ts',data.BiLSTM.TestTargets,data.BiLSTM.TestOutputs);

data = PlotResults(data,'All',...
    data.BiLSTM.AllDataOutputs, ...
    data.BiLSTM.AllDataTargets);
data = plotReg(data,'All',data.BiLSTM.AllDataTargets,data.BiLSTM.AllDataOutputs);

disp('Bi-LSTM network performance evaluated.');

end
function vars = deNorm(data,stdData,deNormMode)
if iscell(stdData(1,1))
    for i=1:size(stdData,1)
        tmp(i,:) = stdData{i,1}';
    end
    stdData = tmp;
end
if strcmpi(deNormMode,'Data Normalization')
    for i=1:size(data,2)
        vars(:,i) = (stdData(:,i).*(max(data(:,i))-min(data(:,i)))) + min(data(:,i));
    end
    vars = vars';
    
elseif strcmpi(deNormMode,'Data Standardization')
    for i=1:size(data,2)
        x.mu(1,i)   = mean(data(:,i),'omitnan');
        x.sig(1,i)  = std (data(:,i),'omitnan');
        vars(:,i) = ((stdData(:,i).* x.sig(1,i))+ x.mu(1,i));
    end
    vars = vars';
    
else
    vars = stdData';
    return;
end
end
% plot the output of networks and real output on test and train data
function data = PlotResults(data,firstTitle,Outputs,Targets)
Errors = Targets - Outputs;
MSE   = mean(Errors.^2);
RMSE  = sqrt(MSE);
NRMSE = RMSE/mean(Targets);
ErrorMean = mean(Errors);
ErrorStd  = std(Errors);
rankCorre = RankCorre(Targets,Outputs);

if strcmpi(firstTitle,'tr')
    Disp1Name = 'OutputGraphEvaluation_TrainData';
    Disp2Name = 'ErrorEvaluation_TrainData';
    Disp3Name = 'ErrorHistogram_TrainData';
elseif strcmpi(firstTitle,'ts')
    Disp1Name = 'OutputGraphEvaluation_TestData';
    Disp2Name = 'ErrorEvaluation_TestData';
    Disp3Name = 'ErrorHistogram_TestData';
elseif strcmpi(firstTitle,'all')
    Disp1Name = 'OutputGraphEvaluation_ALLData';
    Disp2Name = 'ErrorEvaluation_ALLData';
    Disp3Name = 'ErrorHistogram_AllData';
end

figure('Name',Disp1Name,'NumberTitle','off');
plot(1:length(Targets),Targets,...
    1:length(Outputs),Outputs);grid minor
legend('Targets','Outputs','Location','best') ;
title(['Rank Correlation = ' num2str(rankCorre)]);

figure('Name',Disp2Name,'NumberTitle','off');
plot(Errors);grid minor
title({['MSE = ' num2str(MSE) ', RMSE = ' num2str(RMSE)...
    ' NRMSE = ' num2str(NRMSE)] ;});
xlabel(['Error Per Sample']);

figure('Name',Disp3Name,'NumberTitle','off');
histogram(Errors);grid minor

title(['Error Mean = ' num2str(ErrorMean) ', Error StD = ' num2str(ErrorStd)]);
xlabel(['Error Histogram']);

if strcmpi(firstTitle,'tr')
    data.Err.MSETr = MSE;
    data.Err.STDTr = ErrorStd;
    data.Err.NRMSETr     = NRMSE;
    data.Err.rankCorreTr = rankCorre;
elseif strcmpi(firstTitle,'ts')
    data.Err.MSETs = MSE;
    data.Err.STDTs = ErrorStd;
    data.Err.NRMSETs     = NRMSE;
    data.Err.rankCorreTs = rankCorre;
elseif strcmpi(firstTitle,'all')
    data.Err.MSEAll = MSE;
    data.Err.STDAll = ErrorStd;
    data.Err.NRMSEAll     = NRMSE;
    data.Err.rankCorreAll = rankCorre;
end
end
% find rank correlation between network output and real data
function [r]=RankCorre(x,y)
x=x';
y=y';
% Find the data length
N = length(x);
% Get the ranks of x
R = crank(x)';
for i=1:size(y,2)
    % Get the ranks of y
    S = crank(y(:,i))';
    % Calculate the correlation coefficient
    r(i) = 1-6*sum((R-S).^2)/N/(N^2-1); %#ok
end
end
function r=crank(x)
u = unique(x);
[~,z1] = sort(x);
[~,z2] = sort(z1);
r = (1:length(x))';
r=r(z2);
for i=1:length(u)
    s=find(u(i)==x);
    r(s,1) = mean(r(s));
end
end
% plot the regression line of output and real value
function data = plotReg(data,Title,Targets,Outputs)

if strcmpi(Title,'tr')
    DispName = 'RegressionGraphEvaluation_TrainData';
elseif strcmpi(Title,'ts')
    DispName = 'RegressionGraphEvaluation_TestData';
elseif strcmpi(Title,'all')
    DispName = 'RegressionGraphEvaluation_ALLData';
end
figure('Name',DispName,'NumberTitle','off');
x = Targets';
y = Outputs';
format long
b1 = x\y;
yCalc1 = b1*x;
scatter(x,y,'MarkerEdgeColor',[0 0.4470 0.7410],'LineWidth',.7);
hold('on');
plot(x,yCalc1,'Color',[0.8500 0.3250 0.0980]);
xlabel('Prediction');
ylabel('Target');
grid minor
% xgrid = 'on';
% disp.YGrid = 'on';
X = [ones(length(x),1) x];
b = X\y;
yCalc2 = X*b;
plot(x,yCalc2,'-.','MarkerSize',4,"LineWidth",.1,'Color',[0.9290 0.6940 0.1250])
legend('Data','Fit','Y=T','Location','best');
%
Rsq2 = 1 -  sum((y - yCalc1).^2)/sum((y - mean(y)).^2);

if strcmpi(Title,'tr')
    data.Err.RSqur_Tr = Rsq2;
    title(['Train Data, R^2 = ' num2str(Rsq2)]);
elseif strcmpi(Title,'ts')
    data.Err.RSqur_Ts = Rsq2;
    title(['Test Data, R^2 = ' num2str(Rsq2)]);
elseif strcmpi(Title,'all')
    data.Err.RSqur_All = Rsq2;
    title(['All Data, R^2 = ' num2str(Rsq2)]);
end

end

