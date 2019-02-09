%%
subplot(2,1,1);
plot(W.Data{2}');
title('Subplot 1')

subplot(2,1,2);

A = xcorr(LAData.Data{1}, W.Data{2});
for i = [2:9 39:47 77:85 115:123 153:166 226:239 299:312 372:385]
    A = A + xcorr(LAData.Data{i}, W.Data{2});
end
A = A ./ numel([1:9 39:47 77:85 115:123 153:166 226:239 299:312 372:385]);

plot(A.^5);
title('Subplot 2')

%%
item = 2;
plot(LAData.Data{item + 36 * 0}')
hold on
plot(LAData.Data{item + 36 * 1}')
plot(LAData.Data{item + 36 * 2}')
plot(LAData.Data{item + 36 * 3}')
hold off

%% Variables
parentDir = '..\';
dataDir = 'Data\2 - Infinity\';
fileNames =    {'frontleft-lac.csv';  'frontright-lac.csv';...
                'backleft-lac.csv';   'backright-lac.csv';...
                'frontleft2-lac.csv'; 'frontright2-lac.csv';...
                'backleft2-lac.csv';  'backright2-lac.csv'};
windowFilename1 = '1543157423545-FrontLeft-laData.csv';
windowFilename2 = '1543162135568-FrontLeft-laData.csv';
cwtDir = 'svmdata';
            
% SK desired sampling frequency setting
Fs = 50;
avg = 0.5;
fltr = 15;
% [2] 0, 1: driver, passenger
% [4] 0, 1, 2, 3: FL, FR, BL, BR
numClasses = 4;
% [noseg] No segmentation
% [cut] Remove first and last few seconds
% [segfixed] Segmentation (fixed intervals)
% [segfixedavg] Segmentation (fixed intervals) + averaging
% [segfixedfltr] Segmentation (fixed intervals) + filtering
% [segtrb] Segmentation (turns and road bumps)
% [segtrbavg] Segmentation (turns and road bumps) + averaging
% [segtrbfltr] Segmentation (turns and road bumps) + filtering
segMethod = 'segfixedfltrcom\AAA';
cwtDir = [cwtDir '-' segMethod];
expandBefore = 0.2;
expandAfter = 1.5;

%% Load data and labels
%
% 1. Data should be in a structure array with 2 fields
%
%   * Data: each row is a sample, columns are sampling data
%   * Labels: each row is the label for its corresponding Data row class
%
% 2. Data should be sampled at uniform intervals
%
% 3. Create “svmdata” folder, inside it, there should be a folder for each
% class
%

% Classes are FL, FR, BL, BR

W = helperLoadRowsNoSeg(parentDir, dataDir, fileNames, Fs, avg, fltr, numClasses);

LAData = helperLoadRowsFixed(parentDir, dataDir, fileNames, Fs, avg, fltr, numClasses, 5);
% LAData = helperLoadRowsTRB(parentDir, dataDir, fileNames, Fs, avg, fltr, numClasses, windowFilename1, windowFilename2, expandBefore, expandAfter);

% helperCreateCWTFolders(LAData, parentDir, cwtDir);


%% Apply CWT to get time-frequency data
%
%
%

II{1} = 1:368;     OO{1} = 368; CC{1} = 'AAA';
II{2} = 1473:2010; OO{2} = 538; CC{2} = 'AAA';

% II{1} = 1:9;     OO{1} = 36; CC{1} = 'LT';
% II{2} = 145:158; OO{2} = 71; CC{2} = 'LT';
% II{3} = 10:21;   OO{3} = 36; CC{3} = 'RT';
% II{4} = 159:176; OO{4} = 71; CC{4} = 'RT';
% II{5} = 22:36;   OO{5} = 36; CC{5} = 'RB';
% II{6} = 177:215; OO{6} = 71; CC{6} = 'RB';
helperGenerateAndSaveCombinedCWT(LAData, Fs, 224, parentDir, cwtDir, II, OO, CC);


%% Divide into training and validation
% 
% 1. Load scalograms into a datastore
%
% 2. Randomly split scalograms into training and validation
% (splitEachLabel)
%

allImages = imageDatastore(...
    fullfile(parentDir, cwtDir),...
    'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');

% allImages.Labels = class2double4(allImages);

rng default
[imgsTrain, imgsValidation] = splitEachLabel(allImages, 0.8, 'randomized');
disp(['Number of training images: ', num2str(numel(imgsTrain.Files))]);
disp(['Number of validation images: ', num2str(numel(imgsValidation.Files))]);


%% Train (GoogLeNet)
% model = svmtrain(allImages.Labels, 

net = googlenet;

lgraph = layerGraph(net);
numberOfLayers = numel(lgraph.Layers);

lgraph = removeLayers(lgraph,{'pool5-drop_7x7_s1','loss3-classifier','prob','output'});

numClasses = numel(categories(imgsTrain.Labels));
newLayers = [
    dropoutLayer(0.6,'Name','newDropout')
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',5,'BiasLearnRateFactor',5)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'pool5-7x7_s1','newDropout');
inputSize = net.Layers(1).InputSize;

options = trainingOptions('sgdm',...
    'MiniBatchSize',15,...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4,...
    'ValidationData',imgsValidation,...
    'ValidationFrequency',10,...
    'ValidationPatience',Inf,...
    'Verbose',1,...
    'ExecutionEnvironment','cpu',...
    'Plots','training-progress');

rng default
trainedGN = trainNetwork(imgsTrain,lgraph,options);

%%
% can ignore
[YPred,probs] = classify(trainedGN,imgsValidation);
accuracy = mean(YPred==imgsValidation.Labels);
display(['GoogLeNet Accuracy: ',num2str(accuracy)])

%%
% can ignore
wghts = trainedGN.Layers(2).Weights;
wghts = rescale(wghts);
wghts = imresize(wghts,5);
figure
montage(wghts)
title('First Convolutional Layer Weights')



%% Train (AlexNet)

alex = alexnet;

layers = alex.Layers;

numClasses = numel(categories(imgsTrain.Labels));
layers(23) = fullyConnectedLayer(numClasses);
layers(25) = classificationLayer;

inputSize = alex.Layers(1).InputSize;
augimgsTrain = augmentedImageDatastore(inputSize(1:2),imgsTrain);
augimgsValidation = augmentedImageDatastore(inputSize(1:2),imgsValidation);

rng default
mbSize = 10;
mxEpochs = 10;
ilr = 1e-4;
plt = 'training-progress';

opts = trainingOptions('sgdm',...
    'InitialLearnRate',ilr, ...
    'MaxEpochs',mxEpochs ,...
    'MiniBatchSize',mbSize, ...
    'ValidationData',augimgsValidation,...
    'ExecutionEnvironment','auto',...
    'ValidationFrequency',1000,...
    'Plots',plt);

trainedAN = trainNetwork(augimgsTrain,layers,opts);

%%
% can ignore
trainedAN.Layers(end-2:end)

%%
inputSize = alex.Layers(1).InputSize;
augimgsTrain = augmentedImageDatastore(inputSize(1:2),imgsTrain);
augimgsValidation = augmentedImageDatastore(inputSize(1:2),imgsValidation);

%% Helper functions
function LAData = helperLoadRowsNoSeg(parentDir, dataDir, fileNames, Fs, avg, fltr, numClasses)
    for i = 1:numel(fileNames)
        c{i} = SK_LA_2_ROW_UNIFORM(fullfile(parentDir, dataDir, fileNames{i}), Fs, avg, fltr);
        
        if numClasses == 2
            LAData.Labels(i, 1) = filename2class2(fileNames{i});
        elseif numClasses == 4
            LAData.Labels(i, 1) = filename2class4(fileNames{i});
        end
    end

    LAData.Data = c';
end

function LAData = helperLoadRowsCut(parentDir, dataDir, fileNames, Fs, numClasses)
    LAData = helperLoadRowsNoSeg(parentDir, dataDir, fileNames, Fs, numClasses);

    for i = 1:numel(fileNames)
        if i <= 4
            LAData.Data{i} = LAData.Data{i}(1, 138 * Fs : 1965 * Fs);
        else
            LAData.Data{i} = LAData.Data{i}(1,  84 * Fs : 2770 * Fs);
        end
    end
end

function LAData = helperLoadRowsFixed(parentDir, dataDir, fileNames, Fs, avg, fltr, numClasses, windowInterval)
    splitSamples = windowInterval * Fs;
    c = [];
    LAData.Labels = [];
    for i = 1:numel(fileNames)
        a = SK_LA_2_ROW_UNIFORM(fullfile(parentDir, dataDir, fileNames{i}), Fs, avg, fltr);
        
        if i <= 4
            a = a(1, 126 * Fs : 1965 * Fs);
        else
            a = a(1,  84 * Fs : 2770 * Fs);
        end
        
        b = [cellfun(@transpose, num2cell(reshape(a(1, 1 : floor(numel(a)/splitSamples)*splitSamples), splitSamples, []), 1)', 'UniformOutput', false); ...
            a(1, (numel(a) - mod(numel(a), splitSamples) + 1) : end)];
        
        if numClasses == 2
            label = filename2class2(fileNames{i});
        elseif numClasses == 4
            label = filename2class4(fileNames{i});
        end
        
        c = [c; b];
        LAData.Labels = [LAData.Labels; repmat(label, numel(b), 1)];
    end

    LAData.Data = c;
end

function LAData = helperLoadRowsTRB(parentDir, dataDir, fileNames, Fs, avg, fltr, numClasses, windowFilename1, windowFilename2, expandBefore, expandAfter)
    c = [];
    LAData.Labels = [];

    for i = 1:numel(fileNames)
        a = SK_LA_2_ROW_UNIFORM(fullfile(parentDir, dataDir, fileNames{i}), Fs, avg, fltr);
        

        if i == 1
            W = GET_2_W_UNIFORM(fullfile(parentDir, dataDir, windowFilename1), Fs, i, expandBefore, expandAfter);
        elseif i == 5
            W = GET_2_W_UNIFORM(fullfile(parentDir, dataDir, windowFilename2), Fs, i, expandBefore, expandAfter);
        end

        if numel(a) > numel(W(1,:))
            a = a(1, 1:numel(W(1,:)));
        elseif numel(a) < numel(W(1,:))
            a = [a, numel(W(1,:)) - numel(a)];
        end

        if numClasses == 2
            label1 = filename2class2(fileNames{i});
        elseif numClasses == 4
            label1 = filename2class4(fileNames{i});
        end

        for j = 1:3
            b{j} = a .* W(j, :);

            bb = nonzeroGroups(b{j});

            c = [c; bb];

            if j == 1
                label2 = 'LT';
            elseif j == 2
                label2 = 'RT';
            elseif j == 3
                label2 = 'RB';
            end

            LAData.Labels = [LAData.Labels; repmat(join([label1 '_' label2], ''), numel(bb), 1)];
        end
    end

    LAData.Data = c;
end

function class = filename2class4(filename)
    if contains(filename, "frontleft")
        class = "FL";
    elseif contains(filename, "frontright")
        class = "FR";
    elseif contains(filename, "backleft")
        class = "BL";
    elseif contains(filename, "backright")
        class = "BR";
    end
end

function class = filename2class2(filename)
    if contains(filename, "frontleft")
        class = "D";
    elseif contains(filename, "frontright")
        class = "P";
    elseif contains(filename, "backleft")
        class = "P";
    elseif contains(filename, "backright")
        class = "P";
    end
end

function labels = class2double4(datastore)
    labels = zeros(numel(datastore.Labels), 1);
    
    for i = 1:numel(datastore.Labels)
        if datastore.Labels(i,1) == "FL"
            labels(i,1) = 0;
        elseif datastore.Labels(i,1) == "FR"
            labels(i,1) = 1;
        elseif datastore.Labels(i,1) == "BL"
            labels(i,1) = 2;
        elseif datastore.Labels(i,1) == "BR"
            labels(i,1) = 3;
        end
    end
end

function labels = class2double2(datastore)
    labels = zeros(numel(datastore.Labels), 1);
    
    for i = 1:numel(datastore.Labels)
        if datastore.Labels(i,1) == "D"
            labels(i,1) = 0;
        elseif datastore.Labels(i,1) == "P"
            labels(i,1) = 1;
        end
    end
end

function out = nonzeroGroups(b)
    ii = zeros(size(b));
    jj = b > 0;
    ii(strfind([0,jj(:)'],[0 1])) = 1;
    idx = cumsum(ii).*jj;
    out = accumarray( idx(jj)',b(jj)',[],@(x){x'});
end

function helperCreateCWTFolders(Data, parentDir, cwtDir)
    mkdir(fullfile(parentDir, cwtDir))
    folderLabels = unique(Data.Labels);
    for i = 1:numel(folderLabels)
        mkdir(fullfile(parentDir, cwtDir, char(folderLabels(i))));
    end
end

function helperGenerateAndSaveCWT(Data, Fs, size, parentDir, cwtDir)
    r = numel(Data.Data);
    for i = 1:r
        wt = cwt(Data.Data{i}, Fs, 'VoicesPerOctave', 12);
        im = ind2rgb(im2uint8(rescale(abs(wt))), jet(128));

        imgLoc = fullfile(parentDir, cwtDir, char(Data.Labels(i)));
        imFileName = strcat(char(Data.Labels(i)), '_', num2str(i), '.jpg');
        imwrite(imresize(im, [size size]), fullfile(imgLoc, imFileName));
    end
end

function helperGenerateAndSaveCombinedCWTPartial(Data, Fs, size, parentDir, cwtDir, I, offset, category)
    pim = perms([1 2 3 4]);
    
    for i = I
        im = [];
        for j = 0:3
            wt = cwt(Data.Data{i + offset * j}, Fs, 'VoicesPerOctave', 12);
            rgb = ind2rgb(im2uint8(rescale(abs(wt))), jet(128));
            rgbresized = imresize(rgb, [size size]);
            im{j+1, 1} = rgbresized;
        end
        
        for j = 1:numel(pim(:,1))
            imcom = [];
            for k = 1:4
                imcom = [imcom; im{pim(j,k)}];
            end
            imgLoc = fullfile(parentDir, cwtDir, category, string(find(pim(j,:) == 1)));
            mkdir(imgLoc);
            imFileName = strcat(category, '_', num2str(i), '_', join(string(pim(j,:)), ''), '.png');
            imwrite(imcom, fullfile(imgLoc, imFileName), 'Mode', 'lossless');
        end
    end
end

function helperGenerateAndSaveCombinedCWT(Data, Fs, size, parentDir, cwtDir, II, OO, CC)
    for i = 1:numel(II)
        helperGenerateAndSaveCombinedCWTPartial(Data, Fs, size, parentDir, cwtDir, II{i}, OO{i}, CC{i});
    end
end