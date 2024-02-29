clc; clear; close all;

% Step 1: Set up GPU for parallel computing
gpuDevice(1);
D = gpuDevice(1);

% Step 2: Load image stack
imageDirectory = 'D:\img_flipped';
imageFiles = dir(fullfile(imageDirectory, 'A*.tif'));
numImages = numel(imageFiles);

% Step 3: Read images
image = imread(fullfile(imageDirectory, imageFiles(1).name));
imageSize = size(image);

numImagesPerBatch = 10; % Number of images to process per batch
numBatches = ceil(numImages / numImagesPerBatch);

disp('Set up GPU complete.'); % Display completion message
beep;

%% Binning for X, Y axes using GPU

% Step 4: Define binning factors for X, Y axes
binningFactorX = 2; % Reduce size by a factor of 2 along X axis
binningFactorY = 2; % Reduce size by a factor of 2 along Y axis

% Step 5: Implement binning algorithm using GPU-accelerated functions
outputSizeX = round(imageSize(2) / binningFactorX);
outputSizeY = round(imageSize(1) / binningFactorY);
binnedImages = cell(numImages, 1); % Temporary cell array to store binned images

parfor imageIndex = 1:numImages
    % Read the image
    image = imread(fullfile(imageDirectory, imageFiles(imageIndex).name));
    % Transfer the image to the GPU
    gpuImage = gpuArray(image);
    % Perform binning along X and Y axes using GPU-accelerated function
    binnedImageXY = imresize(gpuImage, 1/binningFactorX, 'bilinear');
    % Store the binned image in the cell array
    binnedImages{imageIndex} = gather(binnedImageXY);
    fprintf('Processed image: %d/%d\n', imageIndex, numImages);
end

% Combine binned images from cell array into a single matrix
binnedImagesCombined = cat(3, binnedImages{:});

disp('GPU binning complete.'); % Display completion message
beep;

%% Binning for Z axis using CPU

% Step 1: Compute the new dimensions after binning
binningFactorZ = 4;  % Reduce size by a factor of 2 along Z axis
binnedZSize = ceil(numImages / binningFactorZ);
binnedReleasedImages = zeros(outputSizeY, outputSizeX, binnedZSize, 'uint8');

% Step 2: Iterate through each binning group and perform binning
for i = 1:binnedZSize
    % Compute the range of indices for the binning group
    startIndex = (i - 1) * binningFactorZ + 1;
    endIndex = min(i * binningFactorZ, numImages);
    indices = startIndex:endIndex;
    
    % Perform binning on the Z-direction by averaging the values
    binnedReleasedImages(:, :, i) = uint8(mean(binnedImagesCombined(:, :, indices), 3));
end

disp('Z axis binning complete.'); % Display completion message
beep;

%% Saving binned images

% Set the directory where you want to save the images
outputDir = 'D:\img_flipped\binning';

% Create the output directory if it does not exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Step 1: Iterate through each binned image and save it
for i = 1:binnedZSize
    % Generate the filename for the binned image
    filename = sprintf('Binned_%04d.tif', i);
    
    % Convert the binned image to the appropriate data type
    binnedImage = binnedReleasedImages(:, :, i);
    
    % Step 2: Save the binned image
    imwrite(binnedImage, fullfile(outputDir, filename));
    
    fprintf('Saved image: %s\n', filename);
end

reset(D);

disp('Saving binned images complete.'); % Display completion message
beep;
