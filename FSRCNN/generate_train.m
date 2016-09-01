%9x5x5 training examples 
clear;close all;
%% settings
folder = '../train';
savepath = 'fsrcnn_train.h5';
size_input = 7;
size_label = 19;
scale = 3;
stride = 7;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));
    
for i = 1 : length(filepaths)
    image = imread(fullfile(folder,filepaths(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));
    
    im_label = modcrop(image, scale);
    im_input = imresize(im_label,1/scale,'bicubic');
    [hei,wid] = size(im_input);
    
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1     
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
            subim_label = im_label(x*scale  : x*scale+size_label-1, y*scale : y*scale+size_label-1);

            count=count+1;
            data(:, :, 1, count) = subim_input;
            label(:, :, 1, count) = subim_label;
        end
    end
end

order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order); 

%% writing to HDF5
chunksz = 128;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
