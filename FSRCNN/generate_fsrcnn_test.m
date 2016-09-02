% =========================================================================
% 描述：   产生测试样本，将高分辨率图片下采样指定倍率得到低分辨率图像,应再在高低
%          分辨率图像中提取图像块作为样本，洗牌打乱后写入hdf5文件中                
%          (56,12,4) 
% 参考文献：
%  Dong C, Loy C C, Tang X. Accelerating the Super-Resolution Convolutional 
%  Neural Network[J].
%  
% 王学文
% wangxuewen@yy.com
% =========================================================================
clear;close all;
%% settings
folder = '../data/test/set5';
savepath = 'fsrcnn_test.h5';
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
            subim_label = im_label(x*scale : x*scale+size_label-1, y*scale : y*scale+size_label-1);

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
chunksz = 32;
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