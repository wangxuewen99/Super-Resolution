% =========================================================================
%@description£ºdraw psnr/iter curves
%@author; Xuewen Wang
%@date: 2016/6/30
% =========================================================================

close all;
clear all;

%% set parameters
up_scale = 3;
model = './fsrcnn_mat.prototxt';


%% read ground truth image
im = imread('../test/set14/monarch.bmp');
%% work on illuminance only
if size(im,3)>1
    im = rgb2ycbcr(im);
    im = im(:, :, 1);
end

im_gnd = modcrop(im, up_scale);
im_gnd = single(im_gnd)/255;

%% bicubic interpolation
im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
im_b = imresize(im_l, up_scale, 'bicubic');
im_gnd = uint8(im_gnd * 255);
im_gnd = im_gnd(3:end,3:end);
[input_height ,input_width] = size(im_l);
input_channel = 1;
batch =1;
step = 10000;
min_iter = 10000;
max_iter = 2390000;
count = fix((max_iter - min_iter)/step) +1;
psnr = zeros(1,count);
count = 1;

%% load model
caffe.reset_all(); 
caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(model,'test');
net.blobs('data').reshape([input_height input_width input_channel batch]); % reshape blob 'data'
net.reshape();
net.blobs('data').set_data(im_l);
for fid = min_iter : step : max_iter
    weights = ['./snapshot/snapshot_iter_' num2str(fid) '.caffemodel'];
    %% load weights
    net.copy_from(weights);
    %% super-resolution 
    net.forward_prefilled();
    output = net.blobs('deconv').get_data();
    im_h = uint8(output * 255);
    %% compute PSNR    
    psnr_srcnn = compute_psnr(im_gnd,im_h);
    psnr(count) = psnr_srcnn;
    count = count+1;
end

im_b = uint8(im_b * 255);
im_b = im_b(3:end,3:end);
psnr_bic = compute_psnr(im_gnd,im_b);
x = min_iter : step : max_iter;
y = psnr_bic + zeros(size(x));
plot(x,psnr,x,y);
imwrite(im_b, ['Bicubic Interpolation' '.bmp']);
imwrite(im_h, ['SRCNN Reconstruction' '.bmp']);
