% =========================================================================
% 描述：   绘制PSNR值随网络不断迭代的变化曲线，需要设置min_iter,max_iter,step          
%    
% 参考文献：
%  Dong C, Loy C C, Tang X. Accelerating the Super-Resolution Convolutional 
%  Neural Network[J].
%  
% 王学文
% wangxuewen@yy.com
% =========================================================================
close all;
clear all;

%% set parameters
up_scale = 3;
shift = up_scale - 1;
model = './fsrcnn_mat.prototxt';
step = 10000;
min_iter = 10000;
max_iter = 2390000;

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
im_gnd = im_gnd(shift + 1: end,shift +1 : end);
[input_height ,input_width] = size(im_l);
input_channel = 1;
batch =1;
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
im_b = im_b(shift + 1:end,shift +1 :end);
psnr_bic = compute_psnr(im_gnd,im_b);
x = min_iter : step : max_iter;
y = psnr_bic + zeros(size(x));
plot(x,psnr,x,y);
imwrite(im_b, ['Bicubic Interpolation' '.bmp']);
imwrite(im_h, ['SRCNN Reconstruction' '.bmp']);
