% =========================================================================
% 描述：   一个使用训练好的网络进行超分辨率的应用demo（使用caffe的matlab接口）          
%    
% 参考文献：
%  Shi W, Caballero J, Huszar F, et al. Real-Time Single Image and Video
%  Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network[C]
%  
% 王学文
% wangxuewen@yy.com
% =========================================================================
%% settings
model = './espcn_mat.prototxt';
weights = './snapshot/espcn_iter_10000.caffemodel';
batch = 1;
up_scale = 3;
%% read data
input = imread('../data/test/set14/lenna.bmp');
if size(input,3)>1
    input = rgb2ycbcr(input);
    input = input(:,:, 1);
end;
input = single(input)/255;
input = imresize(input, 1/up_scale, 'bicubic');
[height, width, channel] = size(input);

%% use gpu mode
caffe.reset_all(); 
caffe.set_mode_gpu();
caffe.set_device(0);

%% load model using mat_caffe
net = caffe.Net(model,weights,'test');
net.blobs('data').reshape([height width channel batch]); % reshape blob 'data'
net.reshape();
net.blobs('data').set_data(input);
net.forward_prefilled();
output = net.blobs('conv3').get_data();
[output_height, output_width, output_channel] = size(output);
scaled_height = up_scale * output_height;
scaled_width = up_scale * output_width;
im_h = zeros(scaled_height, scaled_width);
    
for m = 1 : up_scale
     for n = 1 : up_scale
         im_h(m:up_scale:scaled_height+m-up_scale,n:up_scale:scaled_width+n-up_scale) = output(:,:,(m-1)*up_scale+n);   
     end
end
im_h = uint8(im_h * 255);
imwrite(im_h,'espcn_caffe.bmp');

