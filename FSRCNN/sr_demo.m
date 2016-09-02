% =========================================================================
% 描述：   一个使用训练好的网络进行超分辨率的应用demo（使用caffe的matlab接口）          
%    
% 参考文献：
%  Dong C, Loy C C, Tang X. Accelerating the Super-Resolution Convolutional 
%  Neural Network[J].
%
% 王学文
% 欢聚时代|欢聚云|算法中心
% wangxuewen@yy.com
% =========================================================================
%% settings
model = './fsrcnn_mat.prototxt';
weights = './snapshot/snapshot_iter_2546000.caffemodel';
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
output = net.blobs('deconv').get_data();
im_h = uint8(output * 255);
imwrite(im_h,'fsrcnn_caffe.bmp');
