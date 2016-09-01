%% settings
model = './SRCNN_mat.prototxt';
weights = './Game/game_iter_1583500.caffemodel';
batch = 1;

%% read data
input = imread('./Game_test/10_bicubic.bmp');
if size(input,3)>1
    input = rgb2ycbcr(input);
    input = input(:,:, 1);
end;
input = single(input)/255;
input = input(1:1024,1:1024,:); 
[height, width, channel] = size(input);
%% use gpu mode
caffe.set_mode_gpu();
caffe.set_device(0);

%% load model using mat_caffe
net = caffe.Net(model,weights,'test');
net.blobs('data').reshape([height width channel batch]); % reshape blob 'data'
net.reshape();
net.blobs('data').set_data(input);
net.forward_prefilled();
output = net.blobs('conv3').get_data();
output = uint8(output * 255);
imwrite(output,'srcnn_caffe.bmp');

