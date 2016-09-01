model = './FSRCNN_mat.prototxt';
weights = './snapshot/snapshot_iter_1000.caffemodel';
caffe.reset_all(); 
caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(model,weights,'test');

layers = 6;
for idx = 1 : layers
    conv_filters = net.layers(['conv' num2str(idx)]).params(1).get_data();
    [~,fsize,channel,fnum] = size(conv_filters);   
end

deconv_filter = net.layers('deconv').params(1).get_data();