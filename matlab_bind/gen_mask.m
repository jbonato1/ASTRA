data = hdf5read('Contours/SMALL_001.hdf5','cc');
data_bb = hdf5read('Contours/SMALL_001.hdf5','bb');
data_size = size(data);
bd0 = [];
for i = 1:data_size(1)
    
    c_pack = [];

    
    p = reshape(data(i,:,:),246,246);
    p_t = p';
    bb_t = bb';

    c_pack{1} = bwboundaries(p_t);
    c_pack{2} = find(p_t>0);
    c_pack{3} = 'manual';
    c_pack{4} = 'None';
    bd0{end+1}=c_pack;
end


save('c.mat','bd0')