folder_res = '/media/data/jbonato/astro_segm/Aqua_binding/Contours/*.hdf5' 
folder_save = '/media/data/jbonato/astro_segm/Aqua_binding/Aqua_Contours/' 

files = dir(folder_res)

get_bb = false;

for x = 1:size(files,1)

    f0 = files(x).name;
    
    file = strcat(files(x).folder,'/',files(x).name);
    if(get_bb)
    data = h5read(file,'/bb');
    else
    data = h5read(file,'/cc');
    end
    
    data_size = size(data);
    bd0 = [];
    for i = 1:data_size(3)

        c_pack = [];

        %%%% set height-2*rem_px and width-2*rem_px of the FOV where rem_px is the number of pixels removed from the
        %%%% the border in this case is 5 and H and W is 256
        buff = data(6:end-5,6:end-5,i);
        p = reshape(buff,246,246);
        p_t = p';


        c_pack{1} = bwboundaries(p_t);
        c_pack{2} = find(p_t>0);
        c_pack{3} = 'manual';
        c_pack{4} = 'None';
        bd0{end+1}=c_pack;
    end

    save_file = strcat(folder_save,files(x).name(1:end-5),'.mat')
    save(save_file,'bd0')
end