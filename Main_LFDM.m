clear; clc;
addpath(genpath('utils/'));

% Set Parameters %
% M=40; NA=0.8; MLPitch=150; fml=3500; zmin=-30; zmax=30; zspacing=1;
% Nnum=11; lambda=580; n=1.33
dx = 11; % The number of pixels behind a lenslet
Nnum = 11; % The number of virtual pixel you expect to have behing a lenslet. e.g. You may have dx as 23, but you think it¡¯s enough or it¡¯s required only to have 11 pixels for each lenslet, so you could set Nnum as 11.
rectification_enable = 1; % Enable to rectify each stack according to dx and Nnum.
range_adjust = 0.9; % [0,1], An adjustable parameter for dynamic range. The higher it is, the higher the intensity of output image will be. 
bitdepth = 8; % 8 or 16 bit

% ImageRectification %
file = 'data/raw_data/tubulins3d1.tif';
original_stack = imread3d(file);
[row,col,~] = size(original_stack);
if rectification_enable
    xCenter = col/2;   
    yCenter = row/2;
    rectified_stack = VolumeRectify(original_stack,xCenter,yCenter,dx,Nnum);
else
    rectified_stack = original_stack;
end                
rectified_stack = (rectified_stack ./ max(rectified_stack(:))) .* range_adjust .* double(2^bitdepth-1);

% Forward projection parameters %
brightness_adjust = 0.0039; % [0,1] An adjustable parameter for dynamic range. The higher it is, the higher the intensity of output image will be.
poisson_noise = 0; % Enable to add poisson noise.
gaussian_noise = 0; % Enable to add gaussian noise (it will be added after poisson noise if Poisson Noise is also enabled). 
gaussian_sigma = 5e-5; % Specify the standard deviation of the noise.
gpu = 1; % Enable to use GPU for processing.
% H = ComputePSF(40,0.8,150,3500,-30,30,1,11,580,1.33,3);
psf_name = 'PSFmatrix_M40NA0.8MLPitch150fml3500from-30to30zspacing1Nnum11lambda580n1.33.mat';
psf_path = 'data/PSFmatrix/';

% Forward project the HR data into synthetic light field image %
disp('Loading LF_PSF...' );
[LFpsf,psf_h,psf_w,psf_d,Nnum,CAindex,LFpsft] = read_psf([psf_path psf_name]);
disp(['LF_PSF has been loaded. Size: ' num2str(psf_h) 'x' num2str(psf_w) 'x' num2str(Nnum) 'x' num2str(Nnum) 'x' num2str(psf_d) '.']);
depth = psf_d;
volume_dims = size(rectified_stack);
if depth == volume_dims(3)
    if gpu  
        volume = gpuArray(single(rectified_stack));
        stacks = zeros(volume_dims,'double');
        global zeroImageEx;
        global exsize;
        xsize = [volume_dims(1), volume_dims(2)];
        msize = [size(LFpsf,1), size(LFpsf,2)];
        mmid = floor(msize/2);
        exsize = xsize + mmid;  
        exsize = [ min( 2^ceil(log2(exsize(1))), 128*ceil(exsize(1)/128) ), min( 2^ceil(log2(exsize(2))), 128*ceil(exsize(2)/128) ) ];    
        zeroImageEx = gpuArray(zeros(exsize, 'single'));
        for d = 1 : depth 
            for i = 1 : Nnum
                for j = 1 : Nnum
                    sub_region =  gpuArray.zeros(volume_dims(1),volume_dims(2),'single');
                    sub_region(i: Nnum: end,j: Nnum: end) = volume(i: Nnum: end, j: Nnum: end, d);
                    sub_psf = gpuArray(single(squeeze(LFpsf( CAindex(d,1):CAindex(d,2), CAindex(d,1):CAindex(d,2) ,i,j,d))));
                    sub_Out = conv2FFT(sub_region, sub_psf);
                    % sub_Out = conv2(sub_region, sub_psf,'same');
                    sub_out = gather(sub_Out);
                    stacks(:, :, d) = stacks(:, :, d) + sub_out;
                end
            end
        end
    else
        stacks = zeros(volume_dims,'double');
        for d = 1 : depth 
            for i = 1 : Nnum
                for j = 1 : Nnum
                    sub_region =  zeros(volume_dims(1),volume_dims(2));
                    sub_region(i: Nnum: end,j: Nnum: end) = rectified_stack(i: Nnum: end, j: Nnum: end, d);
                    sub_psf = squeeze(LFpsf( CAindex(d,1):CAindex(d,2), CAindex(d,1):CAindex(d,2) ,i,j,d));
                    sub_out = conv2(sub_region,sub_psf,'same');
                    stacks(:, :, d) = stacks(:, :, d) + sub_out;
                end
            end
        end
    end
    LF_raw  = (sum(stacks , 3)).* brightness_adjust;
    if bitdepth == 8
        LF_raw = uint8(LF_raw);
    elseif bitdepth == 16
        LF_raw = uint16(LF_raw);
    end                
    if poisson_noise
        LF_raw_poisson_noise = imnoise(LF_raw, 'poisson');
    else
        LF_raw_poisson_noise = LF_raw;
    end
    if gaussian_noise
        LF_raw_gaussian_noise = imnoise(LF_raw_poisson_noise, 'gaussian', 0, gaussian_sigma);
    else
        LF_raw_gaussian_noise = LF_raw_poisson_noise;
    end
    LF_fp = LF_raw_gaussian_noise;
else
    disp( 'The slice number of the stack does not match the PSF depth');         
end
figure;imshow(LF_fp,[]);
imwrite(mat2gray(LF_fp),'results/Lightfield_forwardprojection.png');

% 3D Reconstruction using deconvolution %
Solver = 'Richardson-Lucy';
maxIter = 8;
GPUON = 1;
edgeSuppress = 1;
[XvolumeFinal,settingRECON] = Reconstruction3D(LFpsf,LFpsft,LF_fp,Solver,maxIter,GPUON,edgeSuppress);

write3d(255*mat2gray(XvolumeFinal),'results/Reconstruction3D.tif', bitdepth);
save('results/Reconstruction3D.mat','XvolumeFinal','LF_fp','rectified_stack','settingRECON');




