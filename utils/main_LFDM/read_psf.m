function [LFpsf,height,width,depth,Nnum,CAindex,LFpsft] = read_psf(file)
    psf = load(file);
    psf5d = psf.H;
    Nnum = psf.Nnum;
    CAindex = psf.CAindex;
    [height, width, ~, ~, depth] = size(psf5d);
    LFpsf = double(psf5d);
    LFpsft = double(psf.Ht);
end