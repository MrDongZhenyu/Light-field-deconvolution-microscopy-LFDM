function [XvolumeFinal,settingRECON] = Reconstruction3D(H,Ht,LF_fp,Solver,maxIter,GPUON,edgeSuppress)
    addpath(genpath('utils/'));
    
    settingRECON.maxIter = maxIter;
    settingRECON.GPUON = GPUON;
    settingRECON.edgeSuppress = edgeSuppress;
    settingRECON.whichSolver  = Solver;

    if class(H) == 'double'
        H = single(H);
        Ht = single(Ht);
    end
    
    global volumeResolution ;
    volumeResolution = [size(LF_fp,1),size(LF_fp,2),size(H,5)]; 
    
    % Prepare parallal computing & GPU computation %
    p = gcp;

    if GPUON
        Nnum = size(H,3);
        backwardFUN = @(projection) backwardProjectGPU(Ht, projection);
        forwardFUN = @(Xguess) forwardProjectGPU( H, Xguess );

        global zeroImageEx;
        global exsize;
        xsize = [volumeResolution(1), volumeResolution(2)];
        msize = [size(H,1), size(H,2)];
        mmid = floor(msize/2);
        exsize = xsize + mmid;  
        exsize = [ min( 2^ceil(log2(exsize(1))), 128*ceil(exsize(1)/128) ), min( 2^ceil(log2(exsize(2))), 128*ceil(exsize(2)/128) ) ];    
        zeroImageEx = gpuArray(zeros(exsize, 'single'));
        disp(['FFT size is ' num2str(exsize(1)) 'X' num2str(exsize(2))]); 
    else
        forwardFUN =  @(Xguess) forwardProjectACC( H, Xguess, CAindex );
        backwardFUN = @(projection) backwardProjectACC(Ht, projection, CAindex );
    end
    
    % Run reconstruction %

    LFIMG = single(LF_fp);        
    tic; Htf = backwardFUN(LFIMG); ttime = toc;
    disp(['  iter ' num2str(0) ' | ' num2str(maxIter) ', took ' num2str(ttime) ' secs']);
    Xguess = Htf; % initial guess
    if Solver == 'Richardson-Lucy'
        Xguess = deconvRL(forwardFUN, backwardFUN, Htf, maxIter, Xguess );
    end
    
    if GPUON
       Xvolume = gather(Xguess);
    else
       Xvolume = Xguess;
    end    
    
    delete(p);
    if edgeSuppress         
        Xvolume( (1:1*Nnum), :,:) = 0;
        Xvolume( (end-1*Nnum+1:end), :,:) = 0;
        Xvolume( :,(1:1*Nnum), :) = 0;
        Xvolume( :,(end-1*Nnum+1:end), :) = 0;
    end

    XvolumeFinal = double(Xvolume);
    disp('Volume reconstruction complete.');
end