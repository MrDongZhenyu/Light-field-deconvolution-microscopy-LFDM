function [Xguess] = deconvRL(forwardFUN, backwardFUN, Htf, maxIter, Xguess )

    for i=1:maxIter
        tic;
        HXguess = forwardFUN(Xguess);
        HXguessBack = backwardFUN(HXguess);
        errorBack = Htf./HXguessBack;
        Xguess = Xguess.*errorBack; 
        Xguess(isnan(Xguess)) = 0;
        ttime = toc;
        disp(['  iter ' num2str(i) ' | ' num2str(maxIter) ', took ' num2str(ttime) ' secs']);
    end
end
    


        