function Backprojection = backwardProjectACC(Ht, projection, CAindex )

    x3length = size(Ht,5);
    Nnum = size(Ht,3);
    Backprojection = zeros(size(projection, 1), size(projection, 2), x3length);
    zeroSlice = zeros(  size(projection,1) , size(projection, 2));

    for cc=1:x3length
        tempSliceBack = zeroSlice;
        for aa=1:Nnum
            for bb=1:Nnum       

                Hts = squeeze(Ht( CAindex(cc,1):CAindex(cc,2), CAindex(cc,1):CAindex(cc,2) ,aa,bb,cc));                  
                tempSlice = zeroSlice;
                tempSlice( (aa:Nnum:end) , (bb:Nnum:end) ) = projection( (aa:Nnum:end) , (bb:Nnum:end) );
                tempSliceBack = tempSliceBack + conv2(tempSlice, Hts, 'same');   

            end
        end
        Backprojection(:,:,cc) = Backprojection(:,:,cc) + tempSliceBack;
    end
end

