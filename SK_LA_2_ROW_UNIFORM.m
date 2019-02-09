function row = SK_LA_2_ROW_UNIFORM(filepath, desiredFs, avg, fltr)
    row = table2array(readtable(filepath))';
    row = [row(1,:); sqrt(row(2,:).^2 + row(3,:).^2 + row(4,:).^2)];
    row = resample(row(2,:), row(1,:), desiredFs);
    
    if fltr > 0
        row = lowpass(row, fltr, desiredFs, 'Steepness', 0.99);
    end
    
    if avg > 0
        row = movmean(row, desiredFs * avg);
    end
end