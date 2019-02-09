function W = GET_2_W_UNIFORM(filepath, desiredFs, whichone, expandBefore, expandAfter)
    T = table2array(readtable(filepath))';
    if whichone == 1
        T(1,:) = (T(1,:) - 1543155536364) ./ 1000 + 125.742217; % align with data
    elseif whichone == 5
        T(1,:) = (T(1,:) - T(1,1)) ./ 1000 - 19500 + 15.766952; % align with data
    end
    T = T(1:4, find(T(1,:) > 0, 1):end); % remove nonzero values
    T = [zeros(4,1), T];
    W(1,:) = resample(T(2,:), T(1,:), desiredFs); % left turn
    W(2,:) = resample(T(3,:), T(1,:), desiredFs); % right turn
    W(3,:) = resample(T(4,:), T(1,:), desiredFs); % road bump
    W = round(W); % will make it either 0 or 1
    
    before = expandBefore * desiredFs;
    after  = expandAfter  * desiredFs;
    Bb = [];
    Ba = [];
    if before > 0
        for i = 1:3
            A = [];
            for j = 2:numel(W(i,:))
                if prod(W(i, j-1:j) == [0 1]) == 1
                    A = [A, j];
                end
            end
            Bb{i} = A;
        end
    end
    
    if after > 0
        for i = 1:3
            A = [];
            for j = 2:numel(W(i,:))
                if prod(W(i, j-1:j) == [1 0]) == 1
                    A = [A, j];
                end
            end
            Ba{i} = A;
        end
    end

    if before > 0
        for i = 1:3
            A = Bb{i};
            for j = A
                W(i, max(0, j - before - 1) : j - 1) = 1;
            end
        end
    end
    
    if after > 0
        for i = 1:3
            A = Ba{i};
            for j = A
                W(i, j : min(j + after, end)) = 1;
            end
        end
    end
end