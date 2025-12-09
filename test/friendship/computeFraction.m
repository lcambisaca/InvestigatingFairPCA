function fractions = computeFraction(labels,gmale)
    unilb = unique(labels);
    k = length(unilb); 
    fractions = zeros(k,1);
    for i = 1:k
        i % Group 1
        idx = labels == unilb(i);
        ci = sum(idx) % Number of elements in that label
        vfandci = sum(gmale(idx)) % Number of males within the label
        fractions(i) = vfandci/ci; % Fraction of males in that label
    end
end

% Number of males in label i / Total number of elements in label i

