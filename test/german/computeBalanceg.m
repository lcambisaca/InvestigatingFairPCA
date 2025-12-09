function balance = computeBalanceg(labels, gmale, k)
%     gfemale = double(~gmale);
    b = zeros(k, 1);  % T0 store balance score for each cluster
    c = zeros(k, 1); %Stores number of elements in each cluster
    for i = 1:k
        idx = find(labels == i); % Indices of points in cluster i
        c(i) = length(idx);
        count = 0;
        for j =1:length(idx)
            if gmale(idx(j)) == 1  % Count how many are male
                count = count + 1;
            end
        end
        % Count = number of males
        % c(i) - Count = number of females
        b(i) = min(count/(c(i)-count),(c(i)-count)/count); % Find min ratio between number of males and number of feamles
    end
    balance = mean(b);
end

