function [A, D, F] = generate_connected_SBM(n,a,b,c,d,k,h,block_sizes,sensitive)
%Generates a SBM Model
%
%INPUT:
%   n ... number of elements
%   a,b,c,d ... parameters / probabilities
%   k ... number of clusters
%   h ... number of groups
%   block_sizes ... vector of length k*h with sum(block_sizes)=n; the 
%                first h entries correspond to the first cluster for 
%                the h groups, and so on ... 
%
%OUTPUT
%   A ... adjacency matrix of size n x n
%   D ... degree matrix of A
%   F ... group membership matrix of size n x (h-1)


if (sum(block_sizes)~=n)||(length(block_sizes)~=(k*h))
    error('wrong input')
end

adja=random('Binomial',1,d,n,n);
for ell=1:k
    for mmm=1:k
        for ggg=1:h
            for fff=1:h
                if ell==mmm
                    if ggg==fff
                        adja((sum(block_sizes(1:((ell-1)*h+(ggg-1))))+1):(sum(block_sizes(1:((ell-1)*h+(ggg))))),...
                            (sum(block_sizes(1:((ell-1)*h+(ggg-1))))+1):(sum(block_sizes(1:((ell-1)*h+(ggg))))))=random('Binomial',1,a,block_sizes((ell-1)*h+(ggg)),block_sizes((ell-1)*h+(ggg)));
                    else
                        adja((sum(block_sizes(1:((ell-1)*h+(ggg-1))))+1):(sum(block_sizes(1:((ell-1)*h+(ggg))))),...
                            (sum(block_sizes(1:((ell-1)*h+(fff-1))))+1):(sum(block_sizes(1:((ell-1)*h+(fff))))))=random('Binomial',1,c,block_sizes((ell-1)*h+(ggg)),block_sizes((ell-1)*h+(fff)));
                    end
                    
                else
                    if ggg==fff
                        adja((sum(block_sizes(1:((ell-1)*h+(ggg-1))))+1):(sum(block_sizes(1:((ell-1)*h+(ggg))))),...
                                (sum(block_sizes(1:((mmm-1)*h+(ggg-1))))+1):(sum(block_sizes(1:((mmm-1)*h+(ggg))))))=random('Binomial',1,b,block_sizes((ell-1)*h+(ggg)),block_sizes((mmm-1)*h+(ggg)));
                    end
                end
            end
        end
    end
end
            



A=triu(adja,1);
A=A+A';

% --- The Only different part accessed when we dont have a fully
% connected graph
G = graph(A);
bins = conncomp(G);
if max(bins) > 1      % if there's more than 1 component...
    disp("BOO IN FUNCTIO")
   % Multiple components exist → connect them
   perm = randperm(n);           % random ordering of all nodes
   for i = 2:n
       u = perm(i);              % pick a node
       v = perm(randi(i-1));     % connect it to some earlier node
       A(u,v) = 1;               % add edge (u,v)
       A(v,u) = 1;               % undirected, so symmetric
   end
end
% -------------------------------------------------------------

% --- The Same above as new one
sens_unique=unique(sensitive);
sens_unique=reshape(sens_unique,[1,h]);

sensitiveNEW=sensitive;
temp=1;
for ell=sens_unique
    sensitiveNEW(sensitive==ell)=temp;
    temp=temp+1;
end
    
F=zeros(n,h-1);

for ell=1:(h-1)
    temp=(sensitiveNEW == ell);
    F(temp,ell)=1; 
    groupSize = sum(temp);
    F(:,ell) = F(:,ell)-groupSize/n;
end

degrees = sum(A, 1);
D = diag(degrees);
end

 

