% fast version of er rank by Toshinari Morimoto
% based on Prof.Hung's program

% last update 2022/07/21 (Thr)

function [r_er, er_val] = er_rank_fast(x,upperbound)

[n,p,T] = size(x);


if(upperbound > min(n-1,p)-1)
    upperbound = min(n-1,p)-1;
end



% mean with respect to n
x_ = repmat(mean(x,1),[n,1,1]); 

% compute covariance matrix
S = pagemtimes(permute(x-x_,[2,1,3]),x-x_)/(n-1);

% svd of S
[~,ll_,~] = pagesvd(S,"vector"); % x: n * p * T
ll = reshape(ll_,[p,T]); % [p,1,T] --> [p,T]

% er_val
er_val = ll(1:upperbound,:)./ll(2:upperbound+1,:);

% rank
[~,r_er] = max(er_val);
end
