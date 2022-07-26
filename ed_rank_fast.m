% fast version of ed estimator by Toshinari Morimoto
% based on Prof.Hung's program

function [r_ed] = ed_rank_fast(x,upperbound)

[n,p,T] = size(x);

if(upperbound > p-4)
    upperbound = p-4;
end

% mean with respect to n
x_ = repmat(mean(x,1),[n,1,1]); 

% compute covariance matrix
S = pagemtimes(permute(x-x_,[2,1,3]),x-x_)/(n-1);

% svd of S
[~,ll_,~] = pagesvd(S,"vector"); % x: n * p * T
ll = reshape(ll_,[p,T]); % [p,1,T] --> [p,T]

% regression
j = upperbound+1;
x0 = [ones(5,1), ([j-1:j+3].^(2/3))'];
x0 = reshape(x0,[5,2,1]);
x0 = repmat(x0,[1,1,T]);
y0 = ll(j-1:j+3,:);
y0 = reshape(y0,[5,1,T]);
x00x = pagemtimes(permute(x0,[2,1,3]),x0);
beta = pagemtimes(pageinv(x00x), pagemtimes(permute(x0,[2,1,3]),y0));
beta = reshape(beta,[2,T]);
delta = abs(beta(2,:)).*2;

% find optimal rank
diff = ll(1:upperbound,:)-ll(2:(upperbound+1),:);
idx = (diff >= delta);
v = (1:upperbound)';
v = reshape(v,[upperbound,1]);
v = repmat(v,[1,T]);
r_ed = max(idx .* v);

end

