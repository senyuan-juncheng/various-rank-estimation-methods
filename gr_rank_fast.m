% fast version of gr by Toshinari Morimoto
% based on Prof Hung's code (just added a small modification)

function [r_gr, gr_val] = gr_rank_fast(x,upperbound)
[n,p,T] = size(x);

if(upperbound > p-2)
    upperbound = p-2;
end

% mean with respect to n
x_ = repmat(mean(x,1),[n,1,1]); 

% compute covariance matrix
S = pagemtimes(permute(x-x_,[2,1,3]),x-x_)/(n-1);

% svd of S
[~,ll_,~] = pagesvd(S,"vector"); % x: n * p * T
ll = reshape(ll_,[p,T]); % [p,1,T] --> [p,T]


% compute gr_val
a = [sum(ll); sum(ll)-cumsum(ll)];
gr_val =log(a(1:upperbound,:)./a(2:upperbound+1,:))./log(a(2:upperbound+1,:)./a(3:upperbound+2,:));

% find the optimal rank
[~, r_gr] = max(gr_val);

end

