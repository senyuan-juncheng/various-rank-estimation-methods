%
% act rank fast version by Toshinari Morimoto
%

% lastupdate 2022/7/20 (Wed)

function [r_act,act_val] = act_rank_fast(x,upperbound)

% adjustment parameter
adj_para = 4;

% These are used in taking elementwise inverse or elementwise log of matrices containing 0. 
% (want to define log0 and 0^-1 as 0 to ignore them)
epsilon = 10^-20;
inf     = 10^16;

% size of input
[n,p,T] = size(x);

% mean with respect to n
x_ = repmat(mean(x,1),[n,1,1]); 

% compute covariance matrix
S = pagemtimes(permute(x-x_,[2,1,3]),x-x_)/(n-1);

% comute pagewise diag
Ip = eye(p);
repmat(reshape(Ip,[p,p,1]),[1,1,T]);
diag_S = S .* Ip;

% compute correlation matrix
C2 = power(diag_S + epsilon,-1);
idx = (C2 <= inf);
C2 = C2 .*idx;
C = power(C2,0.5);
R = pagemtimes(pagemtimes(C,S),C);

% svd of R
[~,ll_,~] = pagesvd(R,"vector"); % x: n * p * T
ll = reshape(ll_,[p,T]); % [p,1,T] --> [p,T]



% compute actval

% A:[p,p,T]: compute the following item
% [l1-l1 l1-l2 l1-l3]
% [l2-l1 l2-l2 l2-l3]
% [l3-l1 l3-l3 l3-l3]
A = repmat(reshape(ll,[p,1,T]),[1,p,1])-repmat(reshape(ll,[1,p,T]), [p,1,1]);

% B:[p,p,T]: range to count up
% [0 0 0]
% [1 0 0]
% [1 1 0]
B = repmat(reshape(tril(ones(p,p))-eye(p),[p,p,1]),[1,1,T]); 

% L: [p,p,T]:
% [0      0     0]
% [l2-l1  0     0]
% [l3-l1 l3-l2  0]
L = A .* B;

% invL = L^(-1) (elementwise)
invL = power(L+epsilon, -1);
idx = (invL <= inf);
invL = invL .* idx;

% compute act_val

% part1
act_val_part1 = reshape(sum(invL,1),[p,T]); % [p,T]
act_val_part1 = act_val_part1(1:end-1,:); % [p-1,T] ignore the last item
act_val_part1 = act_val_part1 / (n-1);

% part2
act_val_part2 = adj_para * power(ll(2:end,:) - ll(1:end-1,:),-1) / (n-1);

% part3
act_val_part3 = (repmat(reshape(flip(1:(p-1)),[p-1,1]),[1,T]) / (n-1) - ones(p-1,T)) .* power(ll(1:end-1,:),-1);

% sum of par1 to part3
act_val = act_val_part1 + act_val_part2 + act_val_part3;

% act_val
act_val = -1./act_val;

% find the rank
act_val_truncated = act_val(1:upperbound,:);

idx = (act_val_truncated > 1+(p/n)^0.5);
r_act = max(repmat(reshape(1:upperbound,[upperbound,1]),[1,T]) .* idx,[],1); % columnwise max

end

