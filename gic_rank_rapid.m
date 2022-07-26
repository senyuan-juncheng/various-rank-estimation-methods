%
% gic rank rapid version by Toshinari Morimoto
%

% last update: 2022/07/22 (Fri)

% You no longer need to call the function repeatedly.
% This function finishes the experiments at once.

% Input:
% x: n x p x T
% n: sample size
% p: data dimension
% T: number of trials (ith experiment i=1...T)

function [r_gic,gic,likelihood, b] = gic_rank_rapid(x,upperbound)

% These are used in taking elementwise inverse or elementwise log of matrices containing 0. 
% (want to define log0 and 0^-1 as 0 to ignore them)
epsilon = 10^-18;
inf     = 10^15;

% size of input
[n,p,T] = size(x);

% number of nonzero eigenvalues
m = min(n-1,p);

% check upperbound
if(upperbound > min(n-1,p))
    upperbound = min(n-1,p);
end


x_ = repmat(mean(x,1),[n,1,1]); % mean with respect to n

if(n>p)
    S = pagemtimes(permute(x-x_,[2,1,3]),x-x_)/(n-1);
else
    S = pagemtimes(x-x_,permute(x-x_,[2,1,3]))/(n-1);
end

[~,L,~] = pagesvd(S,"vector"); % S: [min(n,p),min(n,p),T]

ll = reshape(L,[min(n,p),T]); % [min(n,p),1,T] --> [min(n,p),T]

if(p>n) % When p>n, eigenvalues of 0 are cancelled. So we supplement it.
    ll = [ll;zeros(p-n,T)];
end

% compute the sums of eigenvalues starting from the index i.
% i.e. [l1 +...+ lp; l2+...+lp; ... ;lp] (and T colums together)
sum_eig_vals = triu(ones(p,p)) * ll; % [p,T]

% compute sigma_r^2 (MLE of sigma^2 when the rank is r=0 to p-1)
% note: we want to divide by [p;p-1;...;1] on each raw
sigma_r2 = sum_eig_vals ./ repmat(flip(1:p)',[1,T]); % [p,T]


% compute sum of log(lj) from the index j=0 to r-1
% i.e. [0;log(l1);log(l1)+log(l2);...;log(l1)+...+log(lp-1)]
log_ll = log(ll+epsilon);
idx = (log_ll >= -log(inf));
log_ll = log_ll .* idx;
sum_log_eig_vals = [zeros(1,p);tril(ones(p-1,p))]*log_ll; % [p,T] 


% compute likelihood
log_sigma_r2 = log(sigma_r2 + epsilon);
idx = (log_sigma_r2 >= -log(inf));
log_sigma_r2 = log_sigma_r2 .* idx;

likelihood =  sum_log_eig_vals + repmat(flip(1:p)',[1,T]) .* log_sigma_r2;
likelihood = -1/2 * likelihood; % [p,T] -1/2(sum_j=1^r log(lj) + (p-r)log(sigma_r^2))


% From here we compute b_GIC
% but the computation is complex

% compute D(:,:,t)=[li-lj] (i:raw j:column)
% i.e [l1-l1 l1-l2 l1-l3]
%     [l2-l1 l2-l2 l2-l3]
%     [l3-l1 l3-l2 l3-l3] if p=3 
% D: [p,p,T] and extend as E
Jp = repmat(reshape(eye(p),[1,p,p]),[p,1,1]); % page1: [1p Op], page2: [0p 1p Op], page3:[0p 0p 1p Op] ...
Ip = repmat(reshape(eye(p),[p,p,1]),[1,1,p]); % each page is identity Ip
D = pagemtimes(Jp-Ip,repmat(reshape(ll,[p,T,1]),[1,1,p]));
D = permute(D,[3,1,2]); % [p,p,T] : Dij(T)= [li-lj] (T)

% compute elementwise reciprocal of D
recD = power(D + epsilon,-1);
idx1 = (recD <= inf);
idx2 = (-inf <= recD);
recD = idx1 .* idx2 .* recD;

% Count up the range of double series. 
% In the computation of b_GIC, we have double summations related to li-lj.
% Here we are define the range we take a summation. (0: ignore, 1:summation)
range = pagemtimes(reshape(triu(ones(p,p)-eye(p)),[p,1,p]),reshape(tril(ones(p,p)),[1,p,p])); % [p,p,p]
tr_range = range(:,1:m,:); %  [p,m,p] (we do this because we will multiply by l1-lp later and then lm+1,... are cancelled)
tr_range = reshape(tr_range,[p,m,p,1]); % [p,m,p] -> [p,m,p,1]
tr_range = repmat(tr_range,[1,1,1,T]); % [p,m,p,T]

% E: truncate recD on the 2nd axis and then copy along the 3rd axis
% finally extract over the range
E = recD(:,1:m,:); % [p,m,T]
E = reshape(E,[p,m,1,T]); % [p,m,1,T]
E = repmat(E,[1,1,p,1]);
E = E.* tr_range; % [p,m,p,T]


% compute G: [p,p,p,T]: lk/(lj-lk) j:row k:column for the range of summation
tr_ll = ll(1:m,:); % truncated eigenvalues
k = reshape(tr_ll,[1,m,T]); % [1,m,T]
k = repmat(k,[p,1,1]); % [p,m,T]
k = reshape(k,[p,m,1,T]); % [p,m,1,T] 
k = repmat(k,[1,1,p,1]); % [p,m,p,T]

G = E .* k;

% compute H: [p,p,T]
H = reshape(sum(G,2),[p,p,T]); % sum over the 2nd axis for G

% compute b_part1
inv_sigma_r2 = 1 ./ sigma_r2;

% compute: li/sigmaj^2-1
v = repmat(reshape(ll,[p,1,T]),[1,p,1]) .* repmat(reshape(inv_sigma_r2,[1,p,T]),[p,1,1]) - 1;

b_part1 = reshape(sum(v .* H,1),[p,T]); % [p,T]

% compute b_part2 and b_part3
sum_square_eig_vals = triu(ones(p,p)) * ll.^2;
b_part2 = (sum_square_eig_vals ./ repmat(flip(1:p)',[1,T])) ./ sigma_r2 .^2;
b_part3 = repmat((power(1:p,2) - (1:p))',[1,T]) /2; % just for r(r-1)/2 

% compute b_GIC
b = b_part1+b_part2+b_part3;

% compute penalized likelihood
gic = likelihood - b/n;

% number of estimated rank
[~,r_gic] = max(gic(1:upperbound,:),[],1);
r_gic = r_gic-1;


end

