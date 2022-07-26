%
% aic/bic rank fast version by Toshinari Morimoto
%

% last update 2022/07/20 (Wed)


% You no longer need to call the function repeatedly.
% This function finishes the experiments at once.

% Input:
% x: n x p x T
% n: sample size
% p: data dimension
% T: number of trials (ith experiment i=1...T)

function [r_aic,r_bic, aic, bic, likelihood] = aic_bic_rank_fast(x,upperbound)

% These are used in taking elementwise inverse or elementwise log of matrices containing 0. 
% (want to define log0 and 0^-1 as 0 to ignore them)
epsilon = 10^-20;
inf     = 10^16;

% size of input
[n,p,T] = size(x);

if(upperbound > min(n-1,p))
    upperbound = min(n-1,p);
end


x_ = repmat(mean(x,1),[n,1,1]); % mean with respect to n


[~,L,~] = pagesvd((x-x_)/sqrt(n-1),"vector"); % x: n * p * T


ll_ = L.^2; % [min(n,p),1,T]
ll = reshape(ll_,[min(n,p),T]); % [min(n,p),1,T] --> [min(n,p),T]

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

% penalty term of aic and bic
b_aic = -power(0:(p-1),2)'/2 + (p+1/2)*(0:(p-1))' + p+1;
b_aic = repmat(b_aic,[1,T]);
b_bic = 0.5 * log(n) * b_aic;

% compute penalized likelihood
aic = likelihood-b_aic/n;
bic = likelihood-b_bic/n;

% get the optimal ranks
[~,r_aic] = max(aic(1:upperbound,:),[],1);
r_aic = r_aic-1;

[~,r_bic] = max(bic(1:upperbound,:),[],1);
r_bic = r_bic-1;

end
