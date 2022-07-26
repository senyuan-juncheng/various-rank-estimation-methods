% bema rank rapid version by morimoto

% input: x: [n,p,T] --> n: sample, p: dim, T: replication
% alpha = 0.2;
% beta = 0.1;
% theta_seq = [0.1, 0.5:0.25:4];

function [r_bema,thresholds, theta_hat, sig2_hat] = bema_rank_rapid(y,alpha,beta,theta_seq)

B = 500; % bootstrapping

[n,p,T] = size(y);
pp = min([n,p]);

lower = round(alpha*pp);
upper = round((1-alpha)*pp);
k_seq = lower:upper;
K = length(k_seq);
J = length(theta_seq);

% get qt_k
display('compute qt_k');
tic
qt_k=GetQT1_fast(n,p,alpha,theta_seq);
toc


display('compute r_bema');
tic

% compute the eigenvalues of x
x_ = repmat(mean(y,1),[n,1,1]); % mean with respect to n

if(n>p)
    S = pagemtimes(permute(y-x_,[2,1,3]),y-x_)/(n-1);
else
    S = pagemtimes(y-x_,permute(y-x_,[2,1,3]))/(n-1);
end
[~,L,~] = pagesvd(S,"vector"); % S: [min(n,p),min(n,p),T]

ll = reshape(L,[min(n,p),T]); % [min(n,p),1,T] --> [min(n,p),T]

if(p>n) % When p>n, eigenvalues of 0 are cancelled. So we supplement it.
    ll = [ll;zeros(p-n,T)];
end

% truncate ll
ll_cut = ll(k_seq,:); % [K,T]

% extend ll_cut
ll_cut_extend = reshape(ll_cut,[K,1,T]);
ll_cut_extend = repmat(ll_cut_extend,[1,J,1]);

% extend qt_k
qt_k_extend = reshape(qt_k, [K,J,1]);
qt_k_extend = repmat(qt_k_extend,[1,1,T]); % [K,J,T]

% compute sig2_seq
sig2_seq = sum(qt_k_extend .* ll_cut_extend,1) ./ sum(qt_k_extend.^2,1); % [1,J,T]
% compute v_seq
v_seq = sum((ll_cut_extend-qt_k_extend.*repmat(sig2_seq,[K,1,1])).^2,1); % [1,J,T]

% reshape sig2_seq and v_seq into matrices
sig2_seq = reshape(sig2_seq,[J,T]);
v_seq = reshape(v_seq,[J,T]);
[~, v_ind] = min(v_seq); % [T];

% extend theta_seq
theta_seq_extend = reshape(theta_seq,[J,1]);
theta_seq_extend = repmat(theta_seq_extend,[1,T]);

% find theta_hat and sig2_hat
test = repmat([1:J]',[1,T]);
idx = (test == v_ind);

theta_hat = sum(theta_seq_extend .* idx,1); % [1,T]
sig2_hat = sum(sig2_seq .* idx, 1); % [1,T]

%lam_data = zeros(M,1); % initialize

z = normrnd(0,1,[n,p,B,T]);
sig2_hat_extend = reshape(sig2_hat,[1,1,1,T]);
sig2_hat_extend = repmat(sig2_hat_extend,[n,p,B,1]);

D = zeros(1,p,B,T);
for t=1:T
    theta_temp = theta_hat(t);
    D(1,:,:,t) = gamrnd(theta_temp,1/theta_temp,[p,B]);
end
D = repmat(D,[n,1,1,1]);

y = z .* power(sig2_hat_extend, 1/2) .* power(D, 1/2);

temp = y;

[~,ll_bootstrap,~] = pagesvd(y,"vector"); % [p,1,B,T];
ll_bootstrap = reshape(ll_bootstrap,[p,B,T]);
ll_bootstrap = ll_bootstrap.^2 / n;
ll_bootstrap_top = ll_bootstrap(1,:,:);
ll_bootstrap_top = reshape(ll_bootstrap_top, [B,T]);
thresholds = quantile(ll_bootstrap_top, 1-beta);

r_bema = sum(ll > thresholds);
toc
end









