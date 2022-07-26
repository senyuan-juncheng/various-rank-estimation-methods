% used in bema_rank_fast

function [qt_k] = GetQT1_fast(n,p,alpha,theta_seq)

% number of replication
B = 500;

pp = min([n,p]);

lower = round(alpha*pp);
upper = round((1-alpha)*pp);
k_seq = lower:upper;

K = length(k_seq);
J = length(theta_seq);

% quantile at possible theta values
qt_k = zeros(K,J);

for j = 1:J
    theta = theta_seq(j);
    z = normrnd(0,1,[n,p,B]);
    s = gamrnd(theta,1/theta,[1,p,B]);
    s = repmat(s,[n,1,1]);
    x = z .* power(s,1/2);
    x = reshape(x,[n,p,B]);
    S = pagemtimes(x,permute(x,[2,1,3]));
    [~,L,~] = pagesvd(S,"vector"); % S: [n,1,B]
    ll = reshape(L,[n,B]); % [n,B] 
    llave = mean(ll,2); % [n,1]
    qt_k(:,j) = llave(lower:upper,:) / n;
end

end
