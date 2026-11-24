function out = analyze_ikeda(X,dt_ms,Tm)
% X：E放电矩阵；dt_ms：步长(ms)；Tm：平均间隔(ms)；out：结果
[N,T] = size(X); % N：神经元数；T：步数
K = nnz(X); % spike总数
S = zeros(K,1,'uint32'); % 雪崩大小
t0 = zeros(K,1,'uint32'); % 起始步
t1 = zeros(K,1,'uint32'); % 结束步
n = 0; % 雪崩数

%% 相邻间隔分组
for b = 1:100000:T % 数据块起点
    c = sum(X(:,b:min(b+99999,T)),1); % 各步spike数
    for j = find(c) % 块内放电步
        t = b+j-1; % 全局步
        if Tm > 0
            if n == 0 || (t-double(t1(n)))*dt_ms >= Tm
                n = n+1;
                S(n) = c(j);
                t0(n) = t;
            else
                S(n) = S(n)+uint32(c(j));
            end
            t1(n) = t;
        else
            q = n+(1:c(j)); % 同步独立事件索引
            S(q) = 1;
            t0(q) = t;
            t1(q) = t;
            n = n+c(j);
        end
    end
end
S = S(1:n);
t0 = t0(1:n);
t1 = t1(1:n);
ok = t0 > 1; % 完整事件标记
if Tm > 0
    ok = (double(t0)-1)*dt_ms >= Tm & (double(t1)-1)*dt_ms+Tm <= T*dt_ms;
end

%% 分布与偏差
out = fit_dist(double(S(ok)),N);
out.interval_ms = Tm;
out.time_resolution_ms = dt_ms;
out.events = struct('S',S,'start_ms',(double(t0)-1)*dt_ms, ...
    'end_ms',(double(t1)-1)*dt_ms,'complete',ok);
out.complete_count = nnz(ok);
out.censored_count = nnz(~ok);
end

function out = fit_dist(S,N)
% S：完整雪崩大小；N：神经元数；out：拟合结果
[u,~,g] = unique(S); % u：不同大小；g：大小分组
if isempty(S)
    p = []; % 经验概率
else
    p = accumarray(g,1)/numel(S);
end
out.size_distribution = struct('value',u,'probability',p);
out.fit = struct('slope',NaN,'intercept',NaN,'xmin',NaN, ...
    'xmax',NaN,'valid',false);
out.deviation = struct('S',[],'value',[],'upper',NaN,'lower',NaN);
out.DeltaCr = NaN;
if isempty(u), return; end
hi = min(N,max(u)); % 拟合上限
v = u(u <= hi); % 可拟合大小
best = Inf; % 最小拟合误差
for j = 1:numel(v)-9 % 候选下限索引
    lo = v(j); % 拟合下限
    if hi/lo < 10, continue; end
    use = u >= lo & u <= hi; % 非零拟合点
    x = log10(u(use)); % 对数大小
    y = log10(p(use)); % 对数概率
    a = polyfit(x,y,1); % 回归系数
    err = sum((y-polyval(a,x)).^2); % 残差平方和
    % 原文未给最少点数；取10点、1数量级，避免只拟尾部。
    if err < best
        best = err;
        out.fit = struct('slope',a(1),'intercept',a(2), ...
            'xmin',lo,'xmax',hi,'valid',true);
    end
end
if ~out.fit.valid, return; end
z = (out.fit.xmin:hi)'; % 偏差取值范围
emp = zeros(size(z)); % 范围内经验概率
[has,k] = ismember(z,u); % has：存在标记；k：经验索引
emp(has) = p(k(has));
fit = 10.^polyval([out.fit.slope out.fit.intercept],log10(z)); % 拟合概率
r = emp-fit; % 概率偏差
up = sum(max(r,0)); % 正偏差和
low = sum(min(r,0)); % 负偏差和
out.deviation = struct('S',z,'value',r,'upper',up,'lower',low);
if abs(up) >= abs(low)
    out.DeltaCr = up;
else
    out.DeltaCr = low;
end
end
