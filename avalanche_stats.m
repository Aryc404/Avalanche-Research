function out = avalanche_stats(S,D,bin_ms)
% S：雪崩大小；D：时长(bin)；bin_ms：bin宽(ms)；out：统计结果
out.size_distribution = pmf(S);
out.duration_distribution = pmf(D);
out.tau = out.size_distribution.fit.exponent;
out.alpha_D = out.duration_distribution.fit.exponent;
out.scaling = scale_fit(S,D,out.size_distribution.fit,out.duration_distribution.fit,bin_ms);
out.DCC = out.scaling.DCC;
end

function out = pmf(x)
% x：事件样本；out：分布结果
x = double(x(:));
[u,~,g] = unique(x); % u：不同取值；g：分组
if isempty(x)
    p = []; % 离散概率
else
    p = accumarray(g,1)/numel(x);
end
out.value = u;
out.probability = p;
out.fit = power_fit(x);
f = out.fit; % 幂律拟合结果
ok = u > 0 & p > 0; % 拟合点标记
if isfinite(f.xmin)
    ok = ok & u >= f.xmin & u <= f.xmax;
end
out.ols = struct('slope',NaN,'intercept',NaN,'used',ok);
if nnz(ok) >= 2
    c = polyfit(log(u(ok)),log(p(ok)),1); % 对数直线系数
    out.ols.slope = c(1);
    out.ols.intercept = c(2);
end
end

function f = power_fit(x)
% x：事件样本；f：幂律拟合结果
% MLE与KS筛选，未做重采样检验。
f = struct('support',[],'probability',[],'n_used',0,'n_total',numel(x), ...
    'xmin',NaN,'xmax',NaN,'exponent',NaN,'ks',NaN,'ks_limit',NaN, ...
    'valid',false);
u = unique(x); % 不同取值
if numel(x) < 50 || numel(u) < 5, return; end
idx = unique(round(linspace(1,numel(u)-4,min(20,numel(u)-4)))); % 候选下限索引
best = Inf; % 当前最小KS
for j = idx % 候选索引
    lo = u(j); % 拟合下限
    tail = x(x >= lo); % 尾部样本
    if numel(tail) < 50 || numel(unique(tail)) < 5, continue; end
    z = (lo:max(x))'; % 拟合取值区间
    lx = log(z/lo); % 对数相对取值
    L = sum(log(tail/lo)); % 对数样本和
    n = numel(tail); % 尾部样本数
    fun = @(k) k*L+n*log(sum(exp(-k*lx))); % 负对数似然，k为指数
    k = fminbnd(fun,0,8); % 拟合指数
    p = exp(-k*lx); % 模型概率
    p = p/sum(p);
    obs = accumarray(tail-lo+1,1,[numel(z) 1])/n; % 经验概率
    ks = max(abs(cumsum(obs)-cumsum(p))); % KS距离
    if ks < best
        best = ks;
        f.support = z;
        f.probability = p;
        f.n_used = n;
        f.xmin = lo;
        f.xmax = z(end);
        f.exponent = k;
        f.ks = ks;
        f.ks_limit = 1/sqrt(n);
    end
end
f.valid = isfinite(f.exponent) && f.ks < f.ks_limit && ...
    f.exponent >= 0.01 && f.exponent <= 7.99;
end

function out = scale_fit(S,D,fs,fd,bin_ms)
% S：大小；D：时长(bin)；fs/fd：两种分布拟合
% bin_ms：bin宽(ms)；out：标度结果
out = struct('duration_bins',[],'duration_ms',[],'mean_size',[],'count',[], ...
    'used',false(0,1),'intercept',NaN,'beta_fit',NaN,'beta_pred',NaN, ...
    'DCC',NaN,'valid',false);
if isempty(S), return; end
[D,~,g] = unique(D(:)); % g：时长分组
Sm = accumarray(g,S(:),[],@mean); % 各组平均大小
n = accumarray(g,1); % 各组事件数
ok = n >= 5; % 拟合类别标记
if isfinite(fd.xmin)
    ok = ok & D >= fd.xmin & D <= fd.xmax;
end
if isfinite(fs.xmin)
    ok = ok & Sm >= fs.xmin & Sm <= fs.xmax;
end
out.duration_bins = D;
out.duration_ms = D*bin_ms;
out.mean_size = Sm;
out.count = n;
out.used = ok;
if nnz(ok) >= 3
    c = polyfit(log(D(ok)),log(Sm(ok)),1); % 对数直线系数
    out.beta_fit = c(1);
    out.intercept = c(2);
end
tau = fs.exponent; % 大小分布指数
alpha = fd.exponent; % 时长分布指数
if isfinite(tau) && isfinite(alpha) && abs(tau-1) > 1e-3
    out.beta_pred = (alpha-1)/(tau-1);
end
if nnz(ok) >= 3 && fs.valid && fd.valid && isfinite(out.beta_pred)
    out.DCC = abs(out.beta_fit-out.beta_pred);
    out.valid = true;
end
end
