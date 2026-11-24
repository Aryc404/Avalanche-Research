function out = analyze_stepp(X,W,dt_ms) % X：放电矩阵；W：权重(后×前)；dt_ms：步长(ms)；out：结果
% Stepp：连接追踪、branching ratio、ISI-DFA。
if nargin < 3
    dt_ms = 1;
end

%% 连接追踪
X = logical(X);
[N,T] = size(X); % N：神经元数；T：步数
A = W > 0; % 连接矩阵(后×前)
A(1:N+1:end) = false;
K = nnz(X); % spike总数
lags = 2:4; % 追踪延迟(步)

S = zeros(K,1); % 各雪崩大小
t0 = zeros(K,1,'uint32'); % 起始步
i0 = zeros(K,1,'uint16'); % 起始神经元
t1 = zeros(K,1,'uint32'); % 最后放电步
lab = cell(N,5); % spike标签缓存
prev = cell(1,5); % 发放细胞缓存
nr = 0; % 雪崩数

br = NaN(1,T); % 分支比
npre = NaN(1,T); % 上游计数
npost = NaN(1,T); % 下游计数

for t = 1:T % 当前时间步
    slot = mod(t-1,5)+1; % 当前缓存槽
    old = prev{slot}; % 待清理细胞
    if ~isempty(old)
        lab(old,slot) = {[]};
    end
    ids = find(X(:,t)); % 当前发放细胞
    prev{slot} = ids;

    for i = ids' % 当前神经元
        tags = cell(numel(lags),1); % 各延迟父标签
        for k = 1:numel(lags) % 延迟索引
            tp = t-lags(k); % 上游时间步
            if tp < 1
                continue;
            end
            sp = mod(tp-1,5)+1; % 上游缓存槽
            cand = prev{sp}; % 候选父细胞
            parents = cand(A(i,cand)); % 已连接父细胞
            tags{k} = vertcat(lab{parents,sp});
        end

        labels = unique(vertcat(tags{:})); % 继承的雪崩标签
        if isempty(labels)
            nr = nr+1;
            labels = uint32(nr);
            t0(nr) = t;
            i0(nr) = i;
        end
        S(labels) = S(labels)+1;
        t1(labels) = t;
        lab{i,slot} = labels;
    end

    if t >= 5 && t <= T-4 && ~isempty(ids)
        pre = any(X(:,t-lags),2); % 窗内上游活跃
        post = any(X(:,t+lags),2); % 窗内下游活跃
        npre(t) = sum(A(ids,pre),'all');
        npost(t) = sum(A(post,ids),'all');
        if npre(t) > 0
            br(t) = npost(t)/npre(t);
        end
    end
end

%% 分布与输出
S = S(1:nr);
t0 = t0(1:nr);
i0 = i0(1:nr);
t1 = t1(1:nr);
cut = double(t1)+max(lags) > T; % 右端截断标记
out.parameters = struct('phi',1,'Delta',3,'lags',lags,'dt_ms',dt_ms); % 时间窗参数
out.G = A'; % 连接图(前×后)
out.tracking = struct('size',S,'start_step',t0, ...
    'start_neuron',i0,'last_step',t1,'right_censored',cut, ...
    'complete_sizes',S(~cut),'total_count',nr,'complete_count',nnz(~cut)); % 雪崩记录
out.size_distribution = size_pmf(S(~cut)); % 大小分布
out.branching = struct('time_ms',(0:T-1)*dt_ms,'value',br, ...
    'pre_count',npre,'post_count',npost, ...
    'valid_count',nnz(isfinite(br)),'mean',mean(br,'omitnan')); % 分支比记录
out.dfa = isi_dfa(X,dt_ms); % 各神经元DFA
end

%% 大小分布
function pdf = size_pmf(S) % S：雪崩大小；pdf：大小分布
[s,~,g] = unique(S); % s：不同大小；g：分组编号
n = zeros(size(s)); % 各大小次数
if ~isempty(s)
    n = accumarray(g,1,[numel(s) 1]);
end
p = n/max(numel(S),1); % 大小概率
pdf = struct('S',s,'P',p,'counts',n, ...
    'slope',NaN,'intercept',NaN,'r_squared',NaN,'fit_valid',false, ...
    'fit_min',NaN,'fit_max',NaN,'n_avalanches',numel(S));
if numel(s) >= 3
    x = log10(s); % 对数大小
    y = log10(p); % 对数概率
    coef = polyfit(x,y,1); % 直线系数
    fit = polyval(coef,x); % 拟合值
    ss = sum((y-mean(y)).^2); % 总平方和
    pdf.slope = coef(1);
    pdf.intercept = coef(2);
    if ss > 0
        pdf.r_squared = 1-sum((y-fit).^2)/ss;
    end
    pdf.fit_valid = true;
    pdf.fit_min = min(s);
    pdf.fit_max = max(s);
end
end

%% ISI-DFA
function dfa = isi_dfa(X,dt_ms) % X：放电矩阵；dt_ms：步长(ms)；dfa：DFA结果
N = size(X,1); % 神经元数
dfa.ISI_ms = cell(N,1); % 各细胞ISI(ms)
dfa.scales = cell(N,1); % 分段长度
dfa.F = cell(N,1); % 均方根波动(ms)
dfa.F_squared = cell(N,1); % 波动平方(ms²)
dfa.ISI_count = zeros(N,1); % 各细胞ISI数
dfa.alpha = NaN(N,1); % DFA指数
dfa.intercept = NaN(N,1); % 对数拟合截距
dfa.r_squared = NaN(N,1); % 拟合R²
dfa.scale_span_decades = NaN(N,1); % 尺度跨度(数量级)
dfa.limited_span = true(N,1); % 短跨度标记

for i = 1:N % 神经元编号
    isi = diff(find(X(i,:)))'*dt_ms; % 放电间隔(ms)
    dfa.ISI_ms{i} = isi;
    dfa.ISI_count(i) = numel(isi);
    lmax = floor(numel(isi)/4); % 最长分段
    if lmax < 4
        continue;
    end
    ls = unique(round(logspace(log10(4),log10(lmax),12))); % 段长序列
    cum = cumsum(isi-mean(isi)); % 去均值累积序列
    F = zeros(size(ls)); % 各尺度波动
    for k = 1:numel(ls) % 尺度索引
        l = ls(k); % 当前段长
        M = floor(numel(isi)/l); % 片段数量
        z = reshape(cum(1:M*l),l,M); % 累积序列分段
        x = (1:l)'; % 拟合横坐标
        x = x-mean(x);
        a = (x'*z)/(x'*x); % 各段斜率
        fit = mean(z,1)+x*a; % 拟合值
        r = z-fit; % 去趋势残差
        F(k) = sqrt(mean(r(:).^2));
    end
    dfa.scales{i} = ls;
    dfa.F{i} = F;
    dfa.F_squared{i} = F.^2;
    ok = isfinite(F) & F > 0; % 有效尺度标记
    if nnz(ok) < 4 || all(isi == isi(1))
        continue;
    end
    x = log10(ls(ok));
    y = log10(F(ok)); % 对数波动
    coef = polyfit(x,y,1); % 直线系数
    fit = polyval(coef,x);
    ss = sum((y-mean(y)).^2); % 总平方和
    dfa.alpha(i) = coef(1);
    dfa.intercept(i) = coef(2);
    if ss > 0
        dfa.r_squared(i) = 1-sum((y-fit).^2)/ss;
    end
    dfa.scale_span_decades(i) = max(x)-min(x);
    dfa.limited_span(i) = max(x)-min(x) < 1;
end
dfa.valid_count = nnz(isfinite(dfa.alpha)); % 有效拟合数
dfa.adequate_span_count = nnz(isfinite(dfa.alpha) & ~dfa.limited_span); % 足够跨度数
end
