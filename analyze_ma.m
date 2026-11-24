function out = analyze_ma(X,dt_ms)
% X：E放电矩阵；dt_ms：步长(ms)；out：结果
if nargin < 2, dt_ms = 1; end

%% 分箱
[N,T] = size(X); % N：神经元数；T：步数
B = round(40/dt_ms); % 每bin步数
Nb = floor(T/B); % bin数
C = false(N,Nb); % 各bin发放标记
for j = 1:1000:Nb % 分块起始bin
    k = j:min(j+999,Nb); % 当前bin编号
    Y = reshape(X(:,(j-1)*B+1:k(end)*B),N,B,numel(k)); % 当前数据块
    C(:,k) = reshape(any(Y,2),N,numel(k));
end
A = sum(C,1); % 活跃神经元数
q = prctile(A,35); % 活动阈值
on = A > q; % 超阈标记

%% 雪崩
d = diff([false on false]); % 事件边界差分
t0 = find(d == 1); % 起始bin
t1 = find(d == -1)-1; % 结束bin
D = t1-t0+1; % 时长(bin)
cs = [0 cumsum(A)]; % 累计活动
S = cs(t1+1)-cs(t0); % 雪崩大小
ok = t0 > 1 & t1 < Nb; % 完整事件标记

%% 输出
out = avalanche_stats(S(ok),D(ok),B*dt_ms);
out.bin_ms = B*dt_ms;
out.time_s = ((1:Nb)-0.5)*out.bin_ms/1000;
out.activity = A;
out.threshold = q;
out.threshold_percentile = 35;
out.active = on;
out.recorded_ms = Nb*out.bin_ms;
out.discarded_tail_ms = (T-Nb*B)*dt_ms;
out.events = struct('start_bin',t0,'end_bin',t1, ...
    'start_ms',(t0-1)*out.bin_ms,'end_ms',t1*out.bin_ms, ...
    'S',S,'D_bins',D,'D_ms',D*out.bin_ms,'complete',ok);
out.complete_count = nnz(ok);
out.censored_count = nnz(~ok);
end
