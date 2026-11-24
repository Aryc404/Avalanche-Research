function result = avalanche_test()
% result：实验结果
%% 参数
N_p = 10000; % patch数
N_t = 400; % 每patch步数
dt_ms = 1; % 步长(ms)
seed = 20261123; % 随机种子
T = N_p*N_t; % 总步数

%% 读入
base = fileparts(mfilename('fullpath')); % 实验目录
model = fullfile(base,'..','Ruslim network','bio-spiking-v1-learning-main'); % 原模型目录
file = fullfile(base,'pcb_shuffle_ad_paper_cn1_run01_batch1200.mat'); % 检查点路径
cp = load(file,'checkpoint'); % 检查点数据
cp = cp.checkpoint;
net = cp.state; % 权重与阈值
rng0 = rng; % 原随机状态
path0 = path; % 原搜索路径
cleanup = onCleanup(@() restore(rng0,path0)); % 退出恢复状态
addpath(model);
rng(seed,'twister');

img = load(fullfile(base,'VanHateren_DoG_small.mat'),'IMAGES_DoG'); % DoG图像
N_X = size(net.W_EX,2)/2; % patch像素数
R = min(generate_image_patches(img.IMAGES_DoG,N_X,N_p),100); % 输入率(Hz)
clear img

%% 连续仿真
[X,M,G] = run_net(net,R,N_t,dt_ms); % X：E记录；M：spike数；G：间隔和(ms)
Tm = G/max(M-1,0); % 群体平均间隔(ms)
cfg = struct('checkpoint_file',file,'checkpoint_batch',cp.batch, ... % 实验设置
    'seed',seed,'n_patches',N_p,'patch_ms',N_t*dt_ms, ...
    'dt_ms',dt_ms,'total_ms',T*dt_ms);
folder = fullfile(base,'results','avalanche_test', ... % 保存目录
    char(datetime('now','Format','yyyyMMdd_HHmmss_SSS')));
mkdir(folder);
result = struct('protocol',cfg,'frozen_state',net, ...
    'patch_rates_Hz',R,'patch_start_step',1:N_t:T,'spikes_E',X, ...
    'spike_count',M,'interval_sum_ms',G,'mean_interval_ms',Tm);
result.output_file = fullfile(folder,'result.mat');
result.stepp_figure = fullfile(folder,'stepp_summary.png');
result.ma_figure = fullfile(folder,'ma_summary.png');
result.liang_figure = fullfile(folder,'liang_summary.png');
result.ikeda_figure = fullfile(folder,'ikeda_summary.png');
save(result.output_file,'result','-v7.3');

%% 统计
result.stepp = analyze_stepp(X,net.W_EE,dt_ms);
result.ma = analyze_ma(X,dt_ms);
result.liang = analyze_liang(X,dt_ms,Tm);
result.ikeda = analyze_ikeda(X,dt_ms,Tm);
save(result.output_file,'result','-v7.3');

%% 绘图
plot_stepp(result);
plot_bins(result.ma,result.ma_figure,'Ma');
plot_bins(result.liang,result.liang_figure,'Liang');
plot_ikeda(result.ikeda,result.ikeda_figure);
fprintf('%s\n',folder);
end

function [X,M,G] = run_net(net,R,N_t,dt_ms)
% net：网络参数；R：输入率(Hz)；N_t：每patch步数；dt_ms：步长(ms)
% X：E放电矩阵；M：spike总数；G：间隔和(ms)
N_E = size(net.W_EE,1); % E神经元数
N_I = size(net.W_II,1); % I神经元数
N_p = size(R,2); % patch数
N_L = size(R,1); % LGN通道数
dt = dt_ms/1000; % 步长(s)
a_m = exp(-dt/10e-3); % 膜电位衰减系数
V_E = zeros(N_E,1); % E膜电位
V_I = zeros(N_I,1); % I膜电位
X = false(N_E,N_p*N_t);
M = 0;
G = 0;
tp = 0; % 上次群体放电步

for p = 1:N_p % patch编号
    P_X = R(:,p)*dt; % LGN单步发放概率
    for j = 1:N_t % patch内步编号
        t = (p-1)*N_t+j; % 全局步编号
        S_X = sparse(double(rand(N_L,1)<P_X)); % LGN当前放电
        S_E = V_E>net.theta_E; % E当前放电
        S_I = V_I>net.theta_I; % I当前放电
        X(:,t) = S_E;
        c = nnz(S_E); % 当前E spike数
        if c > 0
            if M > 0, G = G+(t-tp)*dt_ms; end
            M = M+c;
            tp = t;
        end
        V_E(S_E) = 0;
        V_I(S_I) = 0;
        E = sparse(double(S_E)); % E放电向量
        I = sparse(double(S_I)); % I放电向量
        V_E = V_E*a_m+net.W_EX*S_X+net.W_EE*E-net.W_EI*I+randn(N_E,1);
        V_I = V_I*a_m+net.W_IX*S_X+net.W_IE*E-net.W_II*I+randn(N_I,1);
        V_E = max(V_E,-10);
        V_I = max(V_I,-10);
    end
    if mod(p,1000) == 0, fprintf('Patch %d / %d\n',p,N_p); end
end
end

function restore(rng0,path0)
% rng0：原随机状态；path0：原搜索路径
rng(rng0);
path(path0);
end

function plot_stepp(r)
% r：实验结果
s = r.stepp; % Stepp结果
f = figure('Visible','off','Color','w','Theme','light','Position',[100 100 1450 440]); % 图窗
tl = tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact'); % 子图布局

ax = nexttile(tl,1); % 当前坐标轴
d = s.size_distribution; % 大小分布
loglog(ax,d.S,d.P,'o','Color',[0 0.35 0.8],'MarkerSize',4);
hold(ax,'on');
if d.fit_valid
    loglog(ax,d.S,10.^polyval([d.slope d.intercept],log10(d.S)), ...
        '-','Color',[0.8 0.25 0.1],'LineWidth',1.4);
    legend(ax,{'Data','OLS fit'},'Location','best');
end
title(ax,'A  Size distribution');
subtitle(ax,sprintf('k = %.3f; n = %d',d.slope,s.tracking.complete_count));
xlabel(ax,'S (spikes)'); ylabel(ax,'P(S)');

ax = nexttile(tl,2);
b = s.branching; % 分支比结果
B = round(10000/r.protocol.dt_ms); % 显示窗口步数
Nb = ceil(numel(b.value)/B); % 显示窗口数
v = NaN(1,Nb); % 分块平均分支比
t = zeros(1,Nb); % 窗口中心(s)
for j = 1:Nb % 显示窗口编号
    k = (j-1)*B+1:min(j*B,numel(b.value)); % 当前窗口步
    v(j) = mean(b.value(k),'omitnan');
    t(j) = (mean([k(1) k(end)])-1)*r.protocol.dt_ms/1000;
end
plot(ax,t,v,'Color',[0 0.35 0.8],'LineWidth',1.3);
hold(ax,'on');
yline(ax,1,'--','Color',[0.8 0.25 0.1]);
title(ax,'B  Branching ratio');
subtitle(ax,sprintf('Mean = %.3f',b.mean));
xlabel(ax,'Time (s)'); ylabel(ax,'s (10-s mean)');
xlim(ax,[0 r.protocol.total_ms/1000]);

ax = nexttile(tl,3);
ok = isfinite(s.dfa.alpha) & ~s.dfa.limited_span; % 尺度足够的拟合
if any(ok)
    histogram(ax,s.dfa.alpha(ok),20,'FaceColor',[0 0.35 0.8],'EdgeColor','none');
else
    text(ax,0.5,0.5,'Insufficient scale range','Units','normalized', ...
        'HorizontalAlignment','center');
end
title(ax,'C  DFA exponent distribution');
subtitle(ax,sprintf('%d / %d neurons; span >= 1 decade',nnz(ok),size(r.spikes_E,1)));
xlabel(ax,'\alpha_{DFA}'); ylabel(ax,'E neurons');
title(tl,'Stepp');
exportgraphics(f,r.stepp_figure,'Resolution',180);
close(f);
end

function plot_bins(m,file,name)
% m：分箱结果；file：图片路径；name：方法名
f = figure('Visible','off','Color','w','Theme','light','Position',[100 100 1100 820]); % 图窗
tl = tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact'); % 子图布局
ax = nexttile(tl,1); % 当前坐标轴
hold(ax,'on');
tmax = min(4,m.recorded_ms/1000); % 显示时长(s)
use = m.time_s <= tmax; % 显示bin
if any(use)
    ymax = max([m.activity(use) 1])*1.15; % 纵轴上限
    for k = find(m.events.start_ms < tmax*1000) % 显示事件编号
        t0 = m.events.start_ms(k)/1000; % 事件起点(s)
        t1 = min(m.events.end_ms(k)/1000,tmax); % 事件终点(s)
        patch(ax,[t0 t1 t1 t0],[0 0 ymax ymax],[0 0.35 0.8], ...
            'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off');
    end
    plot(ax,m.time_s(use),m.activity(use),'Color',[0.15 0.15 0.15],'LineWidth',1);
    xlim(ax,[0 tmax]); ylim(ax,[0 ymax]);
end
if isfield(m,'threshold')
    yline(ax,m.threshold,'--','Color',[0 0.35 0.8]);
    ylabel(ax,'Active E neurons');
    subtitle(ax,sprintf('Bin = %.3g ms; threshold = %.3g',m.bin_ms,m.threshold));
else
    ylabel(ax,'Spikes / bin');
    subtitle(ax,sprintf('Bin = %.4g ms',m.bin_ms));
end
title(ax,'A  Avalanche detection'); xlabel(ax,'Time (s)');

ax = nexttile(tl,2);
plot_dist(ax,m.size_distribution,'S');
title(ax,'B  Size distribution');
subtitle(ax,sprintf('τ = %.3f; n = %d',m.tau,m.complete_count));

ax = nexttile(tl,3);
plot_dist(ax,m.duration_distribution,'D (bins)');
title(ax,'C  Duration distribution');
subtitle(ax,sprintf('α = %.3f',m.alpha_D));

ax = nexttile(tl,4);
c = m.scaling; % 大小-时长标度
loglog(ax,c.duration_bins,c.mean_size,'o','Color',[0.6 0.6 0.6],'MarkerSize',4);
hold(ax,'on');
if any(c.used)
    loglog(ax,c.duration_bins(c.used),c.mean_size(c.used),'o', ...
        'Color',[0 0.35 0.8],'MarkerFaceColor',[0 0.35 0.8],'MarkerSize',4);
end
if isfinite(c.beta_fit)
    x = logspace(log10(min(c.duration_bins(c.used))),log10(max(c.duration_bins(c.used))),100); % 拟合横坐标
    h1 = loglog(ax,x,exp(c.intercept)*x.^c.beta_fit,'-', ... % 实测拟合线
        'Color',[0 0.35 0.8],'LineWidth',1.4);
    if isfinite(c.beta_pred)
        D0 = exp(mean(log(c.duration_bins(c.used)))); % 对齐时长
        S0 = exp(c.intercept)*D0^c.beta_fit; % 对齐大小
        h2 = loglog(ax,x,S0*(x/D0).^c.beta_pred,'--', ... % 预测斜率线
            'Color',[0.8 0.25 0.1],'LineWidth',1.4);
        legend(ax,[h1 h2],{'Fitted','Predicted'},'Location','best');
    end
end
title(ax,'D  Mean size versus duration');
subtitle(ax,sprintf('β_{fit}=%.3f; β_{pred}=%.3f; DCC=%.3f', ...
    c.beta_fit,c.beta_pred,c.DCC));
xlabel(ax,'D (bins)'); ylabel(ax,'Mean S');
title(tl,name);
exportgraphics(f,file,'Resolution',180);
close(f);
end

function plot_dist(ax,d,xl)
% ax：坐标轴；d：分布结果；xl：横轴名
loglog(ax,d.value,d.probability,'o','Color',[0 0.35 0.8],'MarkerSize',4);
hold(ax,'on');
f = d.fit; % 幂律拟合
if ~isempty(f.support)
    loglog(ax,f.support,f.probability*f.n_used/f.n_total, ...
        '-','Color',[0.8 0.25 0.1],'LineWidth',1.4);
    if f.valid
        lab = 'MLE fit'; % 拟合图例
    else
        lab = 'MLE candidate';
    end
    legend(ax,{'Data',lab},'Location','best');
end
xlabel(ax,xl); ylabel(ax,'Probability');
end

function plot_ikeda(m,file)
% m：Ikeda结果；file：图片路径
f = figure('Visible','off','Color','w','Theme','light','Position',[100 100 1100 440]); % 图窗
tl = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact'); % 子图布局
ax = nexttile(tl,1); % 当前坐标轴
d = m.size_distribution; % 大小分布
loglog(ax,d.value,d.probability,'o','Color',[0 0.35 0.8],'MarkerSize',4);
hold(ax,'on');
c = m.fit; % 线性拟合
if c.valid
    x = c.xmin:c.xmax; % 拟合范围
    loglog(ax,x,10.^polyval([c.slope c.intercept],log10(x)), ...
        '-','Color',[0.8 0.25 0.1],'LineWidth',1.4);
    legend(ax,{'Data','OLS fit'},'Location','best');
end
title(ax,'A  Size distribution');
subtitle(ax,sprintf('k = %.3f; fit = [%g, %g]',c.slope,c.xmin,c.xmax));
xlabel(ax,'S (spikes)'); ylabel(ax,'P(S)');

ax = nexttile(tl,2);
d = m.deviation;
if ~isempty(d.S)
    h = bar(ax,d.S,[max(d.value,0) min(d.value,0)],1,'stacked','EdgeColor','none'); % 偏差柱形
    h(1).FaceColor = [0 0.35 0.8];
    h(2).FaceColor = [0.8 0.25 0.1];
else
    text(ax,0.5,0.5,'Insufficient fit range','Units','normalized','HorizontalAlignment','center');
end
hold(ax,'on');
yline(ax,0,'Color',[0.2 0.2 0.2]);
title(ax,'B  Deviation from fit');
subtitle(ax,sprintf('\\Delta Cr = %.4g',m.DeltaCr));
xlabel(ax,'S (spikes)'); ylabel(ax,'P_{emp} - P_{fit}');
title(tl,sprintf('Ikeda; interval = %.4g ms',m.interval_ms));
exportgraphics(f,file,'Resolution',180);
close(f);
end
