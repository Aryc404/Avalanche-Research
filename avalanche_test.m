function result = avalanche_test()
% 冻结 AD 网络，连续呈现 100 个 patch。

%% 读取网络
experiment_dir = fileparts(mfilename('fullpath'));
model_dir = fullfile(experiment_dir,'..','Ruslim network','bio-spiking-v1-learning-main');
checkpoint_file = fullfile(experiment_dir,'pcb_shuffle_ad_paper_cn1_run01_batch1200.mat');
loaded = load(checkpoint_file,'checkpoint');
checkpoint = loaded.checkpoint;
state = checkpoint.state;

n_patches = 100;
patch_steps = 400;
dt_ms = 1;
dt = dt_ms/1000;
seed = 20261123;
N_E = size(state.W_EE,1);
N_I = size(state.W_II,1);
N_X = size(state.W_EX,2)/2;
total_steps = n_patches*patch_steps;

old_rng = rng;
old_path = path;
session_cleanup = onCleanup(@() restore_session(old_rng,old_path));
addpath(model_dir);
rng(seed,'twister');

protocol = struct('checkpoint_file',checkpoint_file,'checkpoint_batch',checkpoint.batch, ...
    'seed',seed,'n_patches',n_patches,'patch_ms',patch_steps*dt_ms, ...
    'dt_ms',dt_ms,'total_ms',total_steps*dt_ms);

%% 抽取 patch
image_data = load(fullfile(experiment_dir,'VanHateren_DoG_small.mat'),'IMAGES_DoG');
patch_rates_Hz = generate_image_patches(image_data.IMAGES_DoG,N_X,n_patches);
patch_rates_Hz = min(patch_rates_Hz,100);

%% 网络仿真
V_E = zeros(N_E,1); V_I = zeros(N_I,1);
exp_m = exp(-dt/10e-3);
spikes_E = false(N_E,total_steps);
spikes_I = false(N_I,total_steps);
for patch = 1:n_patches
    input_probability = patch_rates_Hz(:,patch)*dt;
    for local_step = 1:patch_steps
        step = (patch-1)*patch_steps+local_step;
        S_X = sparse(double(rand(2*N_X,1)<input_probability));
        S_E = V_E>state.theta_E;
        S_I = V_I>state.theta_I;
        spikes_E(:,step) = S_E;
        spikes_I(:,step) = S_I;
        V_E(S_E) = 0; V_I(S_I) = 0;
        E = sparse(double(S_E)); I = sparse(double(S_I));
        V_E = V_E*exp_m+state.W_EX*S_X+state.W_EE*E-state.W_EI*I+randn(N_E,1);
        V_I = V_I*exp_m+state.W_IX*S_X+state.W_IE*E-state.W_II*I+randn(N_I,1);
        V_E = max(V_E,-10); V_I = max(V_I,-10);
    end
end

%% 保存与统计
output_dir = fullfile(experiment_dir,'results','avalanche_test', ...
    char(datetime('now','Format','yyyyMMdd_HHmmss_SSS')));
mkdir(output_dir);
result = struct('protocol',protocol,'frozen_state',state, ...
    'patch_rates_Hz',patch_rates_Hz,'patch_start_step',1:patch_steps:total_steps, ...
    'spikes_E',spikes_E,'spikes_I',spikes_I);
result.output_file = fullfile(output_dir,'result.mat');
result.stepp_figure = fullfile(output_dir,'stepp_summary.png');
result.ma_figure = fullfile(output_dir,'ma_summary.png');
save(result.output_file,'result','-v7.3');

result.stepp = analyze_stepp(spikes_E,state.W_EE,dt_ms);
result.ma = analyze_ma(spikes_E,dt_ms);
save(result.output_file,'result','-v7.3');

%% 绘图
draw_stepp(result);
draw_ma(result);
fprintf('%s\n',output_dir);
end

function restore_session(old_rng,old_path)
rng(old_rng);
path(old_path);
end

function draw_stepp(result)
s = result.stepp;
f = figure('Visible','off','Color','w','Theme','light','Position',[100 100 1550 950]);
t = tiledlayout(f,2,3,'TileSpacing','compact','Padding','compact');

ax = nexttile(t,1);
raster_end_ms = min(2000,result.protocol.total_ms);
[ids,steps] = find(result.spikes_E(:,1:round(raster_end_ms/result.protocol.dt_ms)));
plot(ax,(steps-1)*result.protocol.dt_ms/1000,ids,'.','Color',[0 0.35 0.8],'MarkerSize',3);
hold(ax,'on');
for boundary = result.protocol.patch_ms:result.protocol.patch_ms:raster_end_ms-1
    xline(ax,boundary/1000,':','Color',[0.7 0.7 0.7]);
end
title(ax,'A  E spike raster; dotted lines = patch changes');
xlabel(ax,'Time (s)'); ylabel(ax,'E neuron ID');
xlim(ax,[0 raster_end_ms/1000]); ylim(ax,[0.5 size(result.spikes_E,1)+0.5]);

ax = nexttile(t,2);
p = s.preview;
edges = double(p.parent_edges);
if ~isempty(edges)
    x = [p.time_ms(edges(:,1))';p.time_ms(edges(:,2))';NaN(1,size(edges,1))];
    y = [double(p.neuron_id(edges(:,1)))';double(p.neuron_id(edges(:,2)))';NaN(1,size(edges,1))];
    plot(ax,x(:),y(:),'-','Color',[0.8 0.8 0.8]);
    hold(ax,'on');
end
is_root = false(numel(p.neuron_id),1);
label_count = cellfun(@numel,p.memberships);
for k = 1:numel(is_root)
    labels = p.memberships{k};
    if isscalar(labels)
        is_root(k) = s.tracking.start_step(labels)==p.step_index(k) && ...
            s.tracking.start_neuron(labels)==p.neuron_id(k);
    end
end
hold(ax,'on');
h1 = plot(ax,p.time_ms(is_root),double(p.neuron_id(is_root)),'s', ...
    'Color',[0.8 0.25 0.1],'MarkerFaceColor',[0.8 0.25 0.1],'MarkerSize',4);
single = ~is_root & label_count==1;
h2 = plot(ax,p.time_ms(single),double(p.neuron_id(single)),'.','Color',[0 0.35 0.8],'MarkerSize',8);
shared = label_count>1;
h3 = plot(ax,p.time_ms(shared),double(p.neuron_id(shared)),'o','Color',[0.15 0.15 0.15],'MarkerSize',5);
title(ax,'B  Connection tracking: first 100 ms, up to 200 spikes');
xlabel(ax,'Time (ms)'); ylabel(ax,'E neuron ID');
xlim(ax,[0 100]); ylim(ax,[0.5 size(result.spikes_E,1)+0.5]);
legend(ax,[h1 h2 h3],{'Root','Inherited','Shared by roots'},'Location','best');

ax = nexttile(t,3);
d = s.size_distribution;
loglog(ax,d.S,d.P,'o','Color',[0 0.35 0.8],'MarkerSize',4);
hold(ax,'on');
if d.fit_valid
    loglog(ax,d.S,10.^polyval([d.slope d.intercept],log10(d.S)), ...
        '-','Color',[0.8 0.25 0.1],'LineWidth',1.4);
    legend(ax,{'Observed','Descriptive OLS'},'Location','best');
end
title(ax,{'C  Avalanche size distribution', ...
    sprintf('Complete events: %d; k = %.3f',s.tracking.complete_count,d.slope)});
xlabel(ax,'S (spikes per avalanche)'); ylabel(ax,'P(S)');
if isempty(d.S)
    text(ax,0.5,0.5,'No complete avalanches','Units','normalized','HorizontalAlignment','center');
end

ax = nexttile(t,4);
b = s.branching;
plot(ax,b.time_ms/1000,b.value,'.','Color',[0.75 0.75 0.75],'MarkerSize',2);
hold(ax,'on');
plot(ax,b.time_ms/1000,movmean(b.value,1000,'omitnan'), ...
    'Color',[0 0.35 0.8],'LineWidth',1.3);
yline(ax,1,'--','Color',[0.8 0.25 0.1]);
title(ax,{'D  Network branching ratio',sprintf('Valid time points: %d; mean = %.3f',b.valid_count,b.mean)});
xlabel(ax,'Time (s)'); ylabel(ax,'s(n)');
legend(ax,{'Raw','1-s moving mean','s = 1'},'Location','best');
xlim(ax,[0 result.protocol.total_ms/1000]);

ax = nexttile(t,5);
hold(ax,'on');
for k = 1:numel(s.dfa.F)
    selected = s.dfa.F{k}>0;
    loglog(ax,s.dfa.scales{k}(selected),s.dfa.F{k}(selected), ...
        'Color',[0.8 0.8 0.8],'LineWidth',0.5);
end
example = find(isfinite(s.dfa.alpha),1);
if ~isempty(example)
    l = s.dfa.scales{example}; F = s.dfa.F{example};
    loglog(ax,l,F,'o-','Color',[0 0.35 0.8],'MarkerSize',3);
    loglog(ax,l,10.^polyval([s.dfa.alpha(example) s.dfa.intercept(example)],log10(l)), ...
        '--','Color',[0.8 0.25 0.1]);
end
set(ax,'XScale','log','YScale','log');
title(ax,'E  ISI-DFA; gray = individual E neurons');
xlabel(ax,'l (ISI samples per segment)'); ylabel(ax,'F(l) (ms)');

ax = nexttile(t,6);
valid = isfinite(s.dfa.alpha);
values = s.dfa.alpha(valid);
if ~isempty(values)
    lo = min(values); hi = max(values);
    if lo==hi, lo=lo-0.1; hi=hi+0.1; end
    bins = linspace(lo,hi,21);
    histogram(ax,s.dfa.alpha(valid & s.dfa.limited_span),bins, ...
        'DisplayStyle','stairs','EdgeColor',[0.8 0.45 0.1],'LineWidth',1.2);
    hold(ax,'on');
    histogram(ax,s.dfa.alpha(valid & ~s.dfa.limited_span),bins, ...
        'DisplayStyle','stairs','EdgeColor',[0 0.35 0.8],'LineWidth',1.2);
    legend(ax,{'Scale span < 1 decade','Scale span >= 1 decade'},'Location','best');
else
    text(ax,0.5,0.5,'Too few ISIs / usable scales','Units','normalized','HorizontalAlignment','center');
end
title(ax,{'F  DFA exponents (descriptive)', ...
    sprintf('%d / %d fits; %d span >= 1 decade',s.dfa.valid_count, ...
    size(result.spikes_E,1),s.dfa.adequate_span_count)});
xlabel(ax,'\alpha_{DFA}'); ylabel(ax,'Number of E neurons');
title(t,'Stepp framework: frozen E network; pre/post lags 2-4 ms');
exportgraphics(f,result.stepp_figure,'Resolution',180);
close(f);
end

function draw_ma(result)
m = result.ma;
f = figure('Visible','off','Color','w','Theme','light','Position',[100 100 1350 950]);
t = tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');

ax = nexttile(t,1);
hold(ax,'on');
shown_end = min(4,m.recorded_ms/1000);
top = max([m.activity m.threshold 1])*1.15;
for k = 1:numel(m.events.S)
    left = m.events.start_ms(k)/1000;
    right = m.events.end_ms(k)/1000;
    if left < shown_end
        patch(ax,[left right right left],[0 0 top top],[0 0.35 0.8], ...
            'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off');
    end
end
h1 = plot(ax,m.time_s,m.activity,'Color',[0.15 0.15 0.15],'LineWidth',1);
h2 = yline(ax,m.threshold,'--','Color',[0 0.35 0.8],'LineWidth',1.2);
title(ax,{'A  Active E neurons in 40-ms bins (first 4 s)', ...
    sprintf('35th-percentile threshold = %.3g; shaded = A > threshold',m.threshold)});
xlabel(ax,'Time (s)'); ylabel(ax,'A (active E neurons)');
xlim(ax,[0 shown_end]); ylim(ax,[0 top]);
legend(ax,[h1 h2],{'Network activity','Threshold'},'Location','best');

ax = nexttile(t,2);
draw_ma_distribution(ax,m.size_distribution,'S (total active-neuron counts)');
title(ax,{'B  Avalanche size distribution',sprintf('τ = %.3f; complete events: %d',m.tau,m.complete_count)});

ax = nexttile(t,3);
draw_ma_distribution(ax,m.duration_distribution,'D (40-ms bins)');
title(ax,{'C  Avalanche duration distribution',sprintf('α_D = %.3f',m.alpha_D)});

ax = nexttile(t,4);
c = m.scaling;
loglog(ax,c.duration_bins,c.mean_size,'o','Color',[0.6 0.6 0.6],'MarkerSize',5);
hold(ax,'on');
if any(c.used)
    loglog(ax,c.duration_bins(c.used),c.mean_size(c.used),'o', ...
        'Color',[0 0.35 0.8],'MarkerFaceColor',[0 0.35 0.8],'MarkerSize',4);
end
if isfinite(c.beta_fit)
    x = logspace(log10(min(c.duration_bins(c.used))),log10(max(c.duration_bins(c.used))),100);
    hfit = loglog(ax,x,exp(c.intercept)*x.^c.beta_fit,'-', ...
        'Color',[0 0.35 0.8],'LineWidth',1.4);
    if isfinite(c.beta_pred)
        pivot = exp(mean(log(c.duration_bins(c.used))));
        anchor = exp(c.intercept)*pivot^c.beta_fit;
        hpred = loglog(ax,x,anchor*(x/pivot).^c.beta_pred,'--', ...
            'Color',[0.8 0.25 0.1],'LineWidth',1.4);
        if c.valid
            labels = {'Fitted slope','Predicted slope'};
        else
            labels = {'Candidate fit','Candidate prediction'};
        end
        legend(ax,[hfit hpred],labels,'Location','best');
    end
end
if c.valid
    summary = sprintf('β_{fit}=%.3f; β_{pred}=%.3f; DCC=%.3f',c.beta_fit,c.beta_pred,c.DCC);
else
    summary = sprintf('DCC unavailable; β_{fit}=%.3f; β_{pred}=%.3f',c.beta_fit,c.beta_pred);
end
title(ax,{'D  Mean size versus duration',summary});
xlabel(ax,'D (40-ms bins)'); ylabel(ax,'Mean S at fixed D');
if isempty(c.duration_bins)
    text(ax,0.5,0.5,'No complete avalanches','Units','normalized','HorizontalAlignment','center');
end
title(t,'Ma framework: 40-ms bins; 35th percentile; S = total event activity');
exportgraphics(f,result.ma_figure,'Resolution',180);
close(f);
end

function draw_ma_distribution(ax,distribution,x_label)
loglog(ax,distribution.value,distribution.probability,'o', ...
    'Color',[0 0.35 0.8],'MarkerSize',4);
hold(ax,'on');
fit = distribution.fit;
if ~isempty(fit.support)
    loglog(ax,fit.support,fit.probability*fit.n_used/fit.n_total, ...
        '-','Color',[0.8 0.25 0.1],'LineWidth',1.4);
    if fit.valid
        label = sprintf('MLE [%g, %g]',fit.xmin,fit.xmax);
    else
        label = 'MLE candidate (screen not passed)';
    end
    legend(ax,{'Observed',label},'Location','best');
end
xlabel(ax,x_label); ylabel(ax,'Probability per discrete value');
if isempty(distribution.value)
    text(ax,0.5,0.5,'No complete avalanches','Units','normalized','HorizontalAlignment','center');
end
end
