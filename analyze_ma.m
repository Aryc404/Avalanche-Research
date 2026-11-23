function out = analyze_ma(Espikes,dt_ms)
% Ma 阈值穿越法。
if nargin < 2, dt_ms = 1; end
bin_steps = round(40/dt_ms);
bin_count = floor(size(Espikes,2)/bin_steps);
binned = reshape(Espikes(:,1:bin_count*bin_steps),size(Espikes,1),bin_steps,bin_count);
active_cells = reshape(any(binned,2),size(Espikes,1),bin_count);
activity = sum(active_cells,1);
threshold = prctile(activity,35);
above = activity > threshold;
change = diff([false above false]);
start_bin = find(change == 1);
end_bin = find(change == -1)-1;
duration = end_bin-start_bin+1;
cumulative = [0 cumsum(activity)];
sizes = cumulative(end_bin+1)-cumulative(start_bin);
complete = start_bin > 1 & end_bin < bin_count;

out.bin_ms = bin_steps*dt_ms;
out.time_s = ((1:bin_count)-0.5)*out.bin_ms/1000;
out.activity = activity;
out.threshold = threshold;
out.threshold_percentile = 35;
out.active = above;
out.recorded_ms = bin_count*out.bin_ms;
out.discarded_tail_ms = (size(Espikes,2)-bin_count*bin_steps)*dt_ms;
out.events = struct('start_bin',start_bin,'end_bin',end_bin, ...
    'start_ms',(start_bin-1)*out.bin_ms,'end_ms',end_bin*out.bin_ms, ...
    'S',sizes,'D_bins',duration,'D_ms',duration*out.bin_ms,'complete',complete);
out.complete_count = nnz(complete);
out.censored_count = nnz(~complete);
out.size_distribution = avalanche_distribution(sizes(complete));
out.duration_distribution = avalanche_distribution(duration(complete));
out.tau = out.size_distribution.fit.exponent;
out.alpha_D = out.duration_distribution.fit.exponent;
out.scaling = size_duration_scaling(sizes(complete),duration(complete), ...
    out.size_distribution.fit,out.duration_distribution.fit,out.bin_ms);
out.DCC = out.scaling.DCC;
end

function distribution = avalanche_distribution(data)
data = double(data(:));
[value,~,index] = unique(data);
if isempty(data)
    probability = [];
else
    probability = accumarray(index,1)/numel(data);
end
distribution.value = value;
distribution.probability = probability;
distribution.fit = truncated_power_law(data);
fit = distribution.fit;
selected = value > 0 & probability > 0;
if isfinite(fit.xmin)
    selected = selected & value >= fit.xmin & value <= fit.xmax;
end
distribution.ols = struct('slope',NaN,'intercept',NaN,'used',selected);
if nnz(selected) >= 2
    line = polyfit(log(value(selected)),log(probability(selected)),1);
    distribution.ols.slope = line(1);
    distribution.ols.intercept = line(2);
end
end

function fit = truncated_power_law(data)
fit = struct('support',[],'probability',[],'n_used',0,'n_total',numel(data), ...
    'xmin',NaN,'xmax',NaN,'exponent',NaN,'ks',NaN,'ks_limit',NaN, ...
    'valid',false);
values = unique(data);
if numel(data) < 50 || numel(values) < 5, return; end
candidate_index = unique(round(linspace(1,numel(values)-4,min(20,numel(values)-4))));
best_ks = Inf;
for candidate = candidate_index
    xmin = values(candidate);
    tail = data(data >= xmin);
    if numel(tail) < 50 || numel(unique(tail)) < 5, continue; end
    support = (xmin:max(data))';
    relative_log = log(support/xmin);
    log_sum = sum(log(tail/xmin));
    n = numel(tail);
    loss = @(exponent) exponent*log_sum+n*log(sum(exp(-exponent*relative_log)));
    exponent = fminbnd(loss,0,8);
    probability = exp(-exponent*relative_log);
    probability = probability/sum(probability);
    observed = accumarray(tail-xmin+1,1,[numel(support) 1])/n;
    ks = max(abs(cumsum(observed)-cumsum(probability)));
    if ks < best_ks
        best_ks = ks;
        fit.support = support;
        fit.probability = probability;
        fit.n_used = n;
        fit.xmin = xmin;
        fit.xmax = support(end);
        fit.exponent = exponent;
        fit.ks = ks;
        fit.ks_limit = 1/sqrt(n);
    end
end
fit.valid = isfinite(fit.exponent) && fit.ks < fit.ks_limit && ...
    fit.exponent >= 0.01 && fit.exponent <= 7.99;
end

function scaling = size_duration_scaling(sizes,duration,size_fit,duration_fit,bin_ms)
scaling = struct('duration_bins',[],'duration_ms',[],'mean_size',[],'count',[], ...
    'used',false(0,1),'intercept',NaN,'beta_fit',NaN,'beta_pred',NaN, ...
    'DCC',NaN,'valid',false);
if isempty(sizes), return; end
[D,~,group] = unique(duration(:));
means = accumarray(group,sizes(:),[],@mean);
counts = accumarray(group,1);
used = counts >= 5;
if isfinite(duration_fit.xmin)
    used = used & D >= duration_fit.xmin & D <= duration_fit.xmax;
end
if isfinite(size_fit.xmin)
    used = used & means >= size_fit.xmin & means <= size_fit.xmax;
end
scaling.duration_bins = D;
scaling.duration_ms = D*bin_ms;
scaling.mean_size = means;
scaling.count = counts;
scaling.used = used;
if nnz(used) >= 3
    line = polyfit(log(D(used)),log(means(used)),1);
    scaling.beta_fit = line(1);
    scaling.intercept = line(2);
end
tau = size_fit.exponent;
alpha_D = duration_fit.exponent;
if isfinite(tau) && isfinite(alpha_D) && abs(tau-1) > 1e-3
    scaling.beta_pred = (alpha_D-1)/(tau-1);
end
if nnz(used) >= 3 && size_fit.valid && duration_fit.valid && isfinite(scaling.beta_pred)
    scaling.DCC = abs(scaling.beta_fit-scaling.beta_pred);
    scaling.valid = true;
end
end
