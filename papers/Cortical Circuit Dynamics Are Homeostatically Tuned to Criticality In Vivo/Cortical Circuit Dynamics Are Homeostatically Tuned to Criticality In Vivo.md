# 1.文中使用的网络模型：
**Binary probabilistic network**
二值随机网络：每个 neuron 每个 time step 只有 spike=1 或 silent=0 两种状态
其中 **E神经元 有 4750个 I神经元 有 250个**
神经元之间并非完全连接 而是在模型初始化时令神经元间随机连接，其中概率为：

**${} E\rightarrow E:3\%,{}$ ${} E\rightarrow I:3\%,{}$ ${}I\rightarrow E:1\%-60\%{}$**

每条连接的权重不是电导，而是：
${}P_{ij} {}$ (j在当前时间步 firing，使 i下一时间步 firing 的 transition probability”。抑制连接允许取值为负)

神经元更新规则:
$$
 X_i(t+1) = Q\left[ (1-h_i) \sum_jP_{ij}X_j(t) + h_i-\xi_i \right]
$$
其中： 
${}Q{}$为阶跃函数，括号中大于零就输出 1，否则输出 0。
${}X_{i}{}$用于形容i神经元的放电状态 $X_i(t)=1$：第${}t{}$步放电。$X_i(t)=0$：第${}t{}$步不放电。
${}P_{ij}{}$为从 神经元j 到 神经元i 的放电转移概率，也即神经元 j在当前时刻发放一个 spike 后，使神经元 i 在下一个 time step 发 spike 的作用强度（其中抑制连接则取负值)
${}h_{i}(t){}$为 ***网络外部输入*** 使 神经元独立 放电的概率
${}\xi_i(t){}$为从 $[0, 1]$ 均匀随机抽出的数

一些对于公式的理解:
	${}\sum_jP_{ij}X_j(t){}$ 为 t时刻网络回路中的驱动          ${}h_i(t){}$ 为 t时刻外界的驱动
	而${}\sum_jP_{ij}X_j(t){}$前面有一个系数${}(1-h_i) {}$的原因:
		可以把${}r_{i}=\sum_jP_{ij}X_j(t){}$视作神经元因为网络回路而放电的概率
		${}h_{i}{}$视作神经元因外部输入而放电的概率
		此时由于两者是独立的，因此那么 神经元 **完全没有 firing** 的概率是：$$(1-r_i)(1-h_i).$$展开后就得到${}h_i+(1-h_i)r_i{}$ 
	而${}\xi_{i}{}$作为随机变量，正好能反应这一随机概率：
		因为：${}\xi_i\sim U(0,1).{}$而 神经元 firing 的条件是:${} p_i-\xi_i>0{}$
		也就是：${}\xi_i<p_i{}$ 均匀随机数落在区间${}[0,p_i]{}$ 里的概率正好就是：$p_i$
	外部输入："$h_i(t)$ was modeled as a binary Poisson process followed by smoothing with a Gaussian filter."
		其中binary Poisson process指的是一个二值泊松过程，也即每个 time step 随机决定有没有一个外部事件发生，可以理解为一个随机的0/1序列
	如：$[0,0,0,1,0,0,1,0,0,0,0,1,...]$ 
		而Gaussian smoothing指的是使用高斯滤波将瞬间出现的事件向前后时间扩展开，使输入随时间平滑变化

网络中的E神经元仅有10%（也即475个）能接受到外部的输入，对于I神经元群体作者则是不断修改能接收到外部输入的比例(20%,40%,80%),为此寻找使得网络达到最接近critical regime的组合

网络中 ${}P_{ij}{}$ 的更新规则：
	**①. STDP（Spike-Timing-Dependent Plasticity，脉冲时序依赖可塑性）**
	（论文中仅给出了 E → E 及 I → E 的STDP)
	1. E → E excitatory STDP:
	$$
 {\delta P_{EE}^{ij}(t) = K_{\mathrm{STDP}} \left[ X_i^E(t)X_j^E(t-1) - X_j^E(t)X_i^E(t-1) \right] } 
	$$
	其中：
	 ${}P_{EE}^{ij}{}$ 为 E 神经元 j → E 神经元 i 的放电转移概率
	${}K_{STDP}{}$为 学习速率
	$X_i^E(t)$：E神经元 i 在 t时刻 的放电情况
		Case A: pre 先 post 后
		如${}X_j^E(t-1)=1,\qquad X_i^E(t)=1{}$ 则 ${} P_{EE}^{ij}\uparrow{}$ 
		也就是$$\boxed{\text{pre}\rightarrow\text{post}\quad\Rightarrow\quad LTP}$$
		Case B: post 先 pre 后
		如${}X_i^E(t-1)=1,\qquad X_j^E(t)=1{}$ 则${} P_{EE}^{ij}\downarrow{}$ 
		也就是$$\boxed{\text{post}\rightarrow\text{pre}\quad\Rightarrow\quad LTD} $$
	2. I → E inhibitory STDP：$$
	{ \delta P_{EI}^{ij}(t) = -K_{\rm inh}X_j^I(t-1) \left[ 1-X_i^E(t) \left(1+\frac{1}{\mu_I}\right) \right] } $$
	其中：
	${}\mu_{I}{}$为 baseline 时所有 I neurons 的平均 firing rate
	${}K_{inh}{}$为 Learning rate
	**②.SS（Synaptic Scaling，突触缩放）**
	$$
	{ \delta P_{ij}^{SS}(t) = K_{SS} \left[ F_i^{ref} - F_i^{hist}(t) \right] } 
	$$
	其中：
	$F_i^{ref}$：第 i 个神经元 的 目标 firing rate
	$F_i^{hist}(t)$：第 i 个 neuron 最近15000个time steps 的 firing rate
	${}K_{SS}{}$:SS的 learning rate

# 2.文中定义的网络的几种状态

根据模型中使用的连接矩阵P 的最大特征值 λ 来描述网络的状态
①: λ ＜1: subcritical regime  活动传播会逐渐衰减
②: λ ≈ 1: critical regime  平均来看，活动既不系统衰减，也不持续放大,处于临界状态
③: λ > 1: supercritical regime 活动传播会逐渐变大

# 3. 通过网络模型进行的实验 以及 得出的结论

实验流程：

Step 1：初始化网络。先建立4750 E + 250 I 的 PIF网络，确定好E→E 与 I→E 的连接情况(**${} E\rightarrow E:3\%,{}$ ${} E\rightarrow I:3\%,{}$ ${}I\rightarrow E:1\%-60\%{}$**) ，然后开始确定${}P_{ij}{}$的取值，对于没有连接的神经元相应${}P_{ij}{}$取值为0，对于存在的连接，${}P_{ij}{}$从某个均匀分布中得到(原文未给出相应分布)，最后将整个矩阵统一缩放，使它的最大特征值达到预设值 1.02 

Step 2: 让网络在正常输入的情况下运行一段时间，${}P_{ij}{}$ 的权重被固定 ，STDP与SS两大机制尚不开启，先让网络稳定下来(${}t_{0}\rightarrow t_{1}{}$)

Step 3: 
开启STDP + SS ，此时 Inhibitory STDP的${}\mu_{I}{}$通过计算前面Step 2时 I神经元的平均放电率得到(${}t_{1}\rightarrow t_{2}{}$)

Step 4: 减少external input ${}h_{i}(t){}$ ,让它呈指数型下降，最后接近 0.5${}h_{0}{}$ 。用于模拟MD（Monocular Deprivation，单眼剥夺）.(${}t_{2}\rightarrow t_{end}{}$)

实验结论：
在进行MD实验后，I 神经元的 放电率首先下降，之后 E 神经元的放电率也下降，同时网络network的 criticality被破坏，λ<1，网络暂时变subcritical
之后STDP 与 SS 开始重新调整权重，三者均回归baseline水平
同时，作者又分别将E神经元的SS与I神经元的SS去除，观察会发生什么。

**关闭E神经元的SS**:E 神经元 与 I神经元的 放电率不断放大 出现 runaway firing 同时λ > 1
因此得出结论 **E的SS主要用于防止firing rate runaway**

**关闭I神经元的SS:** E与I神经元的放电率仍然相对稳定，但是λ无法回到critical regime
因此得出结论 **I的SS主要用于稳定回复 critical regime**
![[Pasted image 20261023174514.png]]其中successful model 指的是MD实验下，满足以下几个条件的模型:
①. E与I的放电率明显下降至 baseline的75% 以下
②. 网络明显偏离 critical regime，要满足${}|\lambda-1.02|>0.05 {}$ 
③. E与I的放电率最后要恢复接近baseline
④. 网络要重新回到 near-critical zone

model主要修改的参数为:
1.${} \frac{N_I}{N} {}$  I 神经元占整个网络的比例
2.I→E connectivity ，也就是每个I 神经元 能连接多少 E 神经元
3.有多少 I 神经元 接收 external input，一共测试了 20%, 40%, 80% 三种情况
以上可以总结为 inhibitory architecture
同时作者做了第二轮测试：选了一个“离成功区域很近但失败”的 inhibitory architecture，固定住它，再去改：
- **SS gain**
- **STDP gain**
- **E→E connectivity**
结果发现仍旧无法达成满足successful model的条件，因此作者得出结论：
	决定模型能否成功的关键，是 inhibitory architecture 本身

# 4. 文中所做的大鼠实验
**实验流程:**

**Step 1:植入电极** 。在 P21（出生后第21天），给 大鼠 两侧的 monocular V1（单眼区初级视觉皮层）植入 16-channel microelectrode array，电极跨越 6 个 cortical layers。术后恢复两天。

**Step 2:连续记录.** P24 开始连续记录，小鼠两只眼正常、没有视觉剥夺时的 V1 activity，记录 24 h作为正式的 baseline ，主要记录**RSU（Regular-Spiking Unit，规则放电单元，主要对应 excitatory pyramidal neurons）** 的 spikes。

Step 3: MD实验。baseline 结束后，把小鼠的一只眼缝起来。这样一侧 V1 对应的视觉输入被明显削弱；另一侧 hemisphere 作为 control

Step4:继续连续记录。P27 起正式算作 MD 开始，之后继续连续记录 132 h

**结果处理:**
记录时把4h作为一个time window,把4h内的时间分成 40 ms bins，
同时记录 每个 神经元 在 每个 bin 内的放电情况 0/1（无论该神经元在40ms内放电几次，只要放电就记为1，没有放电就记作0)
之后 每个bin内所有神经元 相加得到 该bin下的network activity；

记录即可得到 time window 下 的network activity随bin变化的序列，记作**A(t)**

计算 avalanche 的 *size (S)、duration (D)*
* 如何定义一次avalanche：
	用 network activity 的 35th percentile 定义 **avalanche threshold；**
	A(t)从 threshold 下方穿到上方时，一个 avalanche 开始；从上方掉回下方时,avalanche 结束
	而Size 则是avalanche 期间累计的 network activity
	Duration 则是avalanche的持续时长
	
 拟合 $\tau,\alpha,\beta$：
 
 通过得到的avalanche数据，分别拟合${} P(S)\sim S^{-\tau} {}$与${}P(D)\sim D^{-\alpha}{}$ (由此得到${}\tau{}$和${}\alpha{}$)
 **在临界系统中，其平均大小与持续时间有如下关系：**
 $$
\langle S\rangle(D)\sim D^\beta 
 $$
 其中$\langle S\rangle(D)$ 指的是S关于D的函数
 而我们能推导出结论：${} \beta_{pred}=\frac{\alpha-1}{\tau-1}{}$ 
	 推导过程如下：
	 由于 ${}S\sim D^\beta{}$ ,同时我们得知 $P(S)\sim S^{-\tau}$ 因此可以得到 ${}P(S(D))\sim (D^\beta)^{-\tau} =D^{-\beta\tau}{}$
	补全等式，则可得到:${}S=cD^\beta{}$  其中c为比例常数
	因此：${}\frac{\mathrm dS}{\mathrm dD} =c\beta D^{\beta-1}{}$
	由于我们有结论： $\langle S\rangle(D)\sim D^\beta$ ，$S=S(D)$ .因此持续时间落在一个很小区间 $[D,D+\mathrm dD]$ 内的概率，必须等于对应的雪崩大小落在 $[S,S+\mathrm dS]$ 内的概率。
	也即${P(D)dD = P(S)\,\mathrm dS}$，推导可得到 ${ P(D)=P(S(D)) \left|\frac{\mathrm dS}{\mathrm dD}\right| }$ 
	$\begin{aligned} P(D) &\sim D^{-\beta\tau}D^{\beta-1}\\ &=D^{-\beta\tau+\beta-1}\\ &=D^{-[1+\beta(\tau-1)]} \end{aligned}$  
	由于$P(D)\sim D^{-\alpha}$  因此 $\alpha=1+\beta(\tau-1)$ 
	所有我们有结论$$\boxed{\beta_{\mathrm{pred}} =\frac{\alpha-1}{\tau-1}}$$
同时我们可以通过拟合实验数据，拟合得到$\beta_{fit}$。因为我们之前得到$\langle S\rangle(D)=CD^\beta$ 
两边取对数可得到: $\log\langle S\rangle =\log C+\beta\log D$,在对数坐标系上使用直线拟合这些数据点，通过这个直线的斜率得出$\beta_{fit}$ 


**总结一下**
 $\tau$ 用于描述不同大小的雪崩事件出现的概率如何随大小变化。
 $\alpha$ 用于描述不同持续时间的雪崩事件出现的概率如何随持续时间变化。
 $\beta_{pred}$ (预测指数) 根据雪崩大小分布指数 $\tau$ 和持续时间分布指数 $\alpha$。
 $\beta_{fit}$(拟合指数) 直接根据实验数据，拟合平均雪崩大小S与持续时间D的关系 
 
 最后得到 DCC (Deviation from Criticality Coefficient)$$ \mathrm{DCC} = |\beta_{\mathrm{fit}}-\beta_{\mathrm{pred}| }$$
 若DCC接近0，则说明数据符合临界状态下的情况。
 若DCC较大，则说明数据偏离临界状态下的情况。
**文中定义 DCC ＜ 0.2时，网络接近临界状态 而 DCC > 0.2时，网络偏离临界状态**
在大鼠实验中，baseline情况下DCC始终小于0.2。而在大鼠MD后，DCC一度上升至0.6
